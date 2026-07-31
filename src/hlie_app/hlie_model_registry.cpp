#include "hlie_app/hlie_model_registry.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <utility>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {

json llama_params_summary(const HLIEModelInfo& model_info) {
    json summary = json::object();
    if (!model_info.has_effective_llama_params()) {
        return summary;
    }
    const auto params = model_info.get_effective_llama_params().to_json();
    for (const auto& [arg, name] : std::map<std::string, std::string>{
             {"--ctx-size", "ctx_size"}, {"--parallel", "parallel"},
             {"--batch-size", "batch_size"}, {"--ubatch-size", "ubatch_size"}}) {
        if (const auto it = params.find(arg); it != params.end()) {
            summary[name] = *it;
        }
    }
    return summary;
}

} // namespace

std::string HLIEModelRegistry::normalize_model_name(const std::string& name) {
    const auto first = std::find_if_not(name.begin(), name.end(), [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(name.rbegin(), name.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) {
        return {};
    }

    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized;
}

HLIEModelRegistry::ModelInstance* HLIEModelRegistry::find_instance_locked(const std::string& model_name) {
    return const_cast<ModelInstance*>(std::as_const(*this).find_instance_locked(model_name));
}

const HLIEModelRegistry::ModelInstance* HLIEModelRegistry::find_instance_locked(const std::string& model_name) const {
    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        return &it->second;
    }

    const std::string normalized_name = normalize_model_name(model_name);
    if (normalized_name.empty()) {
        return nullptr;
    }

    const ModelInstance* matched = nullptr;
    int matched_rank = 2;
    int matched_state_rank = 2;
    bool ambiguous = false;
    for (const auto& [name, inst] : model_instances_) {
        const bool key_matched = normalize_model_name(name) == normalized_name;
        const bool model_name_matched = normalize_model_name(inst.model_info.get_model_name()) == normalized_name;
        const bool meta_name_matched = normalize_model_name(inst.meta.name) == normalized_name;
        if (key_matched || model_name_matched || meta_name_matched) {
            const int rank = key_matched ? 0 : 1;
            const int state_rank = inst.meta.status == SERVER_MODEL_STATUS_LOADED ? 0 : 1;
            if (!matched || rank < matched_rank || (rank == matched_rank && state_rank < matched_state_rank)) {
                matched = &inst;
                matched_rank = rank;
                matched_state_rank = state_rank;
                ambiguous = false;
            } else if (rank == matched_rank && state_rank == matched_state_rank) {
                ambiguous = true;
            }
        }
    }

    return ambiguous ? nullptr : matched;
}

HLIEModelRegistry::ModelInstance HLIEModelRegistry::build_model_instance(HLIEModelInfo& model_info,
                                                                         const int estimated_mem_mb) {
    ModelInstance model_instance;
    const std::string model_key = HLIEModelInfo::build_model_key(model_info);

    server_model_meta meta;
    meta.name = model_key;
    model_info.model_key_ = model_key;

    model_instance.key = model_key;
    model_instance.path = model_info.get_model_path();
    model_instance.path_mmproj = model_info.get_mm_proj_path();
    model_instance.meta = std::move(meta);
    model_instance.model_info = model_info;
    model_instance.estimated_mem_mb = std::max(0, estimated_mem_mb);
    return model_instance;
}

void HLIEModelRegistry::register_model(HLIEModelInfo& model_info, const int estimated_mem_mb) {
    auto model_instance = build_model_instance(model_info, estimated_mem_mb);
    const std::string model_key = model_instance.key;

    upsert(std::move(model_instance));

    const auto registered = get_instance_snapshot(model_key);
    if (registered) {
        LOG_INFO("Registered model instance: key={}, model_name={}, meta_name={}, model_path={}, mmproj={}",
                 json(model_key).dump(), json(registered->model_info.get_model_name()).dump(),
                 json(registered->meta.name).dump(), json(registered->path).dump(),
                 json(registered->path_mmproj).dump());
    }
}

bool HLIEModelRegistry::register_model_if_absent(HLIEModelInfo& model_info, const int estimated_mem_mb) {
    auto model_instance = build_model_instance(model_info, estimated_mem_mb);
    const std::string model_key = model_instance.key;
    normalize_model_instance(model_instance);

    {
        std::lock_guard lk(mutex_);
        const auto [_, inserted] = model_instances_.try_emplace(model_key, std::move(model_instance));
        if (!inserted) {
            return false;
        }
    }

    const auto registered = get_instance_snapshot(model_key);
    if (registered) {
        LOG_INFO("Discovered new model instance: key={}, model_name={}, model_path={}, mmproj={}",
                 json(model_key).dump(), json(registered->model_info.get_model_name()).dump(),
                 json(registered->path).dump(), json(registered->path_mmproj).dump());
    }
    return true;
}

void HLIEModelRegistry::normalize_model_instance(ModelInstance& model_instance) {
    if (model_instance.key.empty()) {
        model_instance.key = HLIEModelInfo::build_model_key(model_instance.model_info);
    }
    if (model_instance.meta.name.empty()) {
        model_instance.meta.name = model_instance.key;
    }
    if (model_instance.path.empty()) {
        model_instance.path = model_instance.model_info.get_model_path();
    }
    if (model_instance.path_mmproj.empty()) {
        model_instance.path_mmproj = model_instance.model_info.get_mm_proj_path();
    }
    model_instance.model_info.model_key_ = model_instance.key;

    if (!model_instance.meta.loaded_info.is_object()) {
        model_instance.meta.loaded_info = json::object();
    }
    if (!model_instance.model_info.get_model_type().empty()) {
        model_instance.meta.loaded_info["model_type"] = model_instance.model_info.get_model_type();
    }
    model_instance.meta.loaded_info["service_kind"] = model_instance.model_info.get_service_kind();
    model_instance.meta.loaded_info["model_family"] = model_instance.model_info.get_model_family();
    model_instance.meta.loaded_info["architecture"] = model_instance.model_info.get_architecture();
    model_instance.meta.loaded_info["role"] = model_instance.model_info.get_role_name();
    std::vector<int> device_ids;
    if (!model_instance.model_info.get_devices().empty() &&
        HlieUtil::parse_device(model_instance.model_info.get_devices(), device_ids) == 0 &&
        !device_ids.empty()) {
        model_instance.meta.loaded_info["device_ids"] = device_ids;
    } else {
        model_instance.meta.loaded_info.erase("device_ids");
    }
    if (model_instance.model_info.has_effective_llama_params()) {
        model_instance.meta.loaded_info["llama_presets"] = model_instance.model_info.get_matched_llama_presets();
        model_instance.meta.loaded_info["llama_params_hash"] = model_instance.model_info.get_effective_llama_params_hash();
        model_instance.meta.loaded_info["llama_params_summary"] = llama_params_summary(model_instance.model_info);
    }
    model_instance.meta.loaded_info["multimodal"] = model_instance.model_info.is_multi_model_;
    const auto& capabilities = model_instance.model_info.get_capabilities();
    model_instance.meta.loaded_info["capabilities"] = {
        {"text", capabilities.text}, {"vision", capabilities.vision}, {"audio", capabilities.audio}
    };
}

void HLIEModelRegistry::upsert(ModelInstance model_instance) {
    normalize_model_instance(model_instance);

    std::lock_guard lk(mutex_);
    model_instances_[model_instance.key] = std::move(model_instance);
}

bool HLIEModelRegistry::contains(const std::string& model_key) const {
    std::lock_guard lk(mutex_);
    return model_instances_.find(model_key) != model_instances_.end();
}

std::optional<server_model_meta> HLIEModelRegistry::get_meta(const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const ModelInstance* matched = find_instance_locked(model_name);
    return matched ? std::optional<server_model_meta>{matched->meta} : std::nullopt;
}

std::optional<std::reference_wrapper<HLIEModelRegistry::ModelInstance>> HLIEModelRegistry::get_instance(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (matched) {
        return std::ref(*matched);
    }

    LOG_WARN("Model lookup failed or is ambiguous: request={}, normalized={}, instance_count={}",
             json(model_name).dump(), json(normalize_model_name(model_name)).dump(), model_instances_.size());
    return std::nullopt;
}

std::optional<HLIEModelRegistry::ModelInstance> HLIEModelRegistry::get_instance_snapshot(const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return std::nullopt;
    }
    return *matched;
}

std::optional<HLIEModelRegistry::ModelInstance> HLIEModelRegistry::get_exact_instance_snapshot(
    const HLIEModelInfo& model_info) const {
    const std::string model_key = HLIEModelInfo::build_model_key(model_info);
    std::lock_guard lk(mutex_);
    const auto it = model_instances_.find(model_key);
    return it == model_instances_.end() ? std::nullopt : std::optional<ModelInstance>{it->second};
}

std::vector<server_model_meta> HLIEModelRegistry::get_all_metas() const {
    std::lock_guard lk(mutex_);
    std::vector<server_model_meta> result;
    result.reserve(model_instances_.size());
    for (const auto& [name, inst] : model_instances_) {
        result.push_back(inst.meta);
    }
    return result;
}

std::vector<HLIEModelRegistry::ModelInstance> HLIEModelRegistry::get_all_instances() const {
    std::lock_guard lk(mutex_);
    std::vector<ModelInstance> result;
    result.reserve(model_instances_.size());
    for (const auto& [name, inst] : model_instances_) {
        result.push_back(inst);
    }
    return result;
}

std::optional<HLIEModelInfo> HLIEModelRegistry::prepare_model_info_with_devices(const std::string& model_name,
                                                                                const std::string& devices) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched) {
        return std::nullopt;
    }

    matched->model_info.set_devices(devices);
    return matched->model_info;
}

void HLIEModelRegistry::record_access(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched) {
        return;
    }

    matched->last_access_time = std::time(nullptr);
    ++matched->hit_count;
}

bool HLIEModelRegistry::begin_request(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched || matched->meta.status != SERVER_MODEL_STATUS_LOADED || !matched->evictable) {
        return false;
    }

    matched->last_access_time = std::time(nullptr);
    ++matched->hit_count;
    ++matched->active_requests;
    LOG_DEBUG("Model request begin: key={}, active_requests={}", json(matched->key).dump(), matched->active_requests);
    return true;
}

void HLIEModelRegistry::end_request(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched) {
        return;
    }

    if (matched->active_requests > 0) {
        --matched->active_requests;
    }
    LOG_DEBUG("Model request end: key={}, active_requests={}", json(matched->key).dump(), matched->active_requests);
}

bool HLIEModelRegistry::try_begin_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched || matched->meta.status != SERVER_MODEL_STATUS_LOADED || matched->active_requests > 0 || !matched->evictable) {
        if (matched) {
            LOG_WARN("Model eviction refused: key={}, status={}, active_requests={}, evictable={}",
                     json(matched->key).dump(), static_cast<int>(matched->meta.status), matched->active_requests, matched->evictable);
        }
        return false;
    }

    matched->evictable = false;
    LOG_WARN("Model eviction begin: key={}", json(matched->key).dump());
    return true;
}

void HLIEModelRegistry::abort_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);

    if (!matched) {
        return;
    }

    matched->evictable = true;
    LOG_WARN("Model eviction aborted: key={}", json(matched->key).dump());
}

bool HLIEModelRegistry::try_begin_loading(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched || matched->meta.status == SERVER_MODEL_STATUS_LOADING ||
        matched->meta.status == SERVER_MODEL_STATUS_LOADED ||
        matched->meta.status == SERVER_MODEL_STATUS_SLEEPING) {
        return false;
    }

    matched->meta.status = SERVER_MODEL_STATUS_LOADING;
    matched->evictable = false;
    return true;
}

bool HLIEModelRegistry::mark_loading(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }

    matched->meta.status = SERVER_MODEL_STATUS_LOADING;
    matched->evictable = false;
    return true;
}

bool HLIEModelRegistry::mark_loaded(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }

    matched->meta.status = SERVER_MODEL_STATUS_LOADED;
    matched->meta.port = port;
    matched->last_load_time = std::time(nullptr);
    matched->evictable = true;
    return true;
}

bool HLIEModelRegistry::mark_unloaded(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }

    matched->meta.status = SERVER_MODEL_STATUS_UNLOADED;
    matched->meta.port = port;
    matched->last_load_time = 0;
    matched->active_requests = 0;
    matched->evictable = true;
    return true;
}

bool HLIEModelRegistry::update_status(const std::string& model_name, const server_model_status status, const int port) {
    if (status == SERVER_MODEL_STATUS_LOADING) {
        return mark_loading(model_name);
    }
    if (status == SERVER_MODEL_STATUS_LOADED) {
        return mark_loaded(model_name, port);
    }
    if (status == SERVER_MODEL_STATUS_UNLOADED) {
        return mark_unloaded(model_name, port);
    }

    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }
    matched->meta.status = status;
    matched->meta.port = port;
    return true;
}

bool HLIEModelRegistry::update_port(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }

    matched->meta.port = port;
    return true;
}

bool HLIEModelRegistry::update_model_info(const std::string& model_name, const HLIEModelInfo& model_info) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }
    matched->model_info = model_info;
    matched->model_info.model_key_ = matched->key;
    matched->path = model_info.get_model_path();
    matched->path_mmproj = model_info.get_mm_proj_path();
    matched->meta.loaded_info["role"] = model_info.get_role_name();
    std::vector<int> device_ids;
    if (!model_info.get_devices().empty() &&
        HlieUtil::parse_device(model_info.get_devices(), device_ids) == 0 &&
        !device_ids.empty()) {
        matched->meta.loaded_info["device_ids"] = device_ids;
    } else {
        matched->meta.loaded_info.erase("device_ids");
    }
    if (model_info.has_effective_llama_params()) {
        matched->meta.loaded_info["llama_presets"] = model_info.get_matched_llama_presets();
        matched->meta.loaded_info["llama_params_hash"] = model_info.get_effective_llama_params_hash();
        matched->meta.loaded_info["llama_params_summary"] = llama_params_summary(model_info);
    }
    return true;
}

bool HLIEModelRegistry::update_runtime_devices(const std::string& model_name,
                                               const std::vector<int>& device_ids) {
    std::lock_guard lk(mutex_);
    ModelInstance* matched = find_instance_locked(model_name);
    if (!matched) {
        return false;
    }

    if (device_ids.empty()) {
        matched->meta.loaded_info.erase("device_ids");
        matched->model_info.set_devices({});
    } else {
        matched->meta.loaded_info["device_ids"] = device_ids;
        matched->model_info.set_devices(HlieUtil::join_devices(device_ids));
    }
    return true;
}

HLIE_NAMESPACE_END
