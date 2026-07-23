#include "hlie_app/hlie_model_registry.h"

#include <algorithm>
#include <cctype>
#include <ctime>

#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

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
    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        return &it->second;
    }

    const std::string normalized_name = normalize_model_name(model_name);
    if (normalized_name.empty()) {
        return nullptr;
    }

    ModelInstance* matched = nullptr;
    for (auto& [name, inst] : model_instances_) {
        const bool key_matched = normalize_model_name(name) == normalized_name;
        const bool model_name_matched = normalize_model_name(inst.model_info.get_model_name()) == normalized_name;
        const bool meta_name_matched = normalize_model_name(inst.meta.name) == normalized_name;
        if (key_matched || model_name_matched || meta_name_matched) {
            if (!matched || inst.meta.status == SERVER_MODEL_STATUS_LOADED) {
                matched = &inst;
            }
        }
    }

    return matched;
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
    for (const auto& [name, inst] : model_instances_) {
        const bool key_matched = normalize_model_name(name) == normalized_name;
        const bool model_name_matched = normalize_model_name(inst.model_info.get_model_name()) == normalized_name;
        const bool meta_name_matched = normalize_model_name(inst.meta.name) == normalized_name;
        if (key_matched || model_name_matched || meta_name_matched) {
            if (!matched || inst.meta.status == SERVER_MODEL_STATUS_LOADED) {
                matched = &inst;
            }
        }
    }

    return matched;
}

void HLIEModelRegistry::register_model(HLIEModelInfo& model_info) {
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

    upsert(std::move(model_instance));

    const auto registered = get_instance_snapshot(model_key);
    if (registered) {
        LOG_INFO("Registered model instance: key={}, model_name={}, meta_name={}, model_path={}, mmproj={}",
                 json(model_key).dump(), json(registered->model_info.get_model_name()).dump(),
                 json(registered->meta.name).dump(), json(registered->path).dump(),
                 json(registered->path_mmproj).dump());
    }
}

void HLIEModelRegistry::upsert(ModelInstance model_instance) {
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
    model_instance.meta.loaded_info["multimodal"] = model_instance.model_info.is_multi_model_;
    const auto& capabilities = model_instance.model_info.get_capabilities();
    model_instance.meta.loaded_info["capabilities"] = {
        {"text", capabilities.text}, {"vision", capabilities.vision}, {"audio", capabilities.audio}
    };

    std::lock_guard lk(mutex_);
    model_instances_[model_instance.key] = std::move(model_instance);
}

bool HLIEModelRegistry::contains(const std::string& model_key) const {
    std::lock_guard lk(mutex_);
    return model_instances_.find(model_key) != model_instances_.end();
}

std::optional<server_model_meta> HLIEModelRegistry::get_meta(const std::string& model_name) const {
    std::lock_guard lk(mutex_);

    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        return it->second.meta;
    }

    const std::string normalized_name = normalize_model_name(model_name);
    if (normalized_name.empty()) {
        return std::nullopt;
    }

    for (const auto& [name, inst] : model_instances_) {
        if (normalize_model_name(name) == normalized_name ||
            normalize_model_name(inst.model_info.get_model_name()) == normalized_name ||
            normalize_model_name(inst.meta.name) == normalized_name) {
            return inst.meta;
        }
    }

    return std::nullopt;
}

std::optional<std::reference_wrapper<HLIEModelRegistry::ModelInstance>> HLIEModelRegistry::get_instance(const std::string& model_name) {
    std::lock_guard lk(mutex_);

    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        LOG_DEBUG("Model lookup exact key matched: request={}, key={}", json(model_name).dump(), json(it->first).dump());
        return std::ref(it->second);
    }

    const std::string normalized_name = normalize_model_name(model_name);
    if (normalized_name.empty()) {
        LOG_WARN("Model lookup failed because normalized request name is empty: request={}", json(model_name).dump());
        return std::nullopt;
    }

    ModelInstance* matched = nullptr;
    for (auto& [name, inst] : model_instances_) {
        const bool key_matched = normalize_model_name(name) == normalized_name;
        const bool model_name_matched = normalize_model_name(inst.model_info.get_model_name()) == normalized_name;
        const bool meta_name_matched = normalize_model_name(inst.meta.name) == normalized_name;
        if (key_matched || model_name_matched || meta_name_matched) {
            LOG_DEBUG(
                "Model lookup normalized match: request={}, key={}, model_name={}, meta_name={}, key_match={}, model_name_match={}, meta_name_match={}",
                json(model_name).dump(), json(name).dump(), json(inst.model_info.get_model_name()).dump(), json(inst.meta.name).dump(),
                key_matched, model_name_matched, meta_name_matched);
            if (!matched || inst.meta.status == SERVER_MODEL_STATUS_LOADED) {
                matched = &inst;
            }
        }
    }

    if (matched) {
        return std::ref(*matched);
    }

    LOG_WARN("Model lookup failed: request={}, normalized={}, instance_count={}",
             json(model_name).dump(), json(normalized_name).dump(), model_instances_.size());
    for (const auto& [name, inst] : model_instances_) {
        LOG_WARN("Model lookup candidate: key={}, model_name={}, meta_name={}, mmproj={}, status={}, port={}",
                 json(name).dump(), json(inst.model_info.get_model_name()).dump(), json(inst.meta.name).dump(),
                 json(inst.path_mmproj).dump(), static_cast<int>(inst.meta.status), inst.meta.port);
    }

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

HLIE_NAMESPACE_END
