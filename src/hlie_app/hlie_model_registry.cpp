#include "hlie_app/hlie_model_registry.h"

#include <algorithm>
#include <ctime>
#include <limits>
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

HLIEModelLifecycleState runtime_state_from_meta(const server_model_meta& meta) {
    if (meta.exit_code != 0) {
        return HLIEModelLifecycleState::Failed;
    }
    switch (meta.status) {
    case SERVER_MODEL_STATUS_UNLOADED:
        return HLIEModelLifecycleState::Unloaded;
    case SERVER_MODEL_STATUS_LOADING:
        return HLIEModelLifecycleState::Starting;
    case SERVER_MODEL_STATUS_LOADED:
        return HLIEModelLifecycleState::Ready;
    case SERVER_MODEL_STATUS_SLEEPING:
        return HLIEModelLifecycleState::Sleeping;
    }
    return HLIEModelLifecycleState::Failed;
}

server_model_status legacy_status_from_runtime(const HLIEModelLifecycleState state) {
    switch (state) {
    case HLIEModelLifecycleState::Reserving:
    case HLIEModelLifecycleState::Starting:
        return SERVER_MODEL_STATUS_LOADING;
    case HLIEModelLifecycleState::Ready:
        return SERVER_MODEL_STATUS_LOADED;
    case HLIEModelLifecycleState::Sleeping:
        return SERVER_MODEL_STATUS_SLEEPING;
    case HLIEModelLifecycleState::Unloaded:
    case HLIEModelLifecycleState::Draining:
    case HLIEModelLifecycleState::Stopping:
    case HLIEModelLifecycleState::Failed:
    case HLIEModelLifecycleState::Orphaned:
        return SERVER_MODEL_STATUS_UNLOADED;
    }
    return SERVER_MODEL_STATUS_UNLOADED;
}

HLIEModelLifecycleState runtime_state_from_legacy(const server_model_status status) {
    switch (status) {
    case SERVER_MODEL_STATUS_UNLOADED:
        return HLIEModelLifecycleState::Unloaded;
    case SERVER_MODEL_STATUS_LOADING:
        return HLIEModelLifecycleState::Starting;
    case SERVER_MODEL_STATUS_LOADED:
        return HLIEModelLifecycleState::Ready;
    case SERVER_MODEL_STATUS_SLEEPING:
        return HLIEModelLifecycleState::Sleeping;
    }
    return HLIEModelLifecycleState::Failed;
}

} // namespace

HLIEModelRegistry::HLIEModelRegistry(
    std::shared_ptr<HLIEModelCatalog> catalog)
    : catalog_(catalog ? std::move(catalog)
                       : std::make_shared<HLIEModelCatalog>()),
      runtime_registry_([] { return HlieUtil::uuid(); }) {}

HLIEModelRegistry::ModelInstance HLIEModelRegistry::build_model_instance(
    HLIEModelInfo& model_info, const int estimated_mem_mb) {
    ModelInstance model_instance;
    const std::string model_key = HLIEModelInfo::build_model_key(model_info);

    model_instance.key = model_key;
    model_instance.path = model_info.get_model_path();
    model_instance.path_mmproj = model_info.get_mm_proj_path();
    model_instance.meta.name = model_key;
    model_info.model_key_ = model_key;
    model_instance.model_info = model_info;
    model_instance.estimated_mem_mb = std::max(0, estimated_mem_mb);
    return model_instance;
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
        model_instance.meta.loaded_info["model_type"] =
            model_instance.model_info.get_model_type();
    }
    model_instance.meta.loaded_info["service_kind"] =
        model_instance.model_info.get_service_kind();
    model_instance.meta.loaded_info["model_family"] =
        model_instance.model_info.get_model_family();
    model_instance.meta.loaded_info["architecture"] =
        model_instance.model_info.get_architecture();
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
        model_instance.meta.loaded_info["llama_presets"] =
            model_instance.model_info.get_matched_llama_presets();
        model_instance.meta.loaded_info["llama_params_hash"] =
            model_instance.model_info.get_effective_llama_params_hash();
        model_instance.meta.loaded_info["llama_params_summary"] =
            llama_params_summary(model_instance.model_info);
    } else {
        model_instance.meta.loaded_info.erase("llama_presets");
        model_instance.meta.loaded_info.erase("llama_params_hash");
        model_instance.meta.loaded_info.erase("llama_params_summary");
    }
    model_instance.meta.loaded_info["multimodal"] = model_instance.model_info.is_multi_model_;
    const auto& capabilities = model_instance.model_info.get_capabilities();
    model_instance.meta.loaded_info["capabilities"] = {
        {"text", capabilities.text},
        {"vision", capabilities.vision},
        {"audio", capabilities.audio},
    };
}

HLIEModelRuntimeRegistry::Registration HLIEModelRegistry::build_runtime_registration(
    const ModelInstance& model_instance) {
    HLIEModelRuntimeRegistry::Registration registration;
    auto& snapshot = registration.snapshot;
    snapshot.identity.model_key = model_instance.key;
    snapshot.identity.model_name = model_instance.model_info.get_model_name();
    snapshot.identity.display_name = model_instance.meta.name.empty()
        ? model_instance.key
        : model_instance.meta.name;
    snapshot.identity.model_path = model_instance.path;
    snapshot.identity.projector_path = model_instance.path_mmproj;
    snapshot.identity.backend_id = model_instance.model_info.get_service_kind();
    snapshot.state = runtime_state_from_meta(model_instance.meta);
    snapshot.generation = model_instance.generation;
    snapshot.port = model_instance.meta.port != 0
        ? model_instance.meta.port
        : model_instance.port;
    snapshot.failure_code = model_instance.meta.exit_code;
    snapshot.last_access_time = model_instance.last_access_time;
    snapshot.last_load_time = model_instance.last_load_time;
    snapshot.hit_count = model_instance.hit_count;
    snapshot.active_requests = static_cast<std::uint64_t>(
        std::max(0, model_instance.active_requests));
    snapshot.estimated_mem_mb = std::max(0, model_instance.estimated_mem_mb);
    snapshot.pinned = model_instance.pinned;
    snapshot.evictable = model_instance.evictable;
    registration.launch_token = model_instance.launch_token;
    return registration;
}

HLIEModelRegistry::CatalogEntry HLIEModelRegistry::build_catalog_entry(
    ModelInstance model_instance) {
    CatalogEntry entry;
    entry.meta = std::move(model_instance.meta);
    // Runtime fields are projected from core when a legacy DTO is requested.
    // Keeping neutral values here prevents the catalog from becoming a second
    // lifecycle store during migration.
    entry.meta.status = SERVER_MODEL_STATUS_UNLOADED;
    entry.meta.port = 0;
    entry.meta.exit_code = 0;
    entry.model_info = std::move(model_instance.model_info);
    entry.spec = entry.model_info.to_model_spec();
    return entry;
}

server_model_meta HLIEModelRegistry::project_meta(
    const CatalogEntry& catalog, const HLIEModelRuntimeSnapshot& runtime) {
    server_model_meta meta = catalog.meta;
    meta.status = legacy_status_from_runtime(runtime.state);
    meta.port = runtime.port;
    meta.exit_code = runtime.failure_code;
    return meta;
}

std::optional<std::string> HLIEModelRegistry::resolve_model_key_locked(
    const std::string& model_name) const {
    return runtime_registry_.resolve_model_key(model_name);
}

std::optional<HLIEModelRegistry::ModelInstance>
HLIEModelRegistry::build_instance_snapshot_locked(
    const HLIEModelRuntimeSnapshot& runtime) const {
    const auto catalog = catalog_->get(runtime.identity.model_key);
    if (!catalog) {
        return std::nullopt;
    }

    ModelInstance instance;
    instance.port = runtime.port;
    instance.key = runtime.identity.model_key;
    instance.path = runtime.identity.model_path;
    instance.path_mmproj = runtime.identity.projector_path;
    instance.meta = project_meta(*catalog, runtime);
    instance.model_info = catalog->model_info;
    instance.last_access_time = runtime.last_access_time;
    instance.last_load_time = runtime.last_load_time;
    instance.hit_count = runtime.hit_count;
    instance.active_requests = static_cast<int>(std::min<std::uint64_t>(
        runtime.active_requests, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    instance.estimated_mem_mb = runtime.estimated_mem_mb;
    instance.pinned = runtime.pinned;
    instance.evictable = runtime.evictable;
    instance.generation = runtime.generation;
    if (const auto ticket = runtime_registry_.get_launch_ticket(runtime.identity.model_key);
        ticket && ticket->generation == runtime.generation) {
        instance.launch_token = ticket->token;
    }
    return instance;
}

void HLIEModelRegistry::register_model(HLIEModelInfo& model_info,
                                       const int estimated_mem_mb) {
    auto model_instance = build_model_instance(model_info, estimated_mem_mb);
    const std::string model_key = model_instance.key;
    upsert(std::move(model_instance));

    const auto registered = get_instance_snapshot(model_key);
    if (registered) {
        LOG_INFO("Registered model instance: key={}, model_name={}, meta_name={}, model_path={}, mmproj={}",
                 json(model_key).dump(),
                 json(registered->model_info.get_model_name()).dump(),
                 json(registered->meta.name).dump(), json(registered->path).dump(),
                 json(registered->path_mmproj).dump());
    }
}

bool HLIEModelRegistry::register_model_if_absent(HLIEModelInfo& model_info,
                                                  const int estimated_mem_mb) {
    auto model_instance = build_model_instance(model_info, estimated_mem_mb);
    normalize_model_instance(model_instance);
    const std::string model_key = model_instance.key;
    auto registration = build_runtime_registration(model_instance);
    auto catalog = build_catalog_entry(std::move(model_instance));

    {
        std::lock_guard lk(mutex_);
        if (catalog_->contains(model_key) ||
            runtime_registry_.contains(model_key)) {
            return false;
        }
        if (!runtime_registry_.register_if_absent(std::move(registration))) {
            return false;
        }
        if (!catalog_->insert_if_absent(std::move(catalog))) {
            return false;
        }
    }

    const auto registered = get_instance_snapshot(model_key);
    if (registered) {
        LOG_INFO("Discovered new model instance: key={}, model_name={}, model_path={}, mmproj={}",
                 json(model_key).dump(),
                 json(registered->model_info.get_model_name()).dump(),
                 json(registered->path).dump(), json(registered->path_mmproj).dump());
    }
    return true;
}

void HLIEModelRegistry::upsert(ModelInstance model_instance) {
    normalize_model_instance(model_instance);
    auto registration = build_runtime_registration(model_instance);
    auto catalog = build_catalog_entry(std::move(model_instance));

    std::lock_guard lk(mutex_);
    runtime_registry_.upsert(std::move(registration));
    catalog_->upsert(std::move(catalog));
}

bool HLIEModelRegistry::contains(const std::string& model_key) const {
    std::lock_guard lk(mutex_);
    return catalog_->contains(model_key) &&
           runtime_registry_.contains(model_key);
}

std::optional<server_model_meta> HLIEModelRegistry::get_meta(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const auto runtime = runtime_registry_.get_snapshot(model_name);
    if (!runtime) {
        return std::nullopt;
    }
    const auto catalog = catalog_->get(runtime->identity.model_key);
    return catalog
        ? std::optional<server_model_meta>{project_meta(*catalog, *runtime)}
        : std::nullopt;
}

std::optional<HLIEModelRegistry::ModelInstance>
HLIEModelRegistry::get_instance_snapshot(const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const auto runtime = runtime_registry_.get_snapshot(model_name);
    return runtime ? build_instance_snapshot_locked(*runtime) : std::nullopt;
}

std::optional<HLIEModelRegistry::ModelInstance>
HLIEModelRegistry::get_exact_instance_snapshot(const HLIEModelInfo& model_info) const {
    const std::string model_key = HLIEModelInfo::build_model_key(model_info);
    std::lock_guard lk(mutex_);
    const auto runtime = runtime_registry_.get_snapshot(model_key);
    if (!runtime || runtime->identity.model_key != model_key) {
        return std::nullopt;
    }
    return build_instance_snapshot_locked(*runtime);
}

std::vector<server_model_meta> HLIEModelRegistry::get_all_metas() const {
    std::lock_guard lk(mutex_);
    std::vector<server_model_meta> result;
    for (const auto& runtime : runtime_registry_.get_all_snapshots()) {
        if (const auto catalog = catalog_->get(runtime.identity.model_key)) {
            result.push_back(project_meta(*catalog, runtime));
        }
    }
    return result;
}

std::vector<HLIEModelRegistry::ModelInstance> HLIEModelRegistry::get_all_instances() const {
    std::lock_guard lk(mutex_);
    std::vector<ModelInstance> result;
    for (const auto& runtime : runtime_registry_.get_all_snapshots()) {
        if (auto instance = build_instance_snapshot_locked(runtime)) {
            result.push_back(std::move(*instance));
        }
    }
    return result;
}

std::optional<HLIEModelRuntimeSnapshot> HLIEModelRegistry::get_runtime_snapshot(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    return runtime_registry_.get_snapshot(model_name);
}

std::vector<HLIEModelRuntimeSnapshot>
HLIEModelRegistry::get_all_runtime_snapshots() const {
    std::lock_guard lk(mutex_);
    return runtime_registry_.get_all_snapshots();
}

std::optional<HLIEModelInfo> HLIEModelRegistry::prepare_model_info_with_devices(
    const std::string& model_name, const std::string& devices) {
    std::lock_guard lk(mutex_);
    const auto key = resolve_model_key_locked(model_name);
    if (!key) {
        return std::nullopt;
    }
    auto catalog = catalog_->get(*key);
    if (!catalog) {
        return std::nullopt;
    }
    catalog->model_info.set_devices(devices);
    catalog->spec = catalog->model_info.to_model_spec();
    catalog_->upsert(*catalog);
    return catalog->model_info;
}

void HLIEModelRegistry::record_access(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    runtime_registry_.record_access(model_name);
}

std::optional<HLIEModelLaunchTicket> HLIEModelRegistry::begin_request_ticket(
    const std::string& model_name) {
    std::lock_guard lk(mutex_);
    auto ticket = runtime_registry_.begin_request(model_name);
    if (ticket) {
        const auto snapshot = runtime_registry_.get_snapshot(ticket->model_key);
        LOG_DEBUG("Model request begin: key={}, active_requests={}",
                  json(ticket->model_key).dump(),
                  snapshot ? snapshot->active_requests : 0);
    }
    return ticket;
}

void HLIEModelRegistry::end_request(const HLIEModelLaunchTicket& ticket) {
    std::lock_guard lk(mutex_);
    const bool was_current = runtime_registry_.is_current_launch(ticket);
    runtime_registry_.end_request(ticket);
    if (!was_current) {
        LOG_DEBUG("Ignored stale model request end: key={}, generation={}",
                  json(ticket.model_key).dump(), ticket.generation);
        return;
    }
    const auto snapshot = runtime_registry_.get_snapshot(ticket.model_key);
    LOG_DEBUG("Model request end: key={}, generation={}, active_requests={}",
              json(ticket.model_key).dump(), ticket.generation,
              snapshot ? snapshot->active_requests : 0);
}

bool HLIEModelRegistry::try_begin_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    const auto before = runtime_registry_.get_snapshot(model_name);
    if (!runtime_registry_.try_begin_eviction(model_name)) {
        if (before) {
            LOG_WARN("Model eviction refused: key={}, state={}, active_requests={}, evictable={}",
                     json(before->identity.model_key).dump(),
                     static_cast<int>(before->state), before->active_requests,
                     before->evictable);
        }
        return false;
    }
    LOG_INFO("Model eviction begin: key={}",
             json(before ? before->identity.model_key : model_name).dump());
    return true;
}

void HLIEModelRegistry::abort_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    const auto before = runtime_registry_.get_snapshot(model_name);
    runtime_registry_.abort_eviction(model_name);
    if (before) {
        LOG_WARN("Model eviction aborted: key={}",
                 json(before->identity.model_key).dump());
    }
}

bool HLIEModelRegistry::try_begin_loading(const std::string& model_name) {
    return try_begin_loading_ticket(model_name).has_value();
}

std::optional<HLIEModelLaunchTicket> HLIEModelRegistry::try_begin_loading_ticket(
    const std::string& model_name) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.try_begin_loading(model_name);
}

std::optional<HLIEModelLaunchTicket> HLIEModelRegistry::get_launch_ticket(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    return runtime_registry_.get_launch_ticket(model_name);
}

bool HLIEModelRegistry::is_current_launch(const HLIEModelLaunchTicket& ticket) const {
    std::lock_guard lk(mutex_);
    return runtime_registry_.is_current_launch(ticket);
}

bool HLIEModelRegistry::mark_loading(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_loading(model_name);
}

bool HLIEModelRegistry::mark_loaded(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_ready(model_name, port);
}

bool HLIEModelRegistry::mark_unloaded(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_unloaded(model_name, port);
}

bool HLIEModelRegistry::mark_failed(const std::string& model_name,
                                    const int exit_code) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_failed(model_name, exit_code);
}

bool HLIEModelRegistry::mark_loaded_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_ready_if_current(ticket, port);
}

bool HLIEModelRegistry::mark_unloaded_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_unloaded_if_current(ticket, port);
}

bool HLIEModelRegistry::mark_failed_if_current(
    const HLIEModelLaunchTicket& ticket, const int exit_code) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_failed_if_current(ticket, exit_code);
}

bool HLIEModelRegistry::mark_orphaned_if_current(
    const HLIEModelLaunchTicket& ticket, const int failure_code) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.mark_orphaned_if_current(ticket, failure_code);
}

bool HLIEModelRegistry::update_status_if_current(
    const HLIEModelLaunchTicket& ticket,
    const server_model_status status,
    const int port) {
    if (status == SERVER_MODEL_STATUS_LOADED) {
        return mark_loaded_if_current(ticket, port);
    }
    if (status == SERVER_MODEL_STATUS_UNLOADED) {
        return mark_unloaded_if_current(ticket, port);
    }
    std::lock_guard lk(mutex_);
    return runtime_registry_.update_state_if_current(
        ticket, runtime_state_from_legacy(status), port);
}

bool HLIEModelRegistry::update_status(const std::string& model_name,
                                      const server_model_status status,
                                      const int port) {
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
    return runtime_registry_.update_state(
        model_name, runtime_state_from_legacy(status), port);
}

bool HLIEModelRegistry::update_port(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    return runtime_registry_.update_port(model_name, port);
}

bool HLIEModelRegistry::update_model_info(const std::string& model_name,
                                          const HLIEModelInfo& model_info) {
    std::lock_guard lk(mutex_);
    const auto runtime = runtime_registry_.get_snapshot(model_name);
    if (!runtime) {
        return false;
    }
    const auto catalog = catalog_->get(runtime->identity.model_key);
    if (!catalog) {
        return false;
    }

    ModelInstance normalized;
    normalized.key = runtime->identity.model_key;
    normalized.meta = catalog->meta;
    normalized.model_info = model_info;
    normalize_model_instance(normalized);

    HLIEModelIdentity identity = runtime->identity;
    identity.model_name = normalized.model_info.get_model_name();
    identity.display_name = normalized.meta.name;
    identity.model_path = normalized.path;
    identity.projector_path = normalized.path_mmproj;
    identity.backend_id = normalized.model_info.get_service_kind();
    if (!runtime_registry_.update_identity(runtime->identity.model_key,
                                           std::move(identity))) {
        return false;
    }
    catalog_->upsert(build_catalog_entry(std::move(normalized)));
    return true;
}

bool HLIEModelRegistry::update_runtime_devices(
    const std::string& model_name, const std::vector<int>& device_ids) {
    std::lock_guard lk(mutex_);
    const auto key = resolve_model_key_locked(model_name);
    if (!key) {
        return false;
    }
    return catalog_->update(*key, [&](HLIEModelCatalogEntry& entry) {
        if (device_ids.empty()) {
            entry.meta.loaded_info.erase("device_ids");
            entry.model_info.set_devices({});
        } else {
            entry.meta.loaded_info["device_ids"] = device_ids;
            entry.model_info.set_devices(HlieUtil::join_devices(device_ids));
        }
        entry.spec = entry.model_info.to_model_spec();
    });
}

HLIE_NAMESPACE_END
