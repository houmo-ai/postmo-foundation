#include "hlie_app/hlie_model_residency_manager.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include <server/server-models.h>

HLIE_NAMESPACE_BEGIN

HLIEModelResidencyManager::HLIEModelResidencyManager(HLIEModelResidencyPolicy policy)
    : policy_(std::move(policy)) {
}

std::optional<HLIEModelEvictionCandidate> HLIEModelResidencyManager::select_lru_candidate(
    const std::vector<HLIEModelRegistry::ModelInstance>& instances,
    const std::string& protected_model) const {
    const HLIEModelRegistry::ModelInstance* selected = nullptr;

    for (const auto& instance : instances) {
        if (instance.meta.status != SERVER_MODEL_STATUS_LOADED) {
            continue;
        }
        if (is_protected(instance, protected_model)) {
            continue;
        }
        if (!selected) {
            selected = &instance;
            continue;
        }

        const auto selected_access = selected->last_access_time != 0 ? selected->last_access_time : selected->last_load_time;
        const auto current_access = instance.last_access_time != 0 ? instance.last_access_time : instance.last_load_time;
        if (current_access < selected_access ||
            (current_access == selected_access && instance.hit_count < selected->hit_count)) {
            selected = &instance;
        }
    }

    if (!selected) {
        return std::nullopt;
    }

    HLIEModelEvictionCandidate candidate;
    candidate.key = selected->key;
    candidate.model_name = selected->model_info.get_model_name();
    candidate.port = selected->meta.port;
    candidate.estimated_mem_mb = selected->estimated_mem_mb;
    candidate.last_access_time = selected->last_access_time;
    candidate.last_load_time = selected->last_load_time;
    candidate.hit_count = selected->hit_count;
    candidate.reason = "lru";
    return candidate;
}

std::optional<HLIEModelEvictionCandidate> HLIEModelResidencyManager::select_for_loaded_limit(
    const std::vector<HLIEModelRegistry::ModelInstance>& instances,
    const std::string& protected_model) const {
    if (policy_.max_loaded_models <= 0) {
        return std::nullopt;
    }

    const auto loaded_count = std::count_if(instances.begin(), instances.end(), [](const auto& instance) {
        return instance.meta.status == SERVER_MODEL_STATUS_LOADED;
    });
    if (loaded_count < policy_.max_loaded_models) {
        return std::nullopt;
    }

    auto candidate = select_lru_candidate(instances, protected_model);
    if (candidate) {
        candidate->reason = "max_loaded_models";
    }
    return candidate;
}

std::optional<HLIEModelEvictionCandidate> HLIEModelResidencyManager::select_for_idle_ttl(
    const std::vector<HLIEModelRegistry::ModelInstance>& instances,
    const std::time_t now,
    const std::string& protected_model) const {
    if (policy_.idle_ttl_sec <= 0) {
        return std::nullopt;
    }

    const HLIEModelRegistry::ModelInstance* selected = nullptr;
    for (const auto& instance : instances) {
        if (instance.meta.status != SERVER_MODEL_STATUS_LOADED) {
            continue;
        }
        if (is_protected(instance, protected_model)) {
            continue;
        }

        const auto last_activity = instance.last_access_time != 0 ? instance.last_access_time : instance.last_load_time;
        if (last_activity == 0 || now - last_activity < policy_.idle_ttl_sec) {
            continue;
        }

        if (!selected) {
            selected = &instance;
            continue;
        }

        const auto selected_activity = selected->last_access_time != 0 ? selected->last_access_time : selected->last_load_time;
        if (last_activity < selected_activity ||
            (last_activity == selected_activity && instance.hit_count < selected->hit_count)) {
            selected = &instance;
        }
    }

    if (!selected) {
        return std::nullopt;
    }

    HLIEModelEvictionCandidate candidate;
    candidate.key = selected->key;
    candidate.model_name = selected->model_info.get_model_name();
    candidate.port = selected->meta.port;
    candidate.estimated_mem_mb = selected->estimated_mem_mb;
    candidate.last_access_time = selected->last_access_time;
    candidate.last_load_time = selected->last_load_time;
    candidate.hit_count = selected->hit_count;
    candidate.reason = "idle_ttl";
    return candidate;
}

bool HLIEModelResidencyManager::is_protected(const HLIEModelRegistry::ModelInstance& instance,
                                             const std::string& protected_model) const {
    if (!instance.evictable || instance.pinned || instance.active_requests > 0) {
        return true;
    }

    if (!protected_model.empty() && name_matches(instance, protected_model)) {
        return true;
    }

    if (!policy_.default_model.empty() && name_matches(instance, policy_.default_model)) {
        return true;
    }

    return std::any_of(policy_.pinned_models.begin(), policy_.pinned_models.end(),
                       [&instance](const std::string& pinned) { return name_matches(instance, pinned); });
}

std::string HLIEModelResidencyManager::normalize(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) {
        return {};
    }

    value = std::string(first, last);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool HLIEModelResidencyManager::name_matches(const HLIEModelRegistry::ModelInstance& instance, const std::string& name) {
    const std::string normalized = normalize(name);
    if (normalized.empty()) {
        return false;
    }

    return normalize(instance.key) == normalized ||
        normalize(instance.model_info.get_model_name()) == normalized ||
        normalize(instance.meta.name) == normalized;
}

HLIE_NAMESPACE_END
