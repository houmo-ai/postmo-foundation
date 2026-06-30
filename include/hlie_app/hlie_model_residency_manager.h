#ifndef HLIE_MODEL_RESIDENCY_MANAGER_H
#define HLIE_MODEL_RESIDENCY_MANAGER_H

#include <optional>
#include <set>
#include <string>
#include <ctime>
#include <cstdint>
#include <vector>

#include "hlie_app/hlie_model_registry.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Runtime residency policy loaded from [model] config.
// 模型驻留策略：控制自动卸载、TTL、最大驻留数量和保护名单。
struct HLIEModelResidencyPolicy {
    bool auto_unload = false;        // true: resource pressure may evict an idle model.
    int idle_ttl_sec = 0;            // reserved for background TTL cleanup; 0 disables TTL.
    int max_loaded_models = 0;       // max loaded model count; <=0 means no count limit.
    std::set<std::string> pinned_models; // protected from automatic eviction, exact normalized match.
    std::string default_model;       // default model is also protected from automatic eviction.
};

// Immutable description of the model selected for eviction.
// 驱逐候选快照：只描述“该卸谁”，实际 stop/terminate 由 Workflow 执行。
struct HLIEModelEvictionCandidate {
    std::string key;
    std::string model_name;
    int port = 0;
    int estimated_mem_mb = 0;
    std::time_t last_access_time = 0;
    std::time_t last_load_time = 0;
    uint64_t hit_count = 0;
    std::string reason;
};

// Selects residency victims from registry snapshots. It does not unload models.
// 驻留策略选择器：只做候选选择，不直接修改服务生命周期。
class HLIEModelResidencyManager {
public:
    explicit HLIEModelResidencyManager(HLIEModelResidencyPolicy policy);

    // Returns the least-recently-used loaded model that is not protected.
    // 返回一个可驱逐的 LRU 模型；会跳过默认模型、pinned、active_requests>0 等保护对象。
    [[nodiscard]] std::optional<HLIEModelEvictionCandidate> select_lru_candidate(
        const std::vector<HLIEModelRegistry::ModelInstance>& instances,
        const std::string& protected_model = "") const;

    // Returns a victim only when loaded count already reaches max_loaded_models.
    // max_loaded_models <= 0 disables this policy.
    [[nodiscard]] std::optional<HLIEModelEvictionCandidate> select_for_loaded_limit(
        const std::vector<HLIEModelRegistry::ModelInstance>& instances,
        const std::string& protected_model = "") const;

    // Returns an idle loaded model whose last access exceeds idle_ttl_sec.
    // idle_ttl_sec <= 0 disables this policy.
    [[nodiscard]] std::optional<HLIEModelEvictionCandidate> select_for_idle_ttl(
        const std::vector<HLIEModelRegistry::ModelInstance>& instances,
        std::time_t now,
        const std::string& protected_model = "") const;

private:
    [[nodiscard]] bool is_protected(const HLIEModelRegistry::ModelInstance& instance,
                                    const std::string& protected_model) const;

    static std::string normalize(std::string value);
    static bool name_matches(const HLIEModelRegistry::ModelInstance& instance, const std::string& name);

    HLIEModelResidencyPolicy policy_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_RESIDENCY_MANAGER_H
