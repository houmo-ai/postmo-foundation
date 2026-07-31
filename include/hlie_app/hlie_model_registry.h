#ifndef HLIE_MODEL_REGISTRY_H
#define HLIE_MODEL_REGISTRY_H

#include <map>
#include <mutex>
#include <optional>
#include <ctime>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <server/server-models.h>

#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Central registry for discovered and runtime model instances.
// 模型注册表：统一管理模型身份、路由状态和驻留元数据。
class HLIEModelRegistry {
public:
    struct ModelInstance {
        // Routing identity. key is the public unique id used by /models and router lookup.
        // 路由身份：key 是对外暴露和内部查找使用的模型唯一名。
        int port = 0;
        std::string key;
        std::string path;
        std::string path_mmproj;
        server_model_meta meta;
        HLIEModelInfo model_info;

        // Residency metadata used by LRU/TTL eviction.
        // 驻留管理元数据：用于判断最近访问、加载时间和是否可自动卸载。
        std::time_t last_access_time = 0;
        std::time_t last_load_time = 0;
        uint64_t hit_count = 0;
        int active_requests = 0;
        // Captured when the instance is registered. Eviction selection must
        // not perform filesystem or GGUF inspection.
        int estimated_mem_mb = 0;
        bool pinned = false;
        bool evictable = true;
    };

    void register_model(HLIEModelInfo& model_info, int estimated_mem_mb);
    // Discovery uses insert-only registration so a rescan cannot overwrite
    // the runtime status of an already loaded/loading model.
    [[nodiscard]] bool register_model_if_absent(HLIEModelInfo& model_info, int estimated_mem_mb);
    void upsert(ModelInstance model_instance);
    [[nodiscard]] bool contains(const std::string& model_key) const;

    // Lookup accepts key, model_name, or meta.name after trim/lower normalization.
    // 查找支持 key/model_name/meta.name 的规范化全等匹配。
    [[nodiscard]] std::optional<server_model_meta> get_meta(const std::string& model_name) const;
    // Compatibility-only mutable lookup. New callers should use the snapshot API.
    [[nodiscard]] std::optional<std::reference_wrapper<ModelInstance>> get_instance(const std::string& model_name);
    [[nodiscard]] std::optional<ModelInstance> get_instance_snapshot(const std::string& model_name) const;
    [[nodiscard]] std::optional<ModelInstance> get_exact_instance_snapshot(const HLIEModelInfo& model_info) const;
    [[nodiscard]] std::vector<server_model_meta> get_all_metas() const;
    [[nodiscard]] std::vector<ModelInstance> get_all_instances() const;

    // Update launch-time devices and return a copied model_info snapshot for service init.
    // 启动前写入 devices，并返回快照，避免 Workflow 长时间持有 registry 内部可变引用。
    [[nodiscard]] std::optional<HLIEModelInfo> prepare_model_info_with_devices(const std::string& model_name,
                                                                               const std::string& devices);

    // Access accounting. begin_request/end_request protect active models from eviction.
    // 请求计数：active_requests > 0 的模型不会被自动驱逐。
    void record_access(const std::string& model_name);
    bool begin_request(const std::string& model_name);
    void end_request(const std::string& model_name);

    // Eviction gate. try_begin_eviction flips evictable=false to block new requests.
    // 驱逐门闩：开始驱逐时阻止新请求进入；失败时调用 abort_eviction 恢复。
    bool try_begin_eviction(const std::string& model_name);
    void abort_eviction(const std::string& model_name);

    // Atomic state updates for Workflow and status callbacks.
    // 原子状态更新：避免外部拿到引用后分散修改 status/port/load_time。
    bool try_begin_loading(const std::string& model_name);
    bool mark_loading(const std::string& model_name);
    bool mark_loaded(const std::string& model_name, int port);
    bool mark_unloaded(const std::string& model_name, int port = 0);
    bool update_status(const std::string& model_name, server_model_status status, int port);
    bool update_port(const std::string& model_name, int port);
    bool update_model_info(const std::string& model_name, const HLIEModelInfo& model_info);
    bool update_runtime_devices(const std::string& model_name, const std::vector<int>& device_ids);

private:
    static ModelInstance build_model_instance(HLIEModelInfo& model_info, int estimated_mem_mb);
    static void normalize_model_instance(ModelInstance& model_instance);
    static std::string normalize_model_name(const std::string& name);
    ModelInstance* find_instance_locked(const std::string& model_name);
    const ModelInstance* find_instance_locked(const std::string& model_name) const;

    std::map<std::string, ModelInstance> model_instances_;
    mutable std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_REGISTRY_H
