#ifndef HLIE_APP_MODELS_MODEL_UNLOAD_APPLICATION_SERVICE_H
#define HLIE_APP_MODELS_MODEL_UNLOAD_APPLICATION_SERVICE_H

#include <functional>
#include <optional>
#include <string>

#include "hlie_base/hlie_base.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

struct HLIEModelUnloadApplicationDependencies {
    std::function<bool()> replicas_mode_provider;
    std::function<std::optional<std::string>(const std::string&)>
        resolve_model_key;
    std::function<std::optional<std::string>(const std::string&)>
        find_loaded_model_by_type;
    std::function<bool(const std::string&)> is_supported_model_type;
    std::function<bool(const std::string&, const std::string&)> evict_model;
    std::function<void(const std::string&)> publish_unloaded;
};

// Owns the legacy unload HTTP selector/error mapping. Eviction and generation
// fencing remain in the Controller callback supplied by composition.
class HLIEModelUnloadApplicationService final {
public:
    explicit HLIEModelUnloadApplicationService(
        HLIEModelUnloadApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle(
        const server_http_req& request) const;

private:
    HLIEModelUnloadApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODELS_MODEL_UNLOAD_APPLICATION_SERVICE_H
