#ifndef HLIE_APP_MODELS_MODEL_LIST_APPLICATION_SERVICE_H
#define HLIE_APP_MODELS_MODEL_LIST_APPLICATION_SERVICE_H

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "server/server-http.h"
#include "server/server-models.h"

HLIE_NAMESPACE_BEGIN

struct HLIEModelListLocalInstance {
    std::string model_key;
    std::vector<int> device_ids;
};

struct HLIEModelListProcessSnapshot {
    int instance_id = 0;
    std::uint64_t process_id = 0;
    int port = 0;
    std::vector<std::string> model_names;
    std::vector<int> device_ids;
};

struct HLIEModelListApplicationDependencies {
    std::function<std::vector<server_model_meta>()> models;
    std::function<std::vector<HLIEModelListLocalInstance>()>
        local_instances;
    std::function<std::vector<HLIEModelListProcessSnapshot>()>
        process_instances;
    std::function<std::optional<std::string>(const std::string&)>
        resolve_model_key;
    std::function<int()> local_port;
};

// Owns OpenAI-compatible /v1/models DTO assembly. Runtime and process state
// are supplied as neutral snapshots by the composition root.
class HLIEModelListApplicationService final {
public:
    explicit HLIEModelListApplicationService(
        HLIEModelListApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle() const;

private:
    HLIEModelListApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODELS_MODEL_LIST_APPLICATION_SERVICE_H
