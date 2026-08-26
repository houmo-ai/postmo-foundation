#include "hlie_app/models/hlie_model_list_application_service.h"

#include <algorithm>
#include <ctime>
#include <map>
#include <utility>

#include "hlie_base/hlie_common_define.h"

HLIE_NAMESPACE_BEGIN

HLIEModelListApplicationService::HLIEModelListApplicationService(
    HLIEModelListApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

server_http_res_ptr HLIEModelListApplicationService::handle() const {
    auto response = std::make_unique<server_http_res>();
    std::map<std::string, server_model_meta> public_models;
    if (dependencies_.models) {
        for (const auto& meta : dependencies_.models()) {
            if (meta.name.empty()) {
                continue;
            }
            const auto it = public_models.find(meta.name);
            if (it == public_models.end() ||
                (it->second.status != SERVER_MODEL_STATUS_LOADED &&
                 meta.status == SERVER_MODEL_STATUS_LOADED)) {
                public_models[meta.name] = meta;
            }
        }
    }

    const auto process_instances = dependencies_.process_instances
        ? dependencies_.process_instances()
        : std::vector<HLIEModelListProcessSnapshot>{};
    const auto local_instances = dependencies_.local_instances
        ? dependencies_.local_instances()
        : std::vector<HLIEModelListLocalInstance>{};
    const auto resolve_model_key = dependencies_.resolve_model_key;
    const std::time_t created = std::time(nullptr);
    const int local_port = dependencies_.local_port
        ? dependencies_.local_port() : 0;
    json models = json::array();

    for (const auto& [_, meta] : public_models) {
        json status{
            {"value", server_model_status_to_string(meta.status)},
            {"args", meta.args},
        };
        if (meta.is_failed()) {
            status["exit_code"] = meta.exit_code;
            status["failed"] = true;
        }
        json model{
            {"id", meta.name},
            {"aliases", meta.aliases},
            {"tags", meta.tags},
            {"object", "model"},
            {"owned_by", "HLIECpp"},
            {"created", created},
            {"port", meta.port},
            {"status", status},
            {"instances", json::array()},
        };

        if (meta.status == SERVER_MODEL_STATUS_LOADED ||
            meta.status == SERVER_MODEL_STATUS_SLEEPING) {
            const auto append_instance = [&](const int instance_id,
                                             const std::uint64_t process_id,
                                             const int port,
                                             const std::vector<int>& devices) {
                model["instances"].push_back({
                    {"instance_id", instance_id},
                    {"process_id", process_id},
                    {"port", port},
                    {"status", server_model_status_to_string(meta.status)},
                    {"device_ids", devices},
                });
            };

            const auto local = std::find_if(
                local_instances.begin(), local_instances.end(),
                [&meta](const HLIEModelListLocalInstance& instance) {
                    return instance.model_key == meta.name;
                });
            if (local != local_instances.end()) {
                auto devices = local->device_ids;
                if (devices.empty()) {
                    if (const auto it = meta.loaded_info.find("device_ids");
                        it != meta.loaded_info.end() && it->is_array()) {
                        devices = it->get<std::vector<int>>();
                    }
                }
                append_instance(0, 0, local_port, devices);
            }

            for (const auto& process : process_instances) {
                const bool matches = std::any_of(
                    process.model_names.begin(), process.model_names.end(),
                    [&resolve_model_key, &meta](const std::string& name) {
                        if (!resolve_model_key) {
                            return false;
                        }
                        const auto key = resolve_model_key(name);
                        return key && *key == meta.name;
                    });
                if (matches) {
                    append_instance(process.instance_id, process.process_id,
                                    process.port, process.device_ids);
                }
            }
        }

        if (meta.loaded_info.is_object()) {
            if (const auto it = meta.loaded_info.find("model_type");
                it != meta.loaded_info.end() && it->is_string()) {
                model["model_type"] = *it;
            }
            for (const auto* field : {"service_kind", "role", "model_family",
                                      "architecture", "llama_params_hash"}) {
                if (const auto it = meta.loaded_info.find(field);
                    it != meta.loaded_info.end() && it->is_string() &&
                    !it->get<std::string>().empty()) {
                    model[field] = *it;
                }
            }
            if (const auto it = meta.loaded_info.find("multimodal");
                it != meta.loaded_info.end() && it->is_boolean()) {
                model["multimodal"] = *it;
            }
            if (const auto it = meta.loaded_info.find("capabilities");
                it != meta.loaded_info.end() && it->is_object()) {
                model["capabilities"] = *it;
            }
            if (const auto it = meta.loaded_info.find("llama_presets");
                it != meta.loaded_info.end() && it->is_array()) {
                model["llama_presets"] = *it;
            }
            if (const auto it = meta.loaded_info.find("llama_params_summary");
                it != meta.loaded_info.end() && it->is_object()) {
                model["llama_params_summary"] = *it;
            }
        }
        models.push_back(std::move(model));
    }

    response->status = 200;
    response->content_type = MIMETYPE_JSON;
    response->data = json{{"data", std::move(models)},
                          {"object", "list"}}.dump();
    return response;
}

HLIE_NAMESPACE_END
