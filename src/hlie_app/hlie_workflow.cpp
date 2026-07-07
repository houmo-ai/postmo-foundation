#include "hlie_app/hlie_workflow.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <map>
#include <server/server-http.h>
#include <unordered_set>

#include "INIReader.h"
#include "hlie_app/hlie_document_input_adapter.h"
#include "hlie_app/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_residency_manager.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_server_http_proxy.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_rk_performance.h"
#include "hlie_base/hlie_util.h"

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_ws_service.h"

HLIE_NAMESPACE_BEGIN

#define CHECK_MODEL_READY_IMPL(service, check_name)                                                                                             \
    do {                                                                                                                                   \
        auto res = std::make_unique<server_res_generator_light>();                                                                               \
        if (!(service) || (service)->get_state() != SERVER_STATE_READY) {                                                                  \
            res->error(format_error_response("Model is not available for now.", ERROR_TYPE_UNAVAILABLE));                                  \
            return res;                                                                                                                    \
        }                                                                                                                                  \
        if ((check_name)) {                                                                                                                \
            const json body = json::parse(req.body);                                                                                       \
            const std::string name = json_value(body, "model", std::string());                                                             \
            if (!router_validate_model(name, res)) {                                                                                       \
                return res;                                                                                                                \
            }                                                                                                                              \
        }                                                                                                                                  \
    } while (0)

#define CHECK_MODEL_READY(service) CHECK_MODEL_READY_IMPL(service, true)
#define CHECK_SERVICE(service) CHECK_MODEL_READY_IMPL(service, false)

namespace {
    std::string normalize_model_name(const std::string& name) {
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

    template <typename Handler>
    server_http_res_ptr call_llm_service(const std::shared_ptr<LLModelService>& service, Handler&& handler) {
        auto res = std::make_unique<server_res_generator_light>();
        if (!service || service->get_state() != SERVER_STATE_READY) {
            res->error(format_error_response("LLM model is not available for now.", ERROR_TYPE_UNAVAILABLE));
            return res;
        }

        return handler(*service);
    }

    bool is_whisper_transcription_model(const std::string& model) {
        return model.empty() || HlieUtil::contains_ignore_case(model, "whisper");
    }

    void append_le_u16(raw_buffer& out, const uint16_t value) {
        out.push_back(static_cast<uint8_t>(value & 0xff));
        out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    }

    void append_le_u32(raw_buffer& out, const uint32_t value) {
        out.push_back(static_cast<uint8_t>(value & 0xff));
        out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
        out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
        out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    }

    raw_buffer pcmf32_to_wav_bytes(const std::vector<float>& pcmf32, const int sample_rate) {
        raw_buffer out;
        const uint16_t channels = 1;
        const uint16_t bits_per_sample = 16;
        const uint16_t block_align = channels * bits_per_sample / 8;
        const uint32_t byte_rate = static_cast<uint32_t>(std::max(sample_rate, 1)) * block_align;
        const uint32_t data_size = static_cast<uint32_t>(pcmf32.size() * sizeof(int16_t));
        const uint32_t riff_size = 36 + data_size;

        out.reserve(44 + data_size);
        out.insert(out.end(), {'R', 'I', 'F', 'F'});
        append_le_u32(out, riff_size);
        out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
        append_le_u32(out, 16);
        append_le_u16(out, 1);
        append_le_u16(out, channels);
        append_le_u32(out, static_cast<uint32_t>(std::max(sample_rate, 1)));
        append_le_u32(out, byte_rate);
        append_le_u16(out, block_align);
        append_le_u16(out, bits_per_sample);
        out.insert(out.end(), {'d', 'a', 't', 'a'});
        append_le_u32(out, data_size);

        for (const float sample : pcmf32) {
            const auto clipped = std::clamp(sample, -1.0f, 1.0f);
            const auto pcm = static_cast<int16_t>(clipped * 32767.0f);
            append_le_u16(out, static_cast<uint16_t>(pcm));
        }
        return out;
    }

    std::string parse_asr_text_response(const std::string& data) {
        const auto body = json::parse(data, nullptr, false);
        if (body.is_discarded()) {
            return data;
        }
        if (body.contains("text") && body["text"].is_string()) {
            return body["text"].get<std::string>();
        }
        if (body.contains("error")) {
            return "HLIECpp-FAILED-888";
        }
        return body.dump();
    }

    std::string trim_copy(const std::string& value) {
        const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) { return std::isspace(c); });
        const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
        if (first >= last) {
            return {};
        }
        return {first, last};
    }

    std::vector<std::string> split_model_list(const std::string& value) {
        std::vector<std::string> result;
        std::stringstream ss(value);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim_copy(token);
            if (!token.empty()) {
                result.push_back(token);
            }
        }
        return result;
    }

    HLIEModelResidencyPolicy build_residency_policy() {
        const auto& config = Singleton<HLIEConfigService>::instance();
        HLIEModelResidencyPolicy policy;
        policy.auto_unload = config->get_bool("model", "auto_unload", false, true);
        policy.idle_ttl_sec = config->get_int("model", "idle_ttl_sec", 0, true);
        policy.max_loaded_models = config->get_int("model", "max_loaded_models", 0, true);
        policy.default_model = config->get_string("model", "default", "", true);

        const auto pinned = config->get_string("model", "pinned_models", "", true);
        for (const auto& model : split_model_list(pinned)) {
            policy.pinned_models.insert(model);
        }

        return policy;
    }

    std::optional<HLIEModelEvictionCandidate> select_eviction_candidate(const HLIEModelRegistry& registry,
                                                                        const std::string& requested_model,
                                                                        const int required_mem_mb,
                                                                        HLIEModelResidencyPolicy& out_policy) {
        const auto policy = build_residency_policy();
        out_policy = policy;
        const HLIEModelResidencyManager residency(policy);
        const auto candidate = residency.select_lru_candidate(registry.get_all_instances(), requested_model);
        if (!candidate) {
            LOG_WARN("Model residency: no LRU eviction candidate for requested_model={}, required_mem={}MB, auto_unload={}",
                     json(requested_model).dump(), required_mem_mb, policy.auto_unload);
            return std::nullopt;
        }

        LOG_WARN("Model residency: LRU eviction candidate for requested_model={}, required_mem={}MB is key={}, model_name={}, port={}, estimated_mem={}MB, hits={}, auto_unload={}",
                 json(requested_model).dump(), required_mem_mb, json(candidate->key).dump(), json(candidate->model_name).dump(),
                 candidate->port, candidate->estimated_mem_mb, candidate->hit_count, policy.auto_unload);
        return candidate;
    }

    std::optional<HLIEModelEvictionCandidate> select_loaded_limit_candidate(const HLIEModelRegistry& registry,
                                                                            const std::string& requested_model,
                                                                            HLIEModelResidencyPolicy& out_policy) {
        const auto policy = build_residency_policy();
        out_policy = policy;
        const HLIEModelResidencyManager residency(policy);
        const auto candidate = residency.select_for_loaded_limit(registry.get_all_instances(), requested_model);
        if (!candidate) {
            LOG_DEBUG("Model residency: no max_loaded_models eviction needed for requested_model={}, max_loaded_models={}, auto_unload={}",
                      json(requested_model).dump(), policy.max_loaded_models, policy.auto_unload);
            return std::nullopt;
        }

        LOG_WARN("Model residency: max_loaded_models candidate for requested_model={} is key={}, model_name={}, port={}, max_loaded_models={}, auto_unload={}",
                 json(requested_model).dump(), json(candidate->key).dump(), json(candidate->model_name).dump(),
                 candidate->port, policy.max_loaded_models, policy.auto_unload);
        return candidate;
    }

    class ModelRequestLease {
    public:
        ModelRequestLease(HLIEModelRegistry& registry, std::string model_key)
            : registry_(&registry), model_key_(std::move(model_key)), active_(registry_->begin_request(model_key_)) {
        }

        ~ModelRequestLease() {
            if (active_) {
                registry_->end_request(model_key_);
            }
        }

        ModelRequestLease(const ModelRequestLease&) = delete;
        ModelRequestLease& operator=(const ModelRequestLease&) = delete;

        bool active() const {
            return active_;
        }

        void release() {
            active_ = false;
        }

    private:
        HLIEModelRegistry* registry_;
        std::string model_key_;
        bool active_ = false;
    };

    struct ActiveModelResponse final : server_http_res {
        ActiveModelResponse(server_http_res_ptr inner, HLIEModelRegistry& registry, std::string model_key)
            : inner_(std::move(inner)), registry_(registry), model_key_(std::move(model_key)) {
            content_type = inner_->content_type;
            status = inner_->status;
            data = std::move(inner_->data);
            headers = std::move(inner_->headers);
            if (inner_->is_stream()) {
                next = [this](std::string& out) {
                    if (!data.empty()) {
                        out = std::move(data);
                        data.clear();
                        return true;
                    }
                    return inner_->next(out);
                };
            }
        }

        ~ActiveModelResponse() override {
            registry_.end_request(model_key_);
        }

    private:
        server_http_res_ptr inner_;
        HLIEModelRegistry& registry_;
        std::string model_key_;
    };

    server_http_res_ptr attach_model_request_lease(server_http_res_ptr response, HLIEModelRegistry& registry,
                                                   const std::string& model_key, ModelRequestLease& lease) {
        if (!response || !lease.active()) {
            return response;
        }

        auto wrapped = std::make_unique<ActiveModelResponse>(std::move(response), registry, model_key);
        lease.release();
        return wrapped;
    }

    std::string clean_asr_pipeline_text(std::string text) {
        const std::string asr_text_tag = "<asr_text>";
        if (const auto pos = text.find(asr_text_tag); pos != std::string::npos) {
            text = text.substr(pos + asr_text_tag.size());
        }

        const std::string end_tag = "</asr_text>";
        if (const auto pos = text.find(end_tag); pos != std::string::npos) {
            text = text.substr(0, pos);
        }

        return trim_copy(text);
    }

    bool json_boolish(const json& body, const std::string& key) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return false;
        }
        const auto& value = body.at(key);
        if (value.is_boolean()) {
            return value.get<bool>();
        }
        if (value.is_string()) {
            const std::string s = normalize_model_name(value.get<std::string>());
            return s == "true" || s == "1" || s == "yes";
        }
        return false;
    }

    std::string json_string_value(const json& body, const std::string& key) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return {};
        }
        const auto& value = body.at(key);
        if (value.is_string()) {
            return value.get<std::string>();
        }
        return value.dump();
    }

    bool json_array_or_string_contains(const json& body, const std::string& key, const std::string& expected) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return false;
        }
        const auto& value = body.at(key);
        if (value.is_array()) {
            for (const auto& item : value) {
                if (item.is_string() && normalize_model_name(item.get<std::string>()) == expected) {
                    return true;
                }
            }
            return false;
        }
        if (value.is_string()) {
            return normalize_model_name(value.get<std::string>()) == expected;
        }
        return false;
    }

    bool requests_word_timestamps(const json& body) {
        return json_array_or_string_contains(body, "timestamp_granularities", "word") ||
               json_array_or_string_contains(body, "timestamp_granularities[]", "word");
    }

    bool is_force_aligner_name(const std::string& name) {
        return HlieUtil::contains_ignore_case(name, "forcealigner") ||
               HlieUtil::contains_ignore_case(name, "forcedaligner") ||
               HlieUtil::contains_ignore_case(name, "force-aligner") ||
               HlieUtil::contains_ignore_case(name, "forced-aligner") ||
               HlieUtil::contains_ignore_case(name, "force aligner");
    }

    bool is_qwen3_asr_name(const std::string& name) {
        return HlieUtil::contains_ignore_case(name, "qwen3_asr") ||
               HlieUtil::contains_ignore_case(name, "qwen3-asr") ||
               HlieUtil::contains_ignore_case(name, "qwen3asr");
    }

    std::string read_http_response_body(const server_http_res_ptr& response) {
        if (!response) {
            return {};
        }
        if (!response->is_stream()) {
            return response->data;
        }

        std::string data;
        std::string chunk;
        while (response->next(chunk)) {
            data += chunk;
            chunk.clear();
        }
        return data;
    }

    void normalize_openai_timestamp_granularities(json& body) {
        if (!body.contains("timestamp_granularities") && body.contains("timestamp_granularities[]")) {
            body["timestamp_granularities"] = body["timestamp_granularities[]"];
        }
    }

    std::vector<HLIEModelInfo> load_model_info_files_from_directory(const std::string& directory_path) {
        std::vector<HLIEModelInfo> model_infos;

        try {
            // Check whether the model info directory exists.
            if (!std::filesystem::exists(directory_path)) {
                LOG_ERROR("Directory does not exist: {}", directory_path);
                return model_infos;
            }

            if (!std::filesystem::is_directory(directory_path)) {
                LOG_ERROR("Path is not a directory: {}", directory_path);
                return model_infos;
            }

            // Iterate all model info files in the directory.
            for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
                // Only parse JSON files.
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    const std::string file_path = entry.path().string();
                    LOG_INFO("Parsing model info from file: {}", file_path);

                    if (HLIEModelInfo model_info; model_info.parse_from_file(file_path) == 0) {
                        model_infos.push_back(std::move(model_info));
                        LOG_INFO("Successfully parsed model info from: {}", file_path);
                    } else {
                        LOG_ERROR("Failed to parse model info from: {}", file_path);
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_ERROR("Filesystem error when reading directory {}: {}", directory_path, e.what());
        } catch (const std::exception& e) {
            LOG_ERROR("Exception when reading directory {}: {}", directory_path, e.what());
        }

        LOG_INFO("Loaded {} model info files from directory: {}", model_infos.size(), directory_path);
        return model_infos;
    }

    HLIEModelInfo parse_and_build_sd_model(const json& params, const std::unique_ptr<server_res_generator_light>& res,
                                           const std::string& model_type, const std::string& devices) {
        // Validate required SD model fields.
        const char* required_fields[] = {"vae_model", "t5_model", "clip_model", "clip_l_model", "mmdit_model"};

        for (const auto& field : required_fields) {
            if (!params.contains(field)) {
                const json resp = {{"ret", 80007}, {"msg", "params invalid, miss someone"}};
                res->error(resp);
                return {};
            }
        }

        // Parse SD model parameters.
        const auto vae_model = params["vae_model"].get<std::string>();
        const auto t5_model = params["t5_model"].get<std::string>();
        const auto clip_model = params["clip_model"].get<std::string>();
        const auto clip_l_model = params["clip_l_model"].get<std::string>();
        const auto mmdit_model = params["mmdit_model"].get<std::string>();

        // Build model info.
        HLIEModelInfo model_info;
        model_info.build_sd(model_type, vae_model, t5_model, clip_model, clip_l_model, mmdit_model, devices);

        return model_info;
    }

    std::string generate_meta_info(const HLIEModelInfo& model_info, const std::string& model_dir,
                                   const std::unique_ptr<server_res_generator_light>& res) {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto t = system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        const auto meta_info_file_name = "model_info" + oss.str() + ".hlif";
        const auto meta_cache_dir = HlieUtil::join_paths(model_dir, "meta_cache");
        std::string json_info_file;
        try {
            if (std::filesystem::create_directories(meta_cache_dir)) {
                LOG_INF("MetaCache dir created successfully.");
            }
            json_info_file = HlieUtil::join_paths(meta_cache_dir, meta_info_file_name);
        } catch (const std::filesystem::filesystem_error& ex) {
            LOG_ERROR("MetaCache Dir created failed, exception: {}", ex.what());
            json_info_file = HlieUtil::join_paths(model_dir, meta_info_file_name);
        }
        if (model_info.to_json_file(json_info_file) < 0) {
            LOG_ERROR("failed to write model info to json file");
            const json resp = {{"ret", 80007}, "failed to write model info to json file"};
            res->error(resp);
            return "";
        }

        return json_info_file;
    }

    void update_status(const int router_port, const std::string& name, const int port,
                       const server_model_status status = SERVER_MODEL_STATUS_LOADED) {
        // send a notification to the router server that a model instance is ready
        httplib::Client cli("127.0.0.1", router_port);
        cli.set_connection_timeout(0, 200000); // 200 milliseconds
        httplib::Request req;
        req.method = "POST";
        req.path = "/models/status";
        req.set_header("Content-Type", "application/json");

        json body;
        body["model"] = name;
        body["port"] = port;
        body["value"] = server_model_status_to_string(status);
        req.body = body.dump();

        LOG_INFO("notifying router server (port={}) that model {} at [{}] is ready", router_port, name.c_str(), port);
        if (const auto result = cli.send(req); result.error() != httplib::Error::Success) {
            const auto err_str = httplib::to_string(result.error());
            LOG_ERR("failed to notify router server: %s\n", err_str.c_str());
            // exit(1); // force exit
        }
    }

    bool is_equal_before_vit(const std::string& full_name, const std::string& name) {
        if (const size_t pos = full_name.find("_vit_"); pos != std::string::npos) {
            const std::string before_vit = full_name.substr(0, pos);
            return before_vit == name;
        }

        if (const size_t pos = full_name.find("_unknown"); pos != std::string::npos) {
            const std::string before_unknown = full_name.substr(0, pos);
            return before_unknown == name;
        }

        // If _vit_ is absent, compare the full name directly.
        return full_name == name;
    }
} // namespace

int HLIEWorkflow::initialize(const std::string& model_dir, const std::string& program_path) {
    model_dir_ = model_dir;
    program_path_ = program_path;
    if (model_dir.empty()) {
        LOG_ERROR("model dir is empty");
        return -1;
    }

    device_service_ = std::make_shared<HLIEDeviceService>();
    if (device_service_->init() < 0) {
        LOG_ERROR("failed to init the HLIE device service");
        return -1;
    }

    const auto& args = Singleton<HLIEConfigService>::instance()->get_args();
    replicas_mode_ =  Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1) > 1;
    master_port_ = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    std::vector<HLIEModelInfo> model_infos;
    if (args.find("--json_info") != args.end()) { // Slave process startup from generated model info.
        std::string json_path = args.at("--json_info");
        if (HLIEModelInfo model_info; model_info.parse_from_file(args.at("--json_info")) == 0) {
            model_infos.emplace_back(model_info);
            LOG_INFO("Successfully parsed model info from --json_info: {}", json_path);
            is_slave_ = true;
            register_model_instance(model_info);
            // Slave startup now honors the real service kind in json_info.
            // This allows Whisper/CosyVoice subprocesses to load their native service.
            switch (HLIEModelServiceFactory::service_kind(model_info)) {
            case ModelServiceKind::Whisper: {
                LOG_INFO("Initializing WhisperModelService");
                if (!model_service_manager_.load_whisper(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the Whisper model service");
                    return -1;
                }
                break;
            }
            case ModelServiceKind::CosyVoice: {
                LOG_INFO("Initializing CosyVoiceModelService");
                if (!model_service_manager_.load_cosyvoice(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the CosyVoice model service");
                    return -1;
                }
                break;
            }
            case ModelServiceKind::SD: {
                LOG_INFO("Initializing SDModelService");
                if (!model_service_manager_.load_sd(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the SD model service");
                    return -1;
                }
                break;
            }
            case ModelServiceKind::LLM:
            default: {
                LOG_INFO("Initializing LLModelService");
                if (!model_service_manager_.load_llm(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the LLM model service");
                    return -1;
                }
                if (DocumentPipelineService::supports_document_layout(model_dir, model_info) &&
                    !model_service_manager_.load_document_pipeline(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the document pipeline service");
                    model_service_manager_.stop_and_reset_llm();
                    return -1;
                }
                break;
            }
            }

            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 0;
        }

        LOG_ERROR("Failed to parse model info from --json_info: {}", json_path);
        return -1;
    }

    // 主进程或者是负载均衡模式
    model_infos = load_model_info_files_from_directory(model_dir);
    HLIEModelInfo::ScanLocalModelsFromDir(model_infos, model_dir_);
    for (auto& model_info : model_infos) {
        register_model_instance(model_info);
    }

    if (model_infos.empty()) {
        LOG_WARN("No valid model info files found in directory: {}", model_dir);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        return -1;
    }

    const std::string default_model = Singleton<HLIEConfigService>::instance()->get_string("model", "default");
    const bool enable_sd3 = Singleton<HLIEConfigService>::instance()->get_bool("model", "enable_sd3", false);
    if (default_model.empty() && !enable_sd3) {
        LOG_WARN("No default model: {}", default_model);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        return -1;
    }

    bool found_default_model = false;
    bool found_sd3 = false;
    HLIEModelInfo default_model_info;
    HLIEModelInfo sd3_model_info;
    if (!default_model.empty()) {
        for (const auto& m : model_infos) {
            LOG_INFO("Scanned model: {}", m.get_model_name());
            if (m.model_key_ == default_model || (!found_default_model && m.get_model_name() == default_model)) {
                LOG_INFO("default_model found: {}", default_model);
                found_default_model = true;
                default_model_info = m;
            } else if (enable_sd3 && m.get_model_name() == HLIEModelType::SD3) {
                LOG_INFO("SD3 found");
                found_sd3 = true;
                sd3_model_info = m;
            }
        }

        if (!found_default_model) {
            LOG_ERROR("Default model not found: {}, scanned models count: {}", default_model, model_infos.size());
        }
    } else if (enable_sd3) {
        auto it = std::find_if(model_infos.begin(), model_infos.end(),
                               [&](const HLIEModelInfo& info) { return info.get_model_name() == HLIEModelType::SD3; });
        if (it == model_infos.end()) {
            LOG_WARN("No SD3 model found");
            return -1;
        }

        found_sd3 = true;
        sd3_model_info = *it;
    }

    if (!found_default_model && !found_sd3) {
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        LOG_WARN("No default models found for slave instance");
        return -1;
    }

    if (found_default_model) {
        // The configured default model may be served by LLM, Whisper, CosyVoice, or SD.
        // SD3 auto-loading is still handled separately by enable_sd3 below.
        switch (HLIEModelServiceFactory::service_kind(default_model_info)) {
        case ModelServiceKind::Whisper: {
            LOG_INFO("Initializing WhisperModelService");
            if (!model_service_manager_.load_whisper(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the whisper model service");
                return -1;
            }
            break;
        }
        case ModelServiceKind::CosyVoice: {
            LOG_INFO("Initializing CosyVoiceModelService");
            if (!model_service_manager_.load_cosyvoice(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the cosyvoice model service");
                return -1;
            }
            break;
        }
        case ModelServiceKind::SD: {
            LOG_INFO("Initializing SDModelService");
            if (!model_service_manager_.load_sd(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the SD model service");
                return -1;
            }
            break;
        }
        case ModelServiceKind::LLM:
        default: {
            LOG_INFO("Initializing LLModelService");
            if (const std::string default_mmproj = Singleton<HLIEConfigService>::instance()->get_string("model", "default_mmproj");
                !default_mmproj.empty()) {
                auto mmproj_path = default_mmproj + ".gguf";
                if (HlieUtil::is_relative_path(mmproj_path)) {
                    mmproj_path = HlieUtil::join_paths(model_dir, mmproj_path);
                }

                std::filesystem::path mmproj_path_tmp(mmproj_path);
                if (std::error_code ec;
                    !std::filesystem::exists(mmproj_path_tmp, ec) || !std::filesystem::is_regular_file(mmproj_path_tmp, ec)) {
                    LOG_ERROR("default_mmproj file does not exist");
                    return -1;
                }

                default_model_info.mm_proj_path_ = mmproj_path;
                default_model_info.vl_spec_ = HLIEModelInfo::parse_mmproj_vl_spec(mmproj_path);
                default_model_info.model_key_ = HLIEModelInfo::build_model_key(default_model_info);
                if (!model_registry_.contains(default_model_info.model_key_)) {
                    ModelInstance model_instance;
                    server_model_meta meta;
                    meta.name = default_model_info.model_key_;
                    model_instance.path = default_model_info.get_model_path();
                    model_instance.path_mmproj = default_model_info.get_mm_proj_path();
                    model_instance.meta = std::move(meta);
                    model_instance.key = default_model_info.model_key_;
                    model_instance.model_info = default_model_info;
                    model_registry_.upsert(std::move(model_instance));
                }
            }

            if (!model_service_manager_.load_llm(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the LLM model service");
                return -1;
            }
            if (DocumentPipelineService::supports_document_layout(model_dir, default_model_info) &&
                !model_service_manager_.load_document_pipeline(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the document pipeline service");
                model_service_manager_.stop_and_reset_llm();
                return -1;
            }
            break;
        }
        }
    }

    if (found_sd3) {
        LOG_INFO("Initializing SDModelService");
        if (!model_service_manager_.load_sd(model_dir, sd3_model_info, device_service_)) {
            LOG_ERROR("failed to init the SD model service");
            return -1;
        }
    }

    Singleton<HLIEDeviceHelper>::instance()->print_status();
    start_time_ = time(nullptr);
    return 0;
}

void HLIEWorkflow::run() {
    if (model_service_manager_.llm()) {
        model_service_manager_.llm()->run();
        if (model_service_manager_.document_pipeline()) {
            model_service_manager_.document_pipeline()->run();
        }

        const auto model_key = model_service_manager_.llm()->get_model_key();
        const auto inst = get_model_inst(model_key);
        if (!inst) {
            LOG_ERROR("LLM model instance is missing after initialization: key={}", json(model_key).dump());
            return;
        }
        model_registry_.mark_loaded(model_key, inst->get().meta.port);

        if (is_slave_ && !replicas_mode_) {
            std::thread([this, model_key] {
                const auto inst = get_model_inst(model_key);
                if (!inst) {
                    LOG_ERROR("LLM model instance is missing before status update: key={}", json(model_key).dump());
                    return;
                }
                model_registry_.update_port(model_key, port_);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, model_key, port_);
            }).detach();
        }
    }

    if (model_service_manager_.sd()) {
        model_service_manager_.sd()->run();
        model_registry_.mark_loaded(HLIEModelType::SD3, port_);
        if (is_slave_ && !replicas_mode_) {
            std::thread([this] {
                model_registry_.update_port(HLIEModelType::SD3, port_);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, HLIEModelType::SD3, port_);
            }).detach();
        }
    }

    if (model_service_manager_.whisper()) {
        model_service_manager_.whisper()->run();
        const auto model_key = model_service_manager_.whisper()->get_model_key();
        model_registry_.mark_loaded(model_key, port_);
        if (is_slave_ && !replicas_mode_) {
            std::thread([this, model_key] {
                model_registry_.update_port(model_key, port_);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, model_key, port_);
            }).detach();
        }
    }

    if (model_service_manager_.cosyvoice()) {
        model_service_manager_.cosyvoice()->run();
        const auto model_key = model_service_manager_.cosyvoice()->get_model_key();
        model_registry_.mark_loaded(model_key, port_);
        if (is_slave_ && !replicas_mode_) {
            std::thread([this, model_key] {
                model_registry_.update_port(model_key, port_);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, model_key, port_);
            }).detach();
        }
    }

    ensure_asr_websocket();
    start_residency_ttl_worker();
}

void HLIEWorkflow::stop() {
    stop_residency_ttl_worker();

    if (asr_websocket_service_) {
        asr_websocket_service_->stop();
        asr_websocket_service_.reset();
    }

    if (model_service_manager_.sd()) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, HLIEModelType::SD3, port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        model_service_manager_.stop_and_reset_sd();
    }

    if (model_service_manager_.whisper()) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, model_service_manager_.whisper()->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        model_service_manager_.stop_and_reset_whisper();
    }

    if (model_service_manager_.cosyvoice()) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, model_service_manager_.cosyvoice()->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        model_service_manager_.stop_and_reset_cosyvoice();
    }

    if (model_service_manager_.llm()) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, model_service_manager_.llm()->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        model_service_manager_.stop_and_reset_document_pipeline();
        model_service_manager_.stop_and_reset_llm();
    }
}

void HLIEWorkflow::ensure_asr_websocket() {
    if (asr_websocket_service_ || (!model_service_manager_.whisper() && !model_service_manager_.llm())) {
        return;
    }

    asr_websocket_service_ = std::make_unique<HLIEWebSocketService>();
    asr_websocket_service_->init();
    asr_websocket_service_->run(
        [this](const WhisperStreamRequest& request, const TranscriptCallback& callback) {
            return this->handle_asr_stream(request, callback);
        },
        7999);
    LOG_INFO("ASR websocket router started on port 7999");
}

std::string HLIEWorkflow::handle_asr_stream(
    const WhisperStreamRequest& request,
    const std::function<void(const std::string&)>& callback) {
    const std::string model = request.model.empty() ? "whisper-1" : request.model;
    if (is_whisper_transcription_model(request.model)) {
        if (!model_service_manager_.whisper() || model_service_manager_.whisper()->get_state() != SERVER_STATE_READY) {
            LOG_WARN("ASR websocket routed to Whisper, but Whisper service is not ready");
            return "HLIECpp-FAILED-888";
        }
        LOG_INFO("ASR websocket route: backend=whisper, model={}, samples={}", model, request.pcmf32.size());
        ModelRequestLease lease(model_registry_, model_service_manager_.whisper()->get_model_key());
        if (!lease.active()) {
            LOG_WARN("ASR websocket routed to Whisper, but registry refused active request");
            return "HLIECpp-FAILED-888";
        }
        return model_service_manager_.whisper()->handle_stream(request, callback);
    }

    LOG_INFO("ASR websocket route: backend=llm, model={}, samples={}, sample_rate={}",
             model, request.pcmf32.size(), request.sample_rate);
    const raw_buffer wav = pcmf32_to_wav_bytes(request.pcmf32, request.sample_rate);
    const json body = {
        {"model", model},
        {"response_format", request.response_format.empty() ? "json" : request.response_format},
        {"stream", "false"},
    };
    json request_body = body;
    if (!request.prompt.empty()) {
        request_body["prompt"] = request.prompt;
    }
    if (!request.language.empty()) {
        request_body["language"] = request.language;
    }

    std::function<bool()> should_stop = [] { return false; };
    server_http_req transcription_req{
        {},
        {},
        "/v1/audio/transcriptions",
        "",
        request_body.dump(),
        {{"file", uploaded_file{wav, "audio.wav", "audio/wav"}}},
        should_stop,
    };

    const auto response = route_llm_request(
        transcription_req,
        [this](const server_http_req& routed_req) {
            return call_llm_service(model_service_manager_.llm(), [&](const LLModelService& service) {
                return service.handle_transcriptions(routed_req);
            });
        });
    if (!response) {
        LOG_ERROR("ASR websocket LLM transcription returned null response");
        return "HLIECpp-FAILED-888";
    }
    if (response->status < 200 || response->status >= 300) {
        LOG_ERROR("ASR websocket LLM transcription failed: status={}, body={}", response->status, response->data);
        return "HLIECpp-FAILED-888";
    }

    std::string transcript;
    if (response->is_stream()) {
        std::string chunk;
        while (response->next(chunk)) {
            if (!chunk.empty()) {
                transcript += chunk;
                callback(chunk);
            }
            chunk.clear();
        }
    } else {
        transcript = parse_asr_text_response(response->data);
    }

    if (transcript == "HLIECpp-FAILED-888") {
        LOG_ERROR("ASR websocket LLM transcription response contains error: {}", response->data);
        return transcript;
    }
    LOG_INFO("ASR websocket LLM transcription completed: chars={}", transcript.size());
    return transcript;
}

std::shared_ptr<IHLIEModelService> HLIEWorkflow::get_model_service(const std::string& type) const {
    return model_service_manager_.get_by_type(type);
}

int HLIEWorkflow::cacl_thread_num() const {
    if (model_service_manager_.llm()) {
        const auto batch_num = model_service_manager_.llm()->get_batch_num();
        return std::max(batch_num, static_cast<int32_t>(std::thread::hardware_concurrency()) - 1);
    }

    return static_cast<int32_t>(std::thread::hardware_concurrency()) - 1;
}

int HLIEWorkflow::get_model_service_state() const {
    if (model_service_manager_.llm()) {
        return model_service_manager_.llm()->get_state();
    }

    if (model_service_manager_.sd()) {
        return model_service_manager_.sd()->get_state();
    }

    if (model_service_manager_.whisper()) {
        return model_service_manager_.whisper()->get_state();
    }

    if (model_service_manager_.cosyvoice()) {
        return model_service_manager_.cosyvoice()->get_state();
    }

    LOG_WARN("model service not initialized");
    return -1;
}

std::vector<std::string> HLIEWorkflow::get_api_keys() const {
    return api_keys_;
}

server_http_res_ptr HLIEWorkflow::handle_health(const server_http_req& req) const {
    if (model_service_manager_.llm()) {
        return model_service_manager_.llm()->handle_health(req);
    }

    auto res = std::make_unique<server_res_generator_light>();
    res->ok({{"status", "ok"}});
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_models(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
    json models_json = json::array();
    const auto all_models = get_all_model_metas();
    std::map<std::string, server_model_meta> public_models;
    for (const auto& meta : all_models) {
        if (meta.name.empty()) {
            continue;
        }

        auto it = public_models.find(meta.name);
        if (it == public_models.end() ||
            (it->second.status != SERVER_MODEL_STATUS_LOADED && meta.status == SERVER_MODEL_STATUS_LOADED)) {
            public_models[meta.name] = meta;
        }
    }

    std::time_t t = std::time(nullptr);
    for (const auto& [_, meta] : public_models) {
        json status{
            {"value", server_model_status_to_string(meta.status)},
            {"args", meta.args},
        };
        if (meta.is_failed()) {
            status["exit_code"] = meta.exit_code;
            status["failed"] = true;
        }
        models_json.push_back(json{
            {"id", meta.name},
            {"aliases", meta.aliases},
            {"tags", meta.tags},
            {"object", "model"}, // for OAI-compat
            {"owned_by", "HLIECpp"}, // for OAI-compat
            {"created", t}, // for OAI-compat
            {"port", meta.port},
            {"status", status},
            // TODO: add other fields, may require reading GGUF metadata
        });
    }
    res->ok({
        {"data", models_json},
        {"object", "list"},
    });
    return res;
}

server_http_res_ptr HLIEWorkflow::start_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
    if (replicas_mode_) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    start_residency_ttl_worker();

    auto body = json::parse(req.body);
    std::string model_type = body.value("model_type", LLM);
    std::string devices = body.value("devices", "undefined");
    std::string model_name = body.value("model", std::string());
    const std::string model_path = body.value("model_path", std::string());
    LOG_DEBUG("start_service devices: {}, model_type: {}, model_name: {}", devices, model_type, model_name);

    if (!HLIEModelServiceFactory::is_supported_model_type(model_type)) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    if (model_name.empty() && model_type == SD) {
        model_name = HLIEModelType::SD3;
    }

    std::optional<server_model_meta> meta = std::nullopt;
    int mem_est;
    int required_device_num = HLIEModelInfo::estimate_device_num(model_name);
    if (!model_name.empty()) { // Compatibility path: check whether the model exists.
        meta = get_model_meta(model_name);
        if (!meta.has_value()) {
            res->error(format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        if (meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const auto model_inst = get_model_inst(model_name);
        mem_est = model_inst ? model_inst->get().model_info.get_estimated_mem() : HLIEModelInfo::estimate_mem(model_name);
        if (model_inst) {
            required_device_num = model_inst->get().model_info.get_device_num();
        }
    } else if (model_path.empty()) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return res;
    } else {
        std::filesystem::path model_path_tmp(model_path);
        if (std::error_code ec; !std::filesystem::exists(model_path_tmp, ec) || !std::filesystem::is_regular_file(model_path_tmp, ec)) {
            res->error(format_error_response_ex("model_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
            return res;
        }

        model_name = model_path_tmp.filename().stem().string();
        required_device_num = HLIEModelInfo::estimate_device_num(model_name);
        meta = get_model_meta(model_name);
        if (meta && meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        mem_est = HLIEModelInfo::estimate_empirical_mem(model_name);
        if (mem_est <= 0) {
            mem_est = HLIEModelInfo::estimate_mem_from_gguf(model_path, 0, 1);
        }
        if (mem_est <= 0) {
            mem_est = HLIEModelInfo::estimate_mem(model_name);
        }
    }

    const bool is_qwen3_audio_llm =
        HlieUtil::contains_ignore_case(model_name, "qwen3_asr") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3-asr") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3_forcedaligner") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3-forcedaligner");
    if (model_type == WHISPER && is_qwen3_audio_llm) {
        LOG_WARN("Model {} was requested as whisper, overriding model_type to llm", model_name);
        model_type = LLM;
    }

    LOG_INFO("ready to acquire devices");
    auto try_evict_candidate = [&](const std::optional<HLIEModelEvictionCandidate>& candidate,
                                   const HLIEModelResidencyPolicy& policy,
                                   const std::string& reason) {
        if (!candidate || !policy.auto_unload) {
            return false;
        }

        const auto inst = get_model_inst(candidate->key);
        if (!inst) {
            LOG_ERROR("Model residency: eviction candidate disappeared before unload: key={}", json(candidate->key).dump());
            return false;
        }

        const auto model_copy = inst->get();
        return evict_model(model_copy, reason);
    };

    auto try_auto_evict_for_resource = [&]() {
        HLIEModelResidencyPolicy policy;
        const auto candidate = select_eviction_candidate(model_registry_, model_name, mem_est, policy);
        return try_evict_candidate(candidate, policy, "resource_pressure");
    };

    {
        HLIEModelResidencyPolicy policy;
        const auto candidate = select_loaded_limit_candidate(model_registry_, model_name, policy);
        if (candidate && !try_evict_candidate(candidate, policy, "max_loaded_models")) {
            if (policy.auto_unload) {
                res->error(format_error_response_ex("max_loaded_models limit reached, but no model could be evicted.",
                                                     ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
        }
    }

    {
        std::unique_lock lock(mutex_);
        if (devices == "undefined") {
            std::vector<int> devices_vec;
            if (required_device_num == 1) {
                uint32_t device = 0;
                if (device_service_->acquire_mem(mem_est, device, true, true, model_name) < 0) {
                    lock.unlock();
                    const bool evicted = try_auto_evict_for_resource();
                    lock.lock();
                    if (!evicted || device_service_->acquire_mem(mem_est, device, true, true, model_name) < 0) {
                        res->error(format_error_response_ex("The devices is not available, estimated model memory is " + std::to_string(mem_est) + "MB.",
                                                             ERROR_TYPE_UNAVAILABLE, 80004));
                        return res;
                    }
                }
                devices_vec.push_back(static_cast<int>(device));
            } else {
                if (device_service_->acquire_mem_on_device_count(required_device_num, mem_est, devices_vec, true, model_name) < 0) {
                    lock.unlock();
                    const bool evicted = try_auto_evict_for_resource();
                    lock.lock();
                    if (!evicted || device_service_->acquire_mem_on_device_count(required_device_num, mem_est, devices_vec, true, model_name) < 0) {
                        res->error(format_error_response_ex("The devices is not available, estimated model memory is " + std::to_string(mem_est) + "MB.",
                                                             ERROR_TYPE_UNAVAILABLE, 80004));
                        return res;
                    }
                }
            }
            devices = HlieUtil::join_devices(devices_vec);
        } else {
            std::vector<int> devices_vec;
            if (HlieUtil::parse_device(devices, devices_vec) < 0) {
                res->error(format_error_response_ex("The devices parsed failed.", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
            std::sort(devices_vec.begin(), devices_vec.end());
            devices_vec.erase(std::unique(devices_vec.begin(), devices_vec.end()), devices_vec.end());
            if (devices_vec.size() != static_cast<size_t>(required_device_num)) {
                LOG_ERROR("model {} requires {} devices, but {} devices were requested", model_name, required_device_num, devices_vec.size());
                res->error(format_error_response_ex("The requested device count does not match the model chip count.",
                                                     ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
            devices = HlieUtil::join_devices(devices_vec);
            if (device_service_->acquire_mem_on_devices(devices_vec, mem_est, false, true, model_name) < 0) {
                lock.unlock();
                const bool evicted = try_auto_evict_for_resource();
                lock.lock();
                if (!evicted || device_service_->acquire_mem_on_devices(devices_vec, mem_est, false, true, model_name) < 0) {
                    res->error(format_error_response_ex("The devices is not available, estimated model memory is " + std::to_string(mem_est) + "MB.",
                                                         ERROR_TYPE_UNAVAILABLE, 80004));
                    return res;
                }
            }
        }
    }

    LOG_INFO("acquire devices successfully with {}", devices);
    auto instance_id = 0;
    std::string json_info_file;
    ProcessID process_id = 0;
    if (model_type == LLM) {
        if (!meta) {
            HLIEModelInfo model_info_tmp;
            model_info_tmp.parse_from_gguf_filename(model_path);
            if (body.contains("params") && body["params"].is_object()) {
                auto& params = body["params"];
                if (std::string mm_proj_path = params.value("mm_proj_path", std::string{}); !mm_proj_path.empty()) {
                    model_info_tmp.is_multi_model_ = true;
                    model_info_tmp.mm_proj_path_ = mm_proj_path;
                    model_info_tmp.vl_spec_ = HLIEModelInfo::parse_mmproj_vl_spec(mm_proj_path);
                    if (std::error_code ec;
                        !std::filesystem::exists(mm_proj_path, ec) || !std::filesystem::is_regular_file(mm_proj_path, ec)) {
                        res->error(format_error_response_ex("mm_proj_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
                        return res;
                    }
                }
            }

            ModelInstance model_instance;
            server_model_meta model_meta;
            const auto model_key = HLIEModelInfo::build_model_key(model_info_tmp);
            model_meta.name = model_key;
            model_instance.path = model_info_tmp.get_model_path();
            model_instance.path_mmproj = model_info_tmp.get_mm_proj_path();
            model_instance.meta = std::move(model_meta);
            model_info_tmp.model_key_ = model_key;
            model_instance.model_info = model_info_tmp;
            model_instance.key = model_key;
            model_registry_.upsert(std::move(model_instance));
            LOG_INFO("Service Starting Add Model: {} {}", model_key, model_info_tmp.get_mm_proj_path());
        }

        auto model_inst = get_model_inst(model_name);
        if (!model_inst) {
            res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        const std::string model_key = model_inst->get().key;
        auto model_info_opt = model_registry_.prepare_model_info_with_devices(model_key, devices);
        if (!model_info_opt) {
            res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        auto model_info = std::move(*model_info_opt);
        if (model_info.model_version_.empty()) {
            res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
            res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        model_registry_.mark_loading(model_key);
        if (!model_service_manager_.llm()) {
            if (!model_service_manager_.start_llm(model_dir_, model_info, device_service_)) {
                LOG_ERROR("failed to init the LLM model service");
                res->error(format_error_response_ex("failed to init the LLM model service", ERROR_TYPE_SERVER, 80001));
                rollback_model_start(model_key, "llm_init_failed");
                return res;
            }
            if (DocumentPipelineService::supports_document_layout(model_dir_, model_info) &&
                !model_service_manager_.start_document_pipeline(model_dir_, model_info, device_service_)) {
                LOG_ERROR("failed to init the document pipeline service");
                res->error(format_error_response_ex("failed to init the document pipeline service", ERROR_TYPE_SERVER, 80001));
                model_service_manager_.stop_and_reset_llm();
                rollback_model_start(model_key, "document_pipeline_init_failed");
                return res;
            }

            mark_local_model_started(model_key, model_service_manager_.llm()->get_model_key(), true);
        } else {
            if (port_ != master_port_) {
                LOG_ERROR("slave cannot start new slave");
                res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
                rollback_model_start(model_key, "slave_cannot_start_new_slave");
                return res;
            }

            // Subprocess instance arguments.
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                rollback_model_start(model_key, "generate_meta_info_failed");
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            const auto& args = config_services->get_args();

            // Copy llama-params
            for (const auto& [key, val] : args) {
                args_for_sub_process.push_back(key);
                args_for_sub_process.push_back(val);
            }

            process_id = Singleton<HLIEProcessManager>::instance()->create_process(
                program_path_, args_for_sub_process, {model_name}, instance_id);
            if (process_id == 0 || process_id == static_cast<ProcessID>(-1)) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
                res->error(format_error_response_ex("failed to start model subprocess", ERROR_TYPE_SERVER, 80001));
                rollback_model_start(model_key, "create_process_failed");
                return res;
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", process_id, instance_id);
            }
        }
    } else if (model_type == SD) {
        if (!body.contains("params") || !body["params"].is_object()) {
            res->error(format_error_response_ex("params invalid for SD3", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        auto& params = body["params"];
        auto model_info = parse_and_build_sd_model(params, res, model_type, devices);
        if (model_info.get_model_type().empty()) {
            return res;
        }

        if (!model_service_manager_.sd()) {
            model_registry_.mark_loading(HLIEModelType::SD3);
            if (!model_service_manager_.start_sd(model_dir_, model_info, device_service_)) {
                LOG_ERROR("failed to init the SD model service");
                res->error(format_error_response_ex("failed to init the SD model service", ERROR_TYPE_SERVER, 80001));
                rollback_model_start(HLIEModelType::SD3, "sd_init_failed");
                return res;
            }

            mark_local_model_started(HLIEModelType::SD3, HLIEModelType::SD3, false);
        } else {
            if (port_ != master_port_) {
                LOG_ERROR("slave cannot start new slave");
                res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
                return res;
            }

            // Subprocess instance arguments.
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            if (process_id = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process,
                                                                                       {HLIEModelType::SD3}, instance_id);
                process_id == 0 || process_id == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", process_id, instance_id);
            }
        }
    } else if (model_type == WHISPER) {
        if (start_whisper(meta, res, req, model_name, model_path, devices) < 0) {
            LOG_ERROR("failed to start the WHISPER model");
            return res;
        }
    } else if (model_type == HLIEModelType::COSYVOICE) {
        if (start_cosyvoice(meta, res, req, model_name, model_path, devices) < 0) {
            LOG_ERROR("failed to start the COSYVOICE model");
            return res;
        }
    }

    // TODO: 模型启动参数校验, device 是否可用
    json resp = {
        {"code", 0},
        {"port", port_ + instance_id},
        {"instance_id", instance_id},
        {"process_id", process_id},
        {"json_info_file", json_info_file},
    };

    res->ok(resp);
    return res;
}

int HLIEWorkflow::start_whisper(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res,
    const server_http_req& req, const std::string& model_name, const std::string& model_path, const std::string& devices) {

    if (!meta) {
        auto body = json::parse(req.body);
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);

        ModelInstance model_instance;
        server_model_meta model_meta;
        model_meta.name = model_info_tmp.get_model_name();
        model_instance.path = model_info_tmp.get_model_path();
        model_instance.meta = std::move(model_meta);
        auto model_key = model_info_tmp.get_model_name();
        model_info_tmp.model_key_ = model_key;
        model_instance.model_info = model_info_tmp;
        model_instance.key = model_key;
        model_registry_.upsert(std::move(model_instance));
        LOG_INFO("Service Starting Add WHISPER Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    const std::string model_key = model_inst->get().key;
    auto model_info_opt = model_registry_.prepare_model_info_with_devices(model_key, devices);
    if (!model_info_opt) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }
    auto model_info = std::move(*model_info_opt);
    if (model_info.model_version_.empty()) {
        res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
        res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    model_registry_.mark_loading(model_key);
    if (!model_service_manager_.whisper()) {
        if (!model_service_manager_.start_whisper(model_dir_, model_info, device_service_)) {
            LOG_ERROR("failed to init the Whisper model service");
            res->error(format_error_response_ex("failed to init the Whisper model service", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "whisper_init_failed");
            return -1;
        }

        mark_local_model_started(model_key, model_service_manager_.whisper()->get_model_key(), true);
    } else {
        if (port_ != master_port_) {
            LOG_ERROR("slave cannot start new slave");
            res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "slave_cannot_start_new_slave");
            return -1;
        }

        // Subprocess instance arguments.
        std::string json_info_file = generate_meta_info(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            rollback_model_start(model_key, "generate_meta_info_failed");
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--json_info"),   json_info_file};
        const auto& args = Singleton<HLIEConfigService>::instance()->get_args();

        // Copy llama-params
        for (const auto& [key, val] : args) {
            args_for_sub_process.push_back(key);
            args_for_sub_process.push_back(val);
        }

        if (auto pid =
                Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {model_name}, instance_id);
            pid == 0 || pid == -1) {
            LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            res->error(format_error_response_ex("failed to start model subprocess", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "create_process_failed");
            return -1;
        } else {
            LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
        }
    }

    return 0;
}

int HLIEWorkflow::start_cosyvoice(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res,
    const server_http_req& req, const std::string& model_name, const std::string& model_path, const std::string& devices) {

    if (!meta) {
        auto body = json::parse(req.body);
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);

        ModelInstance model_instance;
        server_model_meta model_meta;
        model_meta.name = model_info_tmp.get_model_name();
        model_instance.path = model_info_tmp.get_model_path();
        model_instance.meta = std::move(model_meta);
        auto model_key = model_info_tmp.get_model_name();
        model_info_tmp.model_key_ = model_key;
        model_instance.model_info = model_info_tmp;
        model_instance.key = model_key;
        model_registry_.upsert(std::move(model_instance));
        LOG_INFO("Service Starting Add COSYVOICE Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    const std::string model_key = model_inst->get().key;
    auto model_info_opt = model_registry_.prepare_model_info_with_devices(model_key, devices);
    if (!model_info_opt) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }
    auto model_info = std::move(*model_info_opt);
    if (model_info.model_version_.empty()) {
        res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
        res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    model_registry_.mark_loading(model_key);
    if (!model_service_manager_.cosyvoice()) {
        if (!model_service_manager_.start_cosyvoice(model_dir_, model_info, device_service_)) {
            LOG_ERROR("failed to init the COSYVOICE model service");
            res->error(format_error_response_ex("failed to init the COSYVOICE model service", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "cosyvoice_init_failed");
            return -1;
        }

        mark_local_model_started(model_key, model_service_manager_.cosyvoice()->get_model_key(), false);
    } else {
        if (port_ != master_port_) {
            LOG_ERROR("slave cannot start new slave");
            res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "slave_cannot_start_new_slave");
            return -1;
        }

        // Subprocess instance arguments.
        std::string json_info_file = generate_meta_info(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            rollback_model_start(model_key, "generate_meta_info_failed");
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--json_info"),   json_info_file};
        const auto& args = Singleton<HLIEConfigService>::instance()->get_args();

        // Copy llama-params
        for (const auto& [key, val] : args) {
            args_for_sub_process.push_back(key);
            args_for_sub_process.push_back(val);
        }

        if (auto pid =
                Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {model_name}, instance_id);
            pid == 0 || pid == -1) {
            LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            res->error(format_error_response_ex("failed to start model subprocess", ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "create_process_failed");
            return -1;
        } else {
            LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
        }
    }

    return 0;
}

bool HLIEWorkflow::router_validate_model(const std::string& name, const std::unique_ptr<server_res_generator_light>& res) {
    if (name.empty()) {
        res->error(format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    const auto meta = get_model_inst(name);
    if (!meta.has_value()) {
        res->error(format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    if (meta->get().meta.status != SERVER_MODEL_STATUS_LOADED) {
        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    return true;
}

server_http_res_ptr HLIEWorkflow::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, int port) {
    // if (update_last_used) {
    //     std::unique_lock<std::mutex> lk(mutex);
    //     mapping[name].meta.last_used = ggml_time_ms();
    // }
    LOG_INFO("proxying request to model {} on port {}", name.c_str(), port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += "?" + req.query_string;
    }

    auto proxy = std::make_unique<ServerHttpProxy>(
            method,
            "127.0.0.1",
            port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            req.should_stop);
    return proxy;
}

server_http_res_ptr HLIEWorkflow::handle_transcriptions_pipeline(const server_http_req& req, const json& body) {
    auto res = std::make_unique<server_res_generator_light>();

    if (!requests_word_timestamps(body)) {
        res->error(format_error_response("ASR pipeline requires timestamp_granularities[]=word", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const std::string response_format = json_value(body, "response_format", std::string("json"));
    if (response_format != "verbose_json") {
        res->error(format_error_response(
            "timestamp_granularities requires response_format to be verbose_json", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    if (req.files.find("file") == req.files.end()) {
        res->error(format_error_response("No input file found for transcription", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const std::string requested_model = json_value(body, "model", std::string());
    std::string asr_model;
    if (!requested_model.empty() &&
        !is_force_aligner_name(requested_model) &&
        !is_whisper_transcription_model(requested_model) &&
        get_model_inst(requested_model).has_value()) {
        asr_model = requested_model;
    }
    if (asr_model.empty()) {
        for (const auto& inst : model_registry_.get_all_instances()) {
            const auto& key = inst.key;
            if (is_qwen3_asr_name(key) || is_qwen3_asr_name(inst.model_info.get_model_name())) {
                asr_model = key;
                break;
            }
        }
    }

    std::string aligner_model = json_string_value(body, "force_aligner_model");
    if (aligner_model.empty()) {
        aligner_model = json_string_value(body, "aligner_model");
    }
    if (aligner_model.empty()) {
        aligner_model = json_string_value(body, "timestamp_model");
    }
    if (aligner_model.empty()) {
        for (const auto& inst : model_registry_.get_all_instances()) {
            const auto& key = inst.key;
            if (is_force_aligner_name(key) || is_force_aligner_name(inst.model_info.get_model_name())) {
                aligner_model = key;
                break;
            }
        }
    }

    if (asr_model.empty()) {
        res->error(format_error_response(
            "ASR model is missing; pass a Qwen3 ASR model in the model field for word timestamp transcription",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (aligner_model.empty()) {
        res->error(format_error_response(
            "ForceAligner model is missing; pass aligner_model or load a Qwen3 ForceAligner model",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    LOG_INFO("ASR word timestamp pipeline: asr_model={}, aligner_model={}, file_count={}",
             json(asr_model).dump(), json(aligner_model).dump(), req.files.size());

    json asr_body = body;
    asr_body["model"] = asr_model;
    asr_body["response_format"] = "json";
    asr_body["stream"] = "false";
    asr_body.erase("timestamp_granularities");
    asr_body.erase("timestamp_granularities[]");
    asr_body.erase("asr_model");
    asr_body.erase("transcription_model");
    asr_body.erase("force_aligner_model");
    asr_body.erase("aligner_model");
    asr_body.erase("timestamp_model");
    asr_body.erase("asr_force_align");
    asr_body.erase("force_align");
    asr_body.erase("text");

    server_http_req asr_req{
        req.params,
        req.headers,
        req.path,
        req.query_string,
        asr_body.dump(),
        req.files,
        req.should_stop,
    };

    const auto asr_response = route_llm_request(
        asr_req,
        [this](const server_http_req& routed_req) {
            return call_llm_service(model_service_manager_.llm(), [&](const LLModelService& service) {
                return service.handle_transcriptions(routed_req);
            });
        });
    if (!asr_response) {
        res->error(format_error_response("ASR transcription returned no response", ERROR_TYPE_SERVER));
        return res;
    }

    const std::string asr_data = read_http_response_body(asr_response);
    if (asr_response->status < 200 || asr_response->status >= 300) {
        LOG_ERROR("ASR pipeline transcription failed: status={}, body={}", asr_response->status, asr_data);
        auto failed = std::make_unique<server_http_res>();
        failed->status = asr_response->status;
        failed->content_type = asr_response->content_type;
        failed->data = asr_data;
        return failed;
    }

    const std::string transcript = clean_asr_pipeline_text(parse_asr_text_response(asr_data));
    if (transcript.empty() || transcript == "HLIECpp-FAILED-888") {
        res->error(format_error_response("ASR transcription did not return text", ERROR_TYPE_SERVER));
        return res;
    }

    json align_body = body;
    normalize_openai_timestamp_granularities(align_body);
    align_body["model"] = aligner_model;
    align_body["response_format"] = "verbose_json";
    align_body["stream"] = "false";
    align_body["asr_force_align"] = "true";
    align_body["text"] = transcript;
    if (!align_body.contains("timestamp_granularities")) {
        align_body["timestamp_granularities"] = json::array({"word"});
    }
    align_body.erase("timestamp_granularities[]");
    align_body.erase("asr_model");
    align_body.erase("transcription_model");
    align_body.erase("force_aligner_model");
    align_body.erase("aligner_model");
    align_body.erase("timestamp_model");

    server_http_req align_req{
        req.params,
        req.headers,
        req.path,
        req.query_string,
        align_body.dump(),
        req.files,
        req.should_stop,
    };

    const auto align_response = route_llm_request(
        align_req,
        [this](const server_http_req& routed_req) {
            return call_llm_service(model_service_manager_.llm(), [&](const LLModelService& service) {
                return service.handle_transcriptions(routed_req);
            });
        });
    if (!align_response) {
        res->error(format_error_response("ForceAligner returned no response", ERROR_TYPE_SERVER));
        return res;
    }

    const std::string align_data = read_http_response_body(align_response);
    if (align_response->status < 200 || align_response->status >= 300) {
        LOG_ERROR("ASR pipeline force alignment failed: status={}, body={}", align_response->status, align_data);
        auto failed = std::make_unique<server_http_res>();
        failed->status = align_response->status;
        failed->content_type = align_response->content_type;
        failed->data = align_data;
        return failed;
    }

    json final_body = json::parse(align_data, nullptr, false);
    if (final_body.is_discarded() || !final_body.is_object()) {
        res->error(format_error_response("ForceAligner returned invalid JSON", ERROR_TYPE_SERVER));
        return res;
    }
    if (final_body.contains("error")) {
        auto failed = std::make_unique<server_http_res>();
        failed->status = 500;
        failed->data = safe_json_to_str(final_body);
        return failed;
    }

    final_body["text"] = transcript;
    if (!final_body.contains("task")) {
        final_body["task"] = "transcribe";
    }
    if (body.contains("language") && body.at("language").is_string() && !final_body.contains("language")) {
        final_body["language"] = body.at("language");
    }

    res->ok(final_body);
    return res;
}

server_http_res_ptr HLIEWorkflow::route_llm_request(
    const server_http_req& req, const RequestHandler& handler, const std::string& method) {
    auto res = std::make_unique<server_res_generator_light>();
    json body = json::object();
    std::string name = req.get_param("model");
    if (name.empty() && !req.body.empty()) {
        body = json::parse(req.body);
        name = json_value(body, "model", std::string());
    }
    LOG_INFO("Routing LLM request: path={}, model={}, model_length={}, fields={}, files={}",
             req.path, json(name).dump(), name.size(), body.is_object() ? body.size() : 0, req.files.size());
    if (name.empty()) {
        res->error(format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    auto handle_local_llm_or_document = [this, &req, &handler, &body](const std::string& model_key,
                                                                ModelRequestLease& lease) -> server_http_res_ptr {
        const auto document_input = DocumentInputAdapter::inspect(req);
        if (document_input.has_pdf) {
            if (!model_service_manager_.document_pipeline() ||
                model_service_manager_.document_pipeline()->get_state() != SERVER_STATE_READY) {
                auto doc_res = std::make_unique<server_res_generator_light>();
                doc_res->error(format_error_response_ex(
                    "PDF input requires a loaded document pipeline for this model", ERROR_TYPE_NOT_SUPPORTED, 80005));
                return doc_res;
            }
            LOG_INFO("Routing PDF request to document pipeline: model={}, source={}, filename={}, content_type={}",
                     json(model_key).dump(), json(document_input.source).dump(), json(document_input.filename).dump(),
                     json(document_input.content_type).dump());
            if (json_boolish(body, "document_preprocess_only")) {
                return attach_model_request_lease(model_service_manager_.document_pipeline()->handle_document_request(req),
                                                  model_registry_, model_key, lease);
            }
            return attach_model_request_lease(model_service_manager_.document_pipeline()->handle_document_ocr_request(
                                                  req, *model_service_manager_.llm()),
                                              model_registry_, model_key, lease);
        }

        return attach_model_request_lease(handler(req), model_registry_, model_key, lease);
    };

    const auto inst = get_model_inst(name);
    if (!inst.has_value()) {
        LOG_ERROR("LLM request model lookup failed: path={}, model={}, normalized_model={}",
                  req.path, json(name).dump(), json(normalize_model_name(name)).dump());
        res->error(format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    LOG_INFO("LLM Request route info, model [{}] status [{}] port [{}]" , name, static_cast<int>(inst->get().meta.status), inst->get().meta.port);
    if (inst->get().meta.status == SERVER_MODEL_STATUS_LOADED && (inst->get().meta.port == master_port || master_port != port_)) {
        const auto model_key = inst->get().key;
        ModelRequestLease lease(model_registry_, model_key);
        if (!lease.active()) {
            res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_local_llm_or_document(model_key, lease);
    }

    const bool auto_load = Singleton<HLIEConfigService>::instance()->get_bool("model", "auto_load", true, true);
    // If the model is unloaded, try to auto-load it.
    if (inst->get().meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        if (auto_load) {
            if (method != "POST") {
                res->error(format_error_response(
                    "model is not loaded; load it before using this GET endpoint", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            std::lock_guard lk(slave_mutex_);
            model_registry_.mark_loading(inst->get().key);
            auto rest_res = start_service(req);
            if (rest_res.get()->status != 200) {
                // 加载失败
                return rest_res;
            }
        } else {
            res->error(format_error_response("model not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    const auto inst1 = get_model_inst(name);
    if (inst1 && inst1->get().meta.status == SERVER_MODEL_STATUS_LOADING) {
        std::unique_lock<std::mutex> lk(slave_mutex_);
        slave_cv_.wait(lk, [this, &name]() {
            const auto inst_inner = get_model_inst(name);
            if (inst_inner) {
                return inst_inner->get().meta.status != SERVER_MODEL_STATUS_LOADING;
            }

            return false;
        });
    }

    if (inst1->get().meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        // 加载失败
        res->error("failed to load model: " + name);
        return res;
    }

    // 如果已加载则直接调用
    if (inst1->get().meta.port == port_) {
        const auto model_key = inst1->get().key;
        ModelRequestLease lease(model_registry_, model_key);
        if (!lease.active()) {
            res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_local_llm_or_document(model_key, lease);
    }

    const auto model_key = inst1->get().key;
    ModelRequestLease lease(model_registry_, model_key);
    if (!lease.active()) {
        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    return attach_model_request_lease(proxy_request(req, method, name, inst1->get().meta.port), model_registry_, model_key, lease);
}

std::vector<HLIEWorkflow::RouteConfig> HLIEWorkflow::get_route_configs() {
    auto make_handler = [this](auto func) { return [this, func](const server_http_req& req) { return (this->*func)(req); }; };
    auto route_llm = [this](const server_http_req& req, auto handler, const std::string& method = "POST") -> server_http_res_ptr {
        return route_llm_request(
            req,
            [this, handler](const server_http_req& routed_req) {
                return call_llm_service(model_service_manager_.llm(), [&](const LLModelService& service) {
                    return (service.*handler)(routed_req);
                });
            },
            method);
    };
    return {{
                {"/health", "/v1/health"},
                GET,
                make_handler(&HLIEWorkflow::handle_health),
                false // 不需要模型Running
            },
            {
                {"/v1/models", "/models", "/api/tags"},
                GET,
                make_handler(&HLIEWorkflow::handle_models),
                false // 不需要模型Running
            },
            {
                {"/v1/chat/completions", "/chat/completions", "/api/chat"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_chat_completions);
                },
                true, // 需要模型Running
                LLM // LLM模型类型
            },
            {
                {"/v1/images/generations", "/images/generations"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(model_service_manager_.sd());
                    ModelRequestLease lease(model_registry_, HLIEModelType::SD3);
                    if (!lease.active()) {
                        auto res = std::make_unique<server_res_generator_light>();
                        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
                        return res;
                    }
                    return attach_model_request_lease(model_service_manager_.sd()->handle_image_generations(req),
                                                      model_registry_, HLIEModelType::SD3, lease);
                },
                true,
                SD // 生图模型类型
            },
            {
                {"/v1/audio/transcriptions"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    const json body = json::parse(req.body);
                    const std::string model_name = json_value(body, "model", std::string());
                    const bool force_align_direct = json_boolish(body, "asr_force_align") || json_boolish(body, "force_align");
                    const bool pipeline_requested =
                        requests_word_timestamps(body) &&
                        !force_align_direct &&
                        (!is_whisper_transcription_model(model_name) ||
                         body.contains("force_aligner_model") ||
                         body.contains("aligner_model") ||
                         body.contains("timestamp_model"));
                    if (pipeline_requested) {
                        return handle_transcriptions_pipeline(req, body);
                    }

                    if (model_name.empty() || HlieUtil::contains_ignore_case(model_name, WHISPER)) {
                        auto res = std::make_unique<server_res_generator_light>();
                        if (!model_service_manager_.whisper() || model_service_manager_.whisper()->get_state() != SERVER_STATE_READY) {
                            res->error(format_error_response_ex(
                                "Whisper model is not available for now.", ERROR_TYPE_UNAVAILABLE, 80004));
                            return res;
                        }

                        const auto model_key = model_service_manager_.whisper()->get_model_key();
                        ModelRequestLease lease(model_registry_, model_key);
                        if (!lease.active()) {
                            res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
                            return res;
                        }
                        return attach_model_request_lease(model_service_manager_.whisper()->handle_transcriptions(req),
                                                          model_registry_, model_key, lease);
                    }

                    return route_llm(req, &LLModelService::handle_transcriptions);
                },
                false,
                WHISPER, // 生图模型类型
                nullptr,
            },
        {
                {"/v1/audio/speech"},
                POST,
                nullptr,
                true,
                HLIEModelType::COSYVOICE,
                [this](const httplib::Request& req, httplib::Response& res) {
                    if (!model_service_manager_.cosyvoice() || model_service_manager_.cosyvoice()->get_state() != SERVER_STATE_READY) {
                        auto error_data = format_error_response_ex("Model is not available for now.", ERROR_TYPE_UNAVAILABLE, 80004);
                        const json final_response{{"error", error_data}};
                        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
                        res.status = json_value(error_data, "code", 500);
                        return;
                    }

                    const auto model_key = model_service_manager_.cosyvoice()->get_model_key();
                    ModelRequestLease lease(model_registry_, model_key);
                    if (!lease.active()) {
                        auto error_data = format_error_response_ex("model is not loaded", ERROR_TYPE_INVALID_REQUEST, 80004);
                        const json final_response{{"error", error_data}};
                        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
                        res.status = json_value(error_data, "code", 500);
                        return;
                    }

                    model_service_manager_.cosyvoice()->handle_speech(req, res);
                },
            },
            {
                {"/hlie/v1/devices"},
                GET,
                [this](const server_http_req& req) { return device_service_->handle_devices(req); },
                false // 不需要模型Running
            },
            {
                {"/v1/completions"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_completions_oai);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/completion", "/completions"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_completions);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/v1/responses", "/responses"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_responses);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/embedding", "/embeddings"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_embeddings);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/v1/embeddings"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_embeddings_oai);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/rerank", "/reranking", "/v1/rerank", "/v1/reranking"},
                POST,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm(req, &LLModelService::handle_rerank);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/props"},
                GET,
                [this, route_llm](const server_http_req& req) -> server_http_res_ptr {
                    if (req.get_param("model").empty()) {
                        if (model_service_manager_.llm()) {
                            return model_service_manager_.llm()->handle_props(req);
                        }

                        auto res = std::make_unique<server_res_generator_light>();
                        res->ok({
                            // TODO: add support for this on web UI
                            {"role", "router"},
                            {"max_instances", 4}, // dummy value for testing
                            // this is a dummy response to make sure webui doesn't break
                            {"model_alias", "llama-server"},
                            {"model_path", "none"},
                            {"default_generation_settings",
                             {
                                 {"params", json{}},
                                 {"n_ctx", 0},
                             }},
                        });
                        return res;
                    }

                    return route_llm(req, &LLModelService::handle_props, "GET");
                },
                true,
                LLM // LLM模型类型
            },
            {
                    {"/slots"},
                    GET,
                    [this](const server_http_req& req) -> server_http_res_ptr {
                        CHECK_SERVICE(model_service_manager_.llm());
                        return model_service_manager_.llm()->handle_slots(req);
                    },
                    true,
                    LLM // LLM模型类型
                },
            {
                {"/hlie/v1/start", "/models/load"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return start_service(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/stop"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return stop_service(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/status"},
                GET,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_get_status(req); },
                false // 不需要模型Running
            },
            {
                {"/models/status"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_post_status(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/unload_model", "/models/unload"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_unload_model(req); },
                false // 不需要模型Running
            }};
}

void HLIEWorkflow::set_server_port(const int port) {
    port_ = port;
    if (model_service_manager_.llm()) {
        const auto model_key = model_service_manager_.llm()->get_model_key();
        if (model_registry_.update_port(model_key, port)) {
        } else {
            LOG_ERROR("Cannot set LLM model port because instance is missing: key={}, port={}",
                      json(model_key).dump(), port);
        }
    }

    if (model_service_manager_.sd()) {
        model_registry_.update_port(HLIEModelType::SD3, port);
    }

    if (model_service_manager_.whisper()) {
        model_registry_.update_port(model_service_manager_.whisper()->get_model_key(), port);
    }
}

server_http_res_ptr HLIEWorkflow::stop_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);
    const auto instance_id = body.at("instance_id").get<int>();
    const auto process_id = body.at("process_id").get<int>();

    if (instance_id == 0) {
        std::shared_ptr<LLModelService> llm_to_stop;
        std::shared_ptr<DocumentPipelineService> document_pipeline_to_stop;
        std::shared_ptr<SDModelService> sd_to_stop;
        std::shared_ptr<WhisperModelService> whisper_to_stop;
        std::shared_ptr<CosyVoiceModelService> cosy_voice_to_stop;
        std::vector<std::string> busy_models;
        std::vector<std::string> eviction_keys;
        {
            std::unique_lock lock(mutex_);
            if (model_service_manager_.llm()) {
                const auto model_key = model_service_manager_.llm()->get_model_key();
                if (model_registry_.try_begin_eviction(model_key)) {
                    eviction_keys.push_back(model_key);
                } else {
                    LOG_WARN("Model stop_service skipped busy LLM model: key={}", json(model_key).dump());
                    busy_models.push_back(model_key);
                }
            }

            if (model_service_manager_.sd()) {
                if (model_registry_.try_begin_eviction(HLIEModelType::SD3)) {
                    eviction_keys.push_back(HLIEModelType::SD3);
                } else {
                    LOG_WARN("Model stop_service skipped busy SD model: key={}", json(HLIEModelType::SD3).dump());
                    busy_models.push_back(HLIEModelType::SD3);
                }
            }

            if (model_service_manager_.whisper()) {
                const auto model_key = model_service_manager_.whisper()->get_model_key();
                if (model_registry_.try_begin_eviction(model_key)) {
                    eviction_keys.push_back(model_key);
                } else {
                    LOG_WARN("Model stop_service skipped busy Whisper model: key={}", json(model_key).dump());
                    busy_models.push_back(model_key);
                }
            }

            if (model_service_manager_.cosyvoice()) {
                const auto model_key = model_service_manager_.cosyvoice()->get_model_key();
                if (model_registry_.try_begin_eviction(model_key)) {
                    eviction_keys.push_back(model_key);
                } else {
                    LOG_WARN("Model stop_service skipped busy CosyVoice model: key={}", json(model_key).dump());
                    busy_models.push_back(model_key);
                }
            }

            if (!busy_models.empty()) {
                for (const auto& key : eviction_keys) {
                    model_registry_.abort_eviction(key);
                }
            } else {
                if (model_service_manager_.llm()) {
                    document_pipeline_to_stop = model_service_manager_.detach_document_pipeline();
                    llm_to_stop = model_service_manager_.detach_llm();
                }
                if (model_service_manager_.sd()) {
                    sd_to_stop = model_service_manager_.detach_sd();
                }
                if (model_service_manager_.whisper()) {
                    whisper_to_stop = model_service_manager_.detach_whisper();
                }
                if (model_service_manager_.cosyvoice()) {
                    cosy_voice_to_stop = model_service_manager_.detach_cosyvoice();
                }
            }
        }

        if (!busy_models.empty()) {
            json busy = json::array();
            for (const auto& model : busy_models) {
                busy.push_back(model);
            }
            LOG_WARN("Model stop_service refused because some models are busy: models={}", busy.dump());
            res->error(format_error_response_ex("model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
            return res;
        }

        auto stop_outside_workflow_lock = [](const auto& service, const std::string& service_name) {
            if (!service) {
                return;
            }
            const auto begin = std::chrono::steady_clock::now();
            LOG_WARN("Model stop_service stop begin: type={}", service_name);
            service->stop();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begin).count();
            LOG_WARN("Model stop_service stop done: type={}, elapsed_ms={}", service_name, elapsed_ms);
        };

        stop_outside_workflow_lock(document_pipeline_to_stop, HLIEModelType::DOCUMENT);
        stop_outside_workflow_lock(llm_to_stop, LLM);
        stop_outside_workflow_lock(sd_to_stop, SD);
        stop_outside_workflow_lock(whisper_to_stop, WHISPER);
        stop_outside_workflow_lock(cosy_voice_to_stop, HLIEModelType::COSYVOICE);
        if (llm_to_stop) {
            model_registry_.mark_unloaded(llm_to_stop->get_model_key());
        }
        if (sd_to_stop) {
            model_registry_.mark_unloaded(HLIEModelType::SD3);
        }
        if (whisper_to_stop) {
            model_registry_.mark_unloaded(whisper_to_stop->get_model_key());
        }
        if (cosy_voice_to_stop) {
            model_registry_.mark_unloaded(cosy_voice_to_stop->get_model_key());
        }

        LOG_INFO("Succeed to stop service process for instance_id[{}][{}]", instance_id, process_id);
    } else if (process_id != 0) {
        const auto& process_manager = Singleton<HLIEProcessManager>::instance();
        const auto& models = process_manager->get_working_models(instance_id);

        for (const auto& model : models) {
            auto model_inst = get_model_inst(model);
            if (model_inst) {
                model_registry_.mark_unloaded(model_inst->get().key);
            }
        }

        if (!process_manager->terminate_process(process_id)) {
            LOG_ERROR("Failed to terminate process for instance_id[{}][{}], maybe has been killed or something", instance_id, process_id);
            const json resp = {{"ret", 80001}, "Failed to terminate process for instance_id, maybe has been killed or something"};
            res->error(resp);
            return res;
        }

        LOG_INFO("Succeed to terminate process for instance_id[{}][{}]", instance_id, process_id);
    }

    const json resp = {
        {"code", 0},
        {"instance_id", instance_id},
        {"process_id", process_id},
    };

    res->ok(resp);
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_get_status(const server_http_req& req) {
    const std::vector<HLIEProcessManager::ProcessInfo> process_infos = Singleton<HLIEProcessManager>::instance()->get_running_processes();

    const auto public_model_names = [this](const std::vector<std::string>& names) {
        std::vector<std::string> result;
        std::unordered_set<std::string> seen;
        for (const auto& name : names) {
            const auto model_inst = get_model_inst(name);
            const std::string public_name = model_inst ? model_inst->get().model_info.get_model_name() : name;
            if (public_name.empty()) {
                continue;
            }
            if (seen.insert(public_name).second) {
                result.push_back(public_name);
            }
        }
        return result;
    };

    json processors = json::array();
    json models = json::array();
    if (model_service_manager_.llm()) {
        const auto model_inst = get_model_inst(model_service_manager_.llm()->get_model_key());
        const std::string model_name =
            model_inst ? model_inst->get().model_info.get_model_name() : model_service_manager_.llm()->get_model_key();
        if (!model_name.empty()) {
            models.push_back(model_name);
        }
    }

    if (model_service_manager_.sd()) {
        models.push_back(HLIEModelType::SD3);
    }

    if (model_service_manager_.whisper()) {
        const auto& model_name = model_service_manager_.whisper()->get_model_key();
        if (!model_name.empty()) {
            models.push_back(model_name);
        }
    }

    if (model_service_manager_.cosyvoice()) {
        const auto& model_name = model_service_manager_.cosyvoice()->get_model_key();
        if (!model_name.empty()) {
            models.push_back(model_name);
        }
    }

    processors.push_back({{"params", ""},
                          {"instance_id", 0},
                          {"port", port_},
                          {"uuid", ""},
                          {"process_id", 0},
                          {"start_time", start_time_},
                          {"models", models}});

    for (const auto& p : process_infos) {
        processors.push_back({
            {"params", p.params},
            {"instance_id", p.instance_id},
            {"port", p.instance_id + port_},
            {"uuid", p.instance_id},
            {"process_id", p.process_id},
            {"start_time", p.start_time},
            {"models", public_model_names(p.model_names)},
        });
    }

    const json resp = {{"code", 0}, {"processor", processors}};
    auto res = std::make_unique<server_res_generator_light>();
    res->ok(resp);
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_post_status(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
    const json body = json::parse(req.body);
    std::string model = json_value(body, "model", std::string());
    const std::string value = json_value(body, "value", std::string());
    const int port = json_value(body, "port", 0);

    const auto status = server_model_status_from_string(value);
    LOG_INFO("Received model status update: model={}, model_length={}, value={}, port={}",
             json(model).dump(), model.size(), value, port);

    if (const auto inst = get_model_inst(model); inst.has_value()) {
        std::lock_guard lk(mutex_);
        const auto resolved_key = inst->get().key;
        const auto resolved_model_name = inst->get().model_info.get_model_name();
        model_registry_.update_status(resolved_key, status, port);
        LOG_INFO("Model status updated: request_model={}, resolved_key={}, resolved_model_name={}, status={}, port={}",
                 json(model).dump(), json(resolved_key).dump(), json(resolved_model_name).dump(),
                 static_cast<int>(status), port);

        slave_cv_.notify_all();

        if (status == SERVER_MODEL_STATUS_UNLOADED) {
            Singleton<HLIEProcessManager>::instance()->erase_model(resolved_model_name);
        } else if (status == SERVER_MODEL_STATUS_LOADED) {
            const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
            const int instance_id = port - master_port;
            Singleton<HLIEProcessManager>::instance()->add_model(resolved_model_name, instance_id);
        }
    } else {
        LOG_ERROR("Model status update ignored because lookup failed: model={}, value={}, port={}",
                  json(model).dump(), value, port);
    }

    res->ok({{"success", true}});
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_unload_model(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);
    const std::string model_type = body.value("model_type", LLM);
    if (!HLIEModelServiceFactory::is_supported_model_type(model_type)) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    std::shared_ptr<LLModelService> llm_to_stop;
    std::shared_ptr<DocumentPipelineService> document_pipeline_to_stop;
    std::shared_ptr<SDModelService> sd_to_stop;
    std::shared_ptr<WhisperModelService> whisper_to_stop;
    std::shared_ptr<CosyVoiceModelService> cosy_voice_to_stop;
    std::string notify_model_key;
    bool unload_refused = false;
    {
        std::unique_lock lock(mutex_);
        if (model_type == LLM && model_service_manager_.llm()) {
            notify_model_key = model_service_manager_.llm()->get_model_key();
            if (model_registry_.try_begin_eviction(notify_model_key)) {
                document_pipeline_to_stop = model_service_manager_.detach_document_pipeline();
                llm_to_stop = model_service_manager_.detach_llm();
            } else {
                unload_refused = true;
            }
        } else if (model_type == SD && model_service_manager_.sd()) {
            notify_model_key = HLIEModelType::SD3;
            if (model_registry_.try_begin_eviction(notify_model_key)) {
                sd_to_stop = model_service_manager_.detach_sd();
            } else {
                unload_refused = true;
            }
        } else if (model_type == WHISPER && model_service_manager_.whisper()) {
            notify_model_key = model_service_manager_.whisper()->get_model_key();
            if (model_registry_.try_begin_eviction(notify_model_key)) {
                whisper_to_stop = model_service_manager_.detach_whisper();
            } else {
                unload_refused = true;
            }
        } else if (model_type == HLIEModelType::COSYVOICE && model_service_manager_.cosyvoice()) {
            notify_model_key = model_service_manager_.cosyvoice()->get_model_key();
            if (model_registry_.try_begin_eviction(notify_model_key)) {
                cosy_voice_to_stop = model_service_manager_.detach_cosyvoice();
            } else {
                unload_refused = true;
            }
        }
    }

    if (unload_refused) {
        LOG_WARN("Model unload refused because model is busy: type={}, model={}", model_type, json(notify_model_key).dump());
        res->error(format_error_response_ex("model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
        return res;
    }

    auto stop_outside_workflow_lock = [&](const auto& service, const std::string& service_name) {
        if (!service) {
            return;
        }
        const auto begin = std::chrono::steady_clock::now();
        LOG_WARN("Model unload stop begin: type={}, model={}", service_name, json(notify_model_key).dump());
        service->stop();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - begin).count();
        LOG_WARN("Model unload stop done: type={}, model={}, elapsed_ms={}",
                 service_name, json(notify_model_key).dump(), elapsed_ms);
    };

    stop_outside_workflow_lock(document_pipeline_to_stop, HLIEModelType::DOCUMENT);
    stop_outside_workflow_lock(llm_to_stop, LLM);
    stop_outside_workflow_lock(sd_to_stop, SD);
    stop_outside_workflow_lock(whisper_to_stop, WHISPER);
    stop_outside_workflow_lock(cosy_voice_to_stop, HLIEModelType::COSYVOICE);
    if (!notify_model_key.empty()) {
        model_registry_.mark_unloaded(notify_model_key);
        slave_cv_.notify_all();
    }

    if (!notify_model_key.empty() && is_slave_ && !replicas_mode_) {
        update_status(master_port_, notify_model_key, port_, SERVER_MODEL_STATUS_UNLOADED);
    }

    LOG_INFO("Succeed to uninstall model");

    const json resp = {
        {"code", 0},
    };

    res->ok(resp);
    return res;
}

bool HLIEWorkflow::evict_model(const ModelInstance& model_inst, const std::string& reason) {
    const auto matches = [&](const std::string& name) {
        const std::string requested = normalize_model_name(name);
        return !requested.empty() &&
            (normalize_model_name(model_inst.key) == requested ||
             normalize_model_name(model_inst.model_info.get_model_name()) == requested ||
             normalize_model_name(model_inst.meta.name) == requested);
    };

    LOG_WARN("Model residency: evicting model key={}, model_name={}, port={}, reason={}",
             json(model_inst.key).dump(), json(model_inst.model_info.get_model_name()).dump(), model_inst.meta.port, reason);

    if (!model_registry_.try_begin_eviction(model_inst.key)) {
        return false;
    }

    bool evicted = false;
    std::string local_service_type;
    std::shared_ptr<LLModelService> llm_to_stop;
    std::shared_ptr<DocumentPipelineService> document_pipeline_to_stop;
    std::shared_ptr<SDModelService> sd_to_stop;
    std::shared_ptr<WhisperModelService> whisper_to_stop;
    std::shared_ptr<CosyVoiceModelService> cosy_voice_to_stop;
    {
        std::unique_lock lock(mutex_);
        if (model_service_manager_.llm() && matches(model_service_manager_.llm()->get_model_key())) {
            document_pipeline_to_stop = model_service_manager_.detach_document_pipeline();
            llm_to_stop = model_service_manager_.detach_llm();
            local_service_type = LLM;
            evicted = true;
        } else if (model_service_manager_.sd() && matches(HLIEModelType::SD3)) {
            sd_to_stop = model_service_manager_.detach_sd();
            local_service_type = SD;
            evicted = true;
        } else if (model_service_manager_.whisper() && matches(model_service_manager_.whisper()->get_model_key())) {
            whisper_to_stop = model_service_manager_.detach_whisper();
            local_service_type = WHISPER;
            evicted = true;
        } else if (model_service_manager_.cosyvoice() && matches(model_service_manager_.cosyvoice()->get_model_key())) {
            cosy_voice_to_stop = model_service_manager_.detach_cosyvoice();
            local_service_type = HLIEModelType::COSYVOICE;
            evicted = true;
        }
    }

    auto stop_evicted_service = [&](const auto& service) {
        if (!service) {
            return;
        }
        const auto begin = std::chrono::steady_clock::now();
        LOG_WARN("Model residency: local stop begin key={}, type={}, reason={}",
                 json(model_inst.key).dump(), local_service_type, reason);
        service->stop();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - begin).count();
        LOG_WARN("Model residency: local stop done key={}, type={}, reason={}, elapsed_ms={}",
                 json(model_inst.key).dump(), local_service_type, reason, elapsed_ms);
    };

    stop_evicted_service(document_pipeline_to_stop);
    stop_evicted_service(llm_to_stop);
    stop_evicted_service(sd_to_stop);
    stop_evicted_service(whisper_to_stop);
    stop_evicted_service(cosy_voice_to_stop);

    if (!evicted) {
        const auto processes = Singleton<HLIEProcessManager>::instance()->get_running_processes();
        for (const auto& process : processes) {
            const auto it = std::find_if(process.model_names.begin(), process.model_names.end(), [&](const std::string& name) {
                return matches(name);
            });
            if (it == process.model_names.end()) {
                continue;
            }

            evicted = Singleton<HLIEProcessManager>::instance()->terminate_process(process.process_id);
            if (!evicted) {
                LOG_ERROR("Model residency: failed to terminate process {} for model key={}",
                          process.process_id, json(model_inst.key).dump());
                model_registry_.abort_eviction(model_inst.key);
                return false;
            }
            break;
        }
    }

    if (!evicted) {
        LOG_WARN("Model residency: no running service or process matched eviction candidate key={}", json(model_inst.key).dump());
        model_registry_.abort_eviction(model_inst.key);
        return false;
    }

    model_registry_.mark_unloaded(model_inst.key);
    Singleton<HLIEProcessManager>::instance()->erase_model(model_inst.model_info.get_model_name());
    slave_cv_.notify_all();
    LOG_WARN("Model residency: evicted model key={}, model_name={}",
             json(model_inst.key).dump(), json(model_inst.model_info.get_model_name()).dump());
    return true;
}

void HLIEWorkflow::start_residency_ttl_worker() {
    if (residency_ttl_thread_) {
        return;
    }

    const auto policy = build_residency_policy();
    residency_ttl_stop_.store(false);
    residency_ttl_thread_ = std::make_unique<std::thread>(&HLIEWorkflow::residency_ttl_loop, this);
    LOG_INFO("Model residency TTL worker started: auto_unload={}, idle_ttl_sec={}",
             policy.auto_unload, policy.idle_ttl_sec);
}

void HLIEWorkflow::stop_residency_ttl_worker() {
    residency_ttl_stop_.store(true);
    if (!residency_ttl_thread_) {
        return;
    }

    if (residency_ttl_thread_->joinable()) {
        residency_ttl_thread_->join();
    }
    residency_ttl_thread_.reset();
    LOG_INFO("Model residency TTL worker stopped");
}

void HLIEWorkflow::residency_ttl_loop() {
    HLIERkPerformance::apply_for_current_thread();

    while (!residency_ttl_stop_.load()) {
        const auto policy = build_residency_policy();
        const int ttl_sec = policy.idle_ttl_sec;
        const int sleep_sec = ttl_sec > 0 ? std::clamp(ttl_sec / 2, 5, 60) : 30;

        for (int i = 0; i < sleep_sec && !residency_ttl_stop_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (residency_ttl_stop_.load()) {
            break;
        }

        if (!policy.auto_unload || ttl_sec <= 0) {
            continue;
        }

        const HLIEModelResidencyManager residency(policy);
        const auto candidate = residency.select_for_idle_ttl(model_registry_.get_all_instances(), std::time(nullptr));
        if (!candidate) {
            continue;
        }

        LOG_WARN("Model residency: idle TTL candidate key={}, model_name={}, ttl_sec={}, hits={}",
                 json(candidate->key).dump(), json(candidate->model_name).dump(), ttl_sec, candidate->hit_count);

        const auto inst = get_model_inst(candidate->key);
        if (!inst) {
            LOG_WARN("Model residency: idle TTL candidate disappeared before eviction: key={}", json(candidate->key).dump());
            continue;
        }

        const auto model_copy = inst->get();
        evict_model(model_copy, "idle_ttl");
    }
}

std::vector<server_model_meta> HLIEWorkflow::get_all_model_metas() {
    return model_registry_.get_all_metas();
}

void HLIEWorkflow::register_model_instance(HLIEModelInfo& model_info) {
    model_registry_.register_model(model_info);
}

void HLIEWorkflow::rollback_model_start(const std::string& model_key, const std::string& reason) {
    LOG_WARN("Model start rollback: key={}, reason={}", json(model_key).dump(), reason);
    model_registry_.mark_unloaded(model_key);
    slave_cv_.notify_all();
}

void HLIEWorkflow::mark_local_model_started(const std::string& model_key,
                                            const std::string& notify_model_key,
                                            const bool enable_asr_websocket) {
    if (enable_asr_websocket) {
        ensure_asr_websocket();
    }

    model_registry_.mark_loaded(model_key, port_);
    if (port_ != master_port_ && !replicas_mode_) {
        update_status(master_port_, notify_model_key, port_);
    }
}

std::optional<server_model_meta> HLIEWorkflow::get_model_meta(const std::string& model_name) {
    return model_registry_.get_meta(model_name);
}

std::optional<std::reference_wrapper<HLIEWorkflow::ModelInstance>> HLIEWorkflow::get_model_inst(const std::string& model_name) {
    return model_registry_.get_instance(model_name);
}

HLIE_NAMESPACE_END
