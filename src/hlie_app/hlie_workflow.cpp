#include "hlie_app/hlie_workflow.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <server/server-http.h>
#include <sstream>
#include <tuple>
#include <unordered_set>

#include "INIReader.h"
#include "hlie_app/document/hlie_document_input_adapter.h"
#include "hlie_app/document/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/util/hlie_model_load_waiter.h"
#include "hlie_app/hlie_model_residency_manager.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_qwen_asr_stream_decoder.h"
#include "hlie_app/hlie_replica_config.h"
#include "hlie_app/hlie_server_http_proxy.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_gguf_bundle.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_rk_performance.h"
#include "hlie_base/hlie_util.h"

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_qwen3_tts_model_service.h"
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
    template <typename Callback>
    class ScopeExit {
    public:
        explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
        ~ScopeExit() noexcept {
            if (active_) {
                try {
                    callback_();
                } catch (...) {
                }
            }
        }

        ScopeExit(const ScopeExit&) = delete;
        ScopeExit& operator=(const ScopeExit&) = delete;

        void release() noexcept { active_ = false; }

    private:
        Callback callback_;
        bool active_{true};
    };

    template <typename Callback>
    ScopeExit<Callback> make_scope_exit(Callback callback) {
        return ScopeExit<Callback>(std::move(callback));
    }

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

    ProxyTimeoutOptions build_proxy_timeout_options() {
        const auto& config = Singleton<HLIEConfigService>::instance();
        ProxyTimeoutOptions options;
        options.connect_timeout_ms = config->get_int(
            "proxy", "connect_timeout_ms", options.connect_timeout_ms, true);
        options.non_stream_read_timeout_sec = config->get_int(
            "proxy", "non_stream_read_timeout_sec", options.non_stream_read_timeout_sec, true);
        options.stream_idle_timeout_sec = config->get_int(
            "proxy", "stream_idle_timeout_sec", options.stream_idle_timeout_sec, true);
        options.write_timeout_sec = config->get_int(
            "proxy", "write_timeout_sec", options.write_timeout_sec, true);
        options.max_request_timeout_sec = config->get_int(
            "proxy", "max_request_timeout_sec", options.max_request_timeout_sec, true);
        return options.normalized();
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

    std::vector<HLIEModelInfo> load_sd_model_manifests_from_directory(const std::string& directory_path) {
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
                        if (model_info.get_service_kind() != "sd") {
                            LOG_WARN("Ignoring non-SD JSON model manifest; non-SD models must be discovered from GGUF: {}",
                                     file_path);
                            continue;
                        }
                        model_infos.push_back(std::move(model_info));
                        LOG_INFO("Successfully parsed SD model manifest from: {}", file_path);
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

        LOG_INFO("Loaded {} SD model manifest files from directory: {}", model_infos.size(), directory_path);
        return model_infos;
    }

    std::optional<std::string> build_model_directory_snapshot(const std::string& directory_path) {
        std::error_code ec;
        if (!std::filesystem::is_directory(directory_path, ec) || ec) {
            LOG_WARN("Cannot snapshot model directory: path={}, error={}", directory_path, ec.message());
            return std::nullopt;
        }

        using SnapshotEntry = std::tuple<std::string, std::uintmax_t, std::int64_t>;
        std::vector<SnapshotEntry> entries;
        for (std::filesystem::directory_iterator it(directory_path, ec), end; it != end && !ec; it.increment(ec)) {
            std::error_code entry_ec;
            if (!it->is_regular_file(entry_ec) || entry_ec) {
                continue;
            }

            std::string extension = it->path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension != ".gguf" && extension != ".json") {
                continue;
            }

            const auto size = it->file_size(entry_ec);
            if (entry_ec) {
                LOG_DEBUG("Skipping model discovery snapshot entry {}: {}", it->path().string(), entry_ec.message());
                continue;
            }
            const auto write_time = it->last_write_time(entry_ec);
            if (entry_ec) {
                LOG_DEBUG("Skipping model discovery snapshot entry {}: {}", it->path().string(), entry_ec.message());
                continue;
            }
            entries.emplace_back(it->path().filename().generic_string(), size,
                                 static_cast<std::int64_t>(write_time.time_since_epoch().count()));
        }
        if (ec) {
            LOG_WARN("Model directory snapshot stopped early: path={}, error={}", directory_path, ec.message());
            return std::nullopt;
        }

        std::sort(entries.begin(), entries.end());
        std::ostringstream snapshot;
        for (const auto& [name, size, write_time] : entries) {
            snapshot << name.size() << ':' << name << ':' << size << ':' << write_time << '\n';
        }
        return snapshot.str();
    }

    bool validate_discovered_gguf(const std::string& model_dir, const std::string& configured_path) {
        if (configured_path.empty()) {
            return true;
        }

        std::filesystem::path path(configured_path);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".gguf") {
            return true;
        }
        if (path.is_relative()) {
            path = std::filesystem::path(model_dir) / path;
        }

        HLIEGgufBundle bundle;
        std::string error;
        if (!bundle.open(path.string(), &error)) {
            LOG_WARN("Ignoring incomplete or invalid GGUF during discovery: path={}, reason={}",
                     path.string(), error);
            return false;
        }
        return true;
    }

    bool is_discovered_model_complete(const std::string& model_dir, const HLIEModelInfo& model_info) {
        return validate_discovered_gguf(model_dir, model_info.get_model_path()) &&
               validate_discovered_gguf(model_dir, model_info.get_mm_proj_path());
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

    std::string generate_launch_manifest(const HLIEModelInfo& model_info, const std::string& model_dir,
                                         const std::unique_ptr<server_res_generator_light>& res) {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto t = system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        const auto meta_info_file_name = "model_launch_" + oss.str() + ".hlif";
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
        if (model_info.to_json_file(json_info_file, true) < 0) {
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
    if (!initialize_llm_preset_catalogs()) {
        return -1;
    }
    replicas_mode_ = HLIEReplicaConfig::load().enabled();
    master_port_ = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    std::vector<HLIEModelInfo> model_infos;
    const auto launch_manifest_arg = args.find("--launch_manifest") != args.end()
        ? args.find("--launch_manifest")
        : args.find("--json_info");
    if (launch_manifest_arg != args.end()) { // Slave process startup from generated launch manifest.
        const std::string json_path = launch_manifest_arg->second;
        if (HLIEModelInfo model_info; model_info.parse_from_file(json_path) == 0) {
            if (!resolve_llm_launch_params(model_info)) {
                LOG_ERROR("Failed to resolve llama params from launch manifest: {}", json_path);
                return -1;
            }
            model_infos.emplace_back(model_info);
            LOG_INFO("Successfully parsed model launch manifest: {}", json_path);
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
            case ModelServiceKind::Qwen3TTS: {
                LOG_INFO("Initializing Qwen3TTSModelService");
                if (!model_service_manager_.load_qwen3_tts(model_dir, model_info, device_service_)) {
                    LOG_ERROR("failed to init the Qwen3-TTS model service");
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

            start_time_ = time(nullptr);
            return 0;
        }

        LOG_ERROR("Failed to parse model launch manifest: {}", json_path);
        return -1;
    }

    // 主进程或者是负载均衡模式
    model_infos = load_sd_model_manifests_from_directory(model_dir);
    HLIEModelInfo::ScanLocalModelsFromDir(model_infos, model_dir_);
    model_infos.erase(std::remove_if(model_infos.begin(), model_infos.end(),
                                     [this](const HLIEModelInfo& model_info) {
                                         return !is_discovered_model_complete(model_dir_, model_info);
                                     }),
                      model_infos.end());
    for (auto& model_info : model_infos) {
        register_model_instance(model_info);
    }
    initialize_model_discovery_snapshot();

    if (model_infos.empty()) {
        LOG_WARN("No valid model info files found in directory: {}", model_dir);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
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
        case ModelServiceKind::Qwen3TTS: {
            LOG_INFO("Initializing Qwen3TTSModelService");
            if (!model_service_manager_.load_qwen3_tts(model_dir, default_model_info, device_service_)) {
                LOG_ERROR("failed to init the Qwen3-TTS model service");
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

                default_model_info.set_mm_proj_path(mmproj_path, true);
                if (!model_registry_.contains(default_model_info.model_key_)) {
                    register_model_instance(default_model_info);
                }
            }

            if (!resolve_llm_launch_params(default_model_info)) {
                LOG_ERROR("failed to resolve llama params for default model: {}", default_model_info.model_key_);
                return -1;
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
        model_registry_.update_runtime_devices(model_key, model_service_manager_.llm()->get_devices());
        const auto inst = get_model_inst(model_key);
        if (!inst) {
            LOG_ERROR("LLM model instance is missing after initialization: key={}", json(model_key).dump());
            return;
        }
        model_registry_.mark_loaded(model_key, inst->meta.port);

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
        model_registry_.update_runtime_devices(HLIEModelType::SD3, model_service_manager_.sd()->get_devices());
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
        model_registry_.update_runtime_devices(model_key, model_service_manager_.whisper()->get_devices());
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
        model_registry_.update_runtime_devices(model_key, model_service_manager_.cosyvoice()->get_devices());
        model_registry_.mark_loaded(model_key, port_);
        if (is_slave_ && !replicas_mode_) {
            std::thread([this, model_key] {
                model_registry_.update_port(model_key, port_);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, model_key, port_);
            }).detach();
        }
    }

    if (model_service_manager_.qwen3_tts()) {
        model_service_manager_.qwen3_tts()->run();
        const auto model_key = model_service_manager_.qwen3_tts()->get_model_key();
        model_registry_.update_runtime_devices(model_key, model_service_manager_.qwen3_tts()->get_devices());
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
    start_model_discovery_worker();
}

void HLIEWorkflow::stop() {
    auto_load_tasks_.stop_and_join();
    stop_model_discovery_worker();
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

    if (model_service_manager_.qwen3_tts()) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, model_service_manager_.qwen3_tts()->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        model_service_manager_.stop_and_reset_qwen3_tts();
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
    bool filter_qwen_markers = is_qwen3_asr_name(model);
    if (!filter_qwen_markers) {
        if (const auto inst = get_model_inst(model)) {
            filter_qwen_markers = is_qwen3_asr_name(inst->key) ||
                                  is_qwen3_asr_name(inst->model_info.get_model_name());
        }
    }
    const raw_buffer wav = pcmf32_to_wav_bytes(request.pcmf32, request.sample_rate);
    const json body = {
        {"model", model},
        {"response_format", request.response_format.empty() ? "json" : request.response_format},
        {"stream", std::string(LLAMA_TRANSCRIPTION_SDK_STREAM_ENABLED)},
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
        const bool sse_transport = response->content_type.find("text/event-stream") != std::string::npos;
        HLIEQwenAsrStreamDecoder decoder(sse_transport, callback, filter_qwen_markers);
        std::string chunk;
        while (true) {
            const bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                decoder.push(chunk);
            }
            chunk.clear();
            if (!has_next) {
                break;
            }
        }
        transcript = decoder.finish();
        if (!decoder.saw_stream_delta()) {
            LOG_ERROR("ASR websocket streaming backend returned no transcript delta: model={}, content_type={}",
                      model, response->content_type);
            return "HLIECpp-FAILED-888";
        }
        if (decoder.final_text_mismatch()) {
            LOG_WARN("ASR websocket Qwen stream final text differs from emitted deltas: model={}, chars={}",
                     model, transcript.size());
        }
    } else {
        LOG_ERROR("ASR websocket backend ignored stream=true: model={}, content_type={}",
                  model, response->content_type);
        return "HLIECpp-FAILED-888";
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

    if (model_service_manager_.qwen3_tts()) {
        return model_service_manager_.qwen3_tts()->get_state();
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

    const auto running_processes = Singleton<HLIEProcessManager>::instance()->get_running_processes();
    struct local_model_instance {
        std::string model_key;
        std::vector<int> device_ids;
    };
    std::vector<local_model_instance> local_instances;
    {
        std::lock_guard lock(mutex_);
        if (model_service_manager_.llm()) {
            local_instances.push_back({model_service_manager_.llm()->get_model_key(),
                                       model_service_manager_.llm()->get_devices()});
        }
        if (model_service_manager_.sd()) {
            local_instances.push_back({HLIEModelType::SD3, model_service_manager_.sd()->get_devices()});
        }
        if (model_service_manager_.whisper()) {
            local_instances.push_back({model_service_manager_.whisper()->get_model_key(),
                                       model_service_manager_.whisper()->get_devices()});
        }
        if (model_service_manager_.cosyvoice()) {
            local_instances.push_back({model_service_manager_.cosyvoice()->get_model_key(),
                                       model_service_manager_.cosyvoice()->get_devices()});
        }
        if (model_service_manager_.qwen3_tts()) {
            local_instances.push_back({model_service_manager_.qwen3_tts()->get_model_key(),
                                       model_service_manager_.qwen3_tts()->get_devices()});
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
        json model_json{
            {"id", meta.name},
            {"aliases", meta.aliases},
            {"tags", meta.tags},
            {"object", "model"}, // for OAI-compat
            {"owned_by", "HLIECpp"}, // for OAI-compat
            {"created", t}, // for OAI-compat
            {"port", meta.port},
            {"status", status},
            {"instances", json::array()},
            // TODO: add other fields, may require reading GGUF metadata
        };

        if (meta.status == SERVER_MODEL_STATUS_LOADED ||
            meta.status == SERVER_MODEL_STATUS_SLEEPING) {
            const auto append_instance = [&](const int instance_id, const ProcessID process_id,
                                              const int port, const std::vector<int>& device_ids) {
                json instance{
                    {"instance_id", instance_id},
                    {"process_id", process_id},
                    {"port", port},
                    {"status", server_model_status_to_string(meta.status)},
                    {"device_ids", device_ids},
                };
                model_json["instances"].push_back(std::move(instance));
            };

            const auto local_instance = std::find_if(local_instances.begin(), local_instances.end(),
                                                     [&meta](const local_model_instance& instance) {
                                                         return instance.model_key == meta.name;
                                                     });
            if (local_instance != local_instances.end()) {
                std::vector<int> device_ids = local_instance->device_ids;
                if (device_ids.empty()) {
                    if (const auto it = meta.loaded_info.find("device_ids");
                        it != meta.loaded_info.end() && it->is_array()) {
                        device_ids = it->get<std::vector<int>>();
                    }
                }
                append_instance(0, 0, port_, device_ids);
            }

            for (const auto& process : running_processes) {
                const bool matches = std::any_of(process.model_names.begin(), process.model_names.end(),
                                                 [this, &meta](const std::string& name) {
                                                     const auto inst = get_model_inst(name);
                                                     return inst && inst->key == meta.name;
                                                 });
                if (matches) {
                    append_instance(process.instance_id, process.process_id,
                                    port_ + process.instance_id, process.device_ids);
                }
            }
        }

        if (meta.loaded_info.is_object()) {
            if (const auto it = meta.loaded_info.find("model_type"); it != meta.loaded_info.end() && it->is_string()) {
                model_json["model_type"] = *it;
            }
            for (const auto* field : {"service_kind", "role", "model_family", "architecture", "llama_params_hash"}) {
                if (const auto it = meta.loaded_info.find(field); it != meta.loaded_info.end() && it->is_string() && !it->get<std::string>().empty()) {
                    model_json[field] = *it;
                }
            }
            if (const auto it = meta.loaded_info.find("multimodal"); it != meta.loaded_info.end() && it->is_boolean()) {
                model_json["multimodal"] = *it;
            }
            if (const auto it = meta.loaded_info.find("capabilities"); it != meta.loaded_info.end() && it->is_object()) {
                model_json["capabilities"] = *it;
            }
            if (const auto it = meta.loaded_info.find("llama_presets"); it != meta.loaded_info.end() && it->is_array()) {
                model_json["llama_presets"] = *it;
            }
            if (const auto it = meta.loaded_info.find("llama_params_summary");
                it != meta.loaded_info.end() && it->is_object()) {
                model_json["llama_params_summary"] = *it;
            }
        }
        models_json.push_back(std::move(model_json));
    }
    res->ok({
        {"data", models_json},
        {"object", "list"},
    });
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_model_discovery_refresh(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
    json body = json::object();
    if (!req.body.empty()) {
        body = json::parse(req.body);
    }

    const bool wait = !body.contains("wait") || json_boolish(body, "wait");
    const uint64_t generation = request_model_discovery_scan(true);
    if (generation == 0) {
        res->ok({
            {"code", 0},
            {"generation", 0},
            {"status", "skipped"},
        });
        return res;
    }

    const bool completed = wait && wait_for_model_discovery(generation);
    res->ok({
        {"code", 0},
        {"generation", generation},
        {"status", wait ? (completed ? "completed" : "timeout") : "queued"},
    });
    return res;
}

server_http_res_ptr HLIEWorkflow::start_service(const server_http_req& req,
                                                const std::string& preclaimed_model_key) {
    bool preclaim_rollback_pending = !preclaimed_model_key.empty();
    auto preclaim_rollback = make_scope_exit([this, &preclaimed_model_key, &preclaim_rollback_pending] {
        if (preclaim_rollback_pending) {
            rollback_model_start(preclaimed_model_key, "auto_load_start_failed");
        }
    });
    const auto rollback_start = [this, &preclaimed_model_key, &preclaim_rollback_pending](
                                    const std::string& model_key, const std::string& reason) {
        if (preclaim_rollback_pending && model_key == preclaimed_model_key) {
            preclaim_rollback_pending = false;
        }
        rollback_model_start(model_key, reason);
    };
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
    if (model_type == HLIEModelType::TTS) {
        // Keep the historical generic TTS alias bound to CosyVoice. Qwen3-TTS
        // has its own explicit model type and must not hijack legacy clients.
        model_type = HLIEModelType::COSYVOICE;
    }
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
    std::optional<ModelInstance> resolved_model_inst = std::nullopt;
    std::optional<HLIEModelInfo> path_llm_model_info = std::nullopt;
    std::string resolved_model_key;
    int mem_est;
    int required_device_num = HLIEModelInfo::estimate_device_num(model_name);

    auto prepare_path_llm_model_info = [&]() {
        if (path_llm_model_info) {
            return true;
        }

        HLIEModelInfo model_info;
        if (model_info.parse_from_gguf_filename(model_path) != 0) {
            res->error(format_error_response_ex("failed to parse model_path.", ERROR_TYPE_INVALID_REQUEST, 80006));
            return false;
        }

        if (body.contains("params") && body["params"].is_object()) {
            const auto& params = body["params"];
            if (const std::string mm_proj_path = params.value("mm_proj_path", std::string{}); !mm_proj_path.empty()) {
                if (std::error_code ec;
                    !std::filesystem::exists(mm_proj_path, ec) || !std::filesystem::is_regular_file(mm_proj_path, ec)) {
                    res->error(format_error_response_ex("mm_proj_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
                    return false;
                }
                model_info.set_mm_proj_path(mm_proj_path, true);
            }
        }

        resolved_model_key = HLIEModelInfo::build_model_key(model_info);
        path_llm_model_info = std::move(model_info);
        return true;
    };

    auto validate_requested_llm_paths = [&](const ModelInstance& existing) {
        if (!path_llm_model_info) {
            return true;
        }

        const bool model_path_matched = HlieUtil::paths_refer_to_same_file(
            existing.path, path_llm_model_info->get_model_path(), model_dir_);
        const bool mmproj_path_matched = HlieUtil::paths_refer_to_same_file(
            existing.path_mmproj, path_llm_model_info->get_mm_proj_path(), model_dir_);
        if (model_path_matched && mmproj_path_matched) {
            return true;
        }

        LOG_ERROR("Model key path conflict: key={}, registered_model_path={}, requested_model_path={}, "
                  "registered_mmproj={}, requested_mmproj={}",
                  json(existing.key).dump(), json(existing.path).dump(),
                  json(path_llm_model_info->get_model_path()).dump(), json(existing.path_mmproj).dump(),
                  json(path_llm_model_info->get_mm_proj_path()).dump());
        res->error(format_error_response_ex(
            "model key is already registered with different model or mmproj paths.",
            ERROR_TYPE_INVALID_REQUEST, 80006));
        return false;
    };

    if (!model_name.empty()) { // Compatibility path: check whether the model exists.
        meta = get_model_meta(model_name);
        if (!meta.has_value()) {
            wait_for_model_discovery(request_model_discovery_scan());
            meta = get_model_meta(model_name);
            if (!meta.has_value()) {
                res->error(format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
        }

        if (meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        resolved_model_inst = get_model_inst(model_name);
        if (!resolved_model_inst) {
            res->error(format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        resolved_model_key = resolved_model_inst->key;
        mem_est = resolved_model_inst->model_info.get_estimated_mem();
        required_device_num = resolved_model_inst->model_info.get_device_num();
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
        if (model_type == LLM) {
            if (!prepare_path_llm_model_info()) {
                return res;
            }
            resolved_model_inst = model_registry_.get_exact_instance_snapshot(*path_llm_model_info);
            if (resolved_model_inst) {
                if (!validate_requested_llm_paths(*resolved_model_inst)) {
                    return res;
                }
                meta = resolved_model_inst->meta;
                required_device_num = resolved_model_inst->model_info.get_device_num();
            } else {
                required_device_num = path_llm_model_info->get_device_num();
            }
        } else {
            required_device_num = HLIEModelInfo::estimate_device_num(model_name);
            meta = get_model_meta(model_name);
        }
        if (meta && meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        mem_est = resolved_model_inst ? resolved_model_inst->model_info.get_estimated_mem()
                                      : HLIEModelInfo::estimate_empirical_mem(model_name);
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

    if (model_type == LLM && !model_path.empty() && resolved_model_key.empty()) {
        if (!prepare_path_llm_model_info()) {
            return res;
        }
        resolved_model_inst = model_registry_.get_exact_instance_snapshot(*path_llm_model_info);
        if (resolved_model_inst) {
            if (!validate_requested_llm_paths(*resolved_model_inst)) {
                return res;
            }
            meta = resolved_model_inst->meta;
            required_device_num = resolved_model_inst->model_info.get_device_num();
            mem_est = resolved_model_inst->model_info.get_estimated_mem();
            if (meta->status == SERVER_MODEL_STATUS_LOADED) {
                res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
    }

    const std::string routing_model_name = resolved_model_key.empty() ? model_name : resolved_model_key;

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

        const auto model_copy = *inst;
        return evict_model(model_copy, reason);
    };

    auto try_auto_evict_for_resource = [&]() {
        HLIEModelResidencyPolicy policy;
        const auto candidate = select_eviction_candidate(model_registry_, routing_model_name, mem_est, policy);
        return try_evict_candidate(candidate, policy, "resource_pressure");
    };

    {
        HLIEModelResidencyPolicy policy;
        const auto candidate = select_loaded_limit_candidate(model_registry_, routing_model_name, policy);
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
                if (device_service_->acquire_mem(mem_est, device, true, true, routing_model_name) < 0) {
                    lock.unlock();
                    const bool evicted = try_auto_evict_for_resource();
                    lock.lock();
                    if (!evicted || device_service_->acquire_mem(mem_est, device, true, true, routing_model_name) < 0) {
                        res->error(format_error_response_ex("The devices is not available, estimated model memory is " + std::to_string(mem_est) + "MB.",
                                                             ERROR_TYPE_UNAVAILABLE, 80004));
                        return res;
                    }
                }
                devices_vec.push_back(static_cast<int>(device));
            } else {
                if (device_service_->acquire_mem_on_device_count(required_device_num, mem_est, devices_vec, true, routing_model_name) < 0) {
                    lock.unlock();
                    const bool evicted = try_auto_evict_for_resource();
                    lock.lock();
                    if (!evicted || device_service_->acquire_mem_on_device_count(required_device_num, mem_est, devices_vec, true, routing_model_name) < 0) {
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
            if (device_service_->acquire_mem_on_devices(devices_vec, mem_est, false, true, routing_model_name) < 0) {
                lock.unlock();
                const bool evicted = try_auto_evict_for_resource();
                lock.lock();
                if (!evicted || device_service_->acquire_mem_on_devices(devices_vec, mem_est, false, true, routing_model_name) < 0) {
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
        if (!resolved_model_inst && path_llm_model_info) {
            const bool inserted = model_registry_.register_model_if_absent(*path_llm_model_info, mem_est);
            resolved_model_inst = model_registry_.get_exact_instance_snapshot(*path_llm_model_info);
            LOG_INFO("Service start resolved LLM model: key={}, registry_action={}, model_path={}, mmproj={}",
                     json(resolved_model_key).dump(), inserted ? "insert" : "reuse",
                     json(path_llm_model_info->get_model_path()).dump(),
                     json(path_llm_model_info->get_mm_proj_path()).dump());
        }

        // Refresh the exact instance after device selection. Another start request
        // may have registered or loaded the same variant while this request was resolving.
        resolved_model_inst = path_llm_model_info
            ? model_registry_.get_exact_instance_snapshot(*path_llm_model_info)
            : (resolved_model_key.empty() ? std::nullopt : get_model_inst(resolved_model_key));
        if (!resolved_model_inst) {
            res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (!validate_requested_llm_paths(*resolved_model_inst)) {
            return res;
        }

        const std::string model_key = resolved_model_inst->key;
        const bool owns_preclaim = !preclaimed_model_key.empty() && preclaimed_model_key == model_key;
        if (resolved_model_inst->meta.status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (resolved_model_inst->meta.status == SERVER_MODEL_STATUS_LOADING && !owns_preclaim) {
            res->error(format_error_response("model is already loading", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (resolved_model_inst->meta.status == SERVER_MODEL_STATUS_SLEEPING) {
            res->error(format_error_response("model is sleeping and cannot be started again", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (owns_preclaim && resolved_model_inst->meta.status != SERVER_MODEL_STATUS_LOADING) {
            res->error(format_error_response("preclaimed model is no longer loading", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!owns_preclaim && !model_registry_.try_begin_loading(model_key)) {
            const auto latest = get_model_inst(model_key);
            const std::string message = latest && latest->meta.status == SERVER_MODEL_STATUS_LOADED
                ? "model is already loaded"
                : "model is already loading";
            res->error(format_error_response(message, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto model_info_opt = model_registry_.prepare_model_info_with_devices(model_key, devices);
        if (!model_info_opt) {
            rollback_start(model_key, "prepare_model_info_failed");
            res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        auto model_info = std::move(*model_info_opt);
        if (!resolve_llm_launch_params(model_info)) {
            rollback_start(model_key, "resolve_llm_launch_params_failed");
            res->error(format_error_response_ex("failed to resolve llama params", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }
        if (model_info.model_version_.empty()) {
            rollback_start(model_key, "model_version_missing");
            res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
            rollback_start(model_key, "model_version_unsupported");
            res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        if (!model_service_manager_.llm()) {
            if (!model_service_manager_.start_llm(model_dir_, model_info, device_service_)) {
                LOG_ERROR("failed to init the LLM model service");
                res->error(format_error_response_ex("failed to init the LLM model service", ERROR_TYPE_SERVER, 80001));
                rollback_start(model_key, "llm_init_failed");
                return res;
            }
            if (DocumentPipelineService::supports_document_layout(model_dir_, model_info) &&
                !model_service_manager_.start_document_pipeline(model_dir_, model_info, device_service_)) {
                LOG_ERROR("failed to init the document pipeline service");
                res->error(format_error_response_ex("failed to init the document pipeline service", ERROR_TYPE_SERVER, 80001));
                model_service_manager_.stop_and_reset_llm();
                rollback_start(model_key, "document_pipeline_init_failed");
                return res;
            }

            mark_local_model_started(model_key, model_service_manager_.llm()->get_model_key(), true);
        } else {
            if (port_ != master_port_) {
                LOG_ERROR("slave cannot start new slave");
                res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
                rollback_start(model_key, "slave_cannot_start_new_slave");
                return res;
            }

            // Subprocess instance arguments.
            json_info_file = generate_launch_manifest(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                rollback_start(model_key, "generate_launch_manifest_failed");
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--launch_manifest"), json_info_file};
            const auto& args = config_services->get_args();

            // Copy llama-params
            for (const auto& [key, val] : args) {
                args_for_sub_process.push_back(key);
                args_for_sub_process.push_back(val);
            }

            process_id = Singleton<HLIEProcessManager>::instance()->create_process(
                program_path_, args_for_sub_process, {model_key}, instance_id);
            if (process_id == 0 || process_id == static_cast<ProcessID>(-1)) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
                res->error(format_error_response_ex("failed to start model subprocess", ERROR_TYPE_SERVER, 80001));
                rollback_start(model_key, "create_process_failed");
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
                rollback_start(HLIEModelType::SD3, "sd_init_failed");
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
            json_info_file = generate_launch_manifest(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--launch_manifest"), json_info_file};
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
    } else if (model_type == HLIEModelType::QWEN3_TTS) {
        if (start_qwen3_tts(meta, res, req, model_name, model_path, devices) < 0) {
            LOG_ERROR("failed to start the Qwen3-TTS model");
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

    preclaim_rollback.release();
    res->ok(resp);
    return res;
}

int HLIEWorkflow::start_whisper(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res,
    const server_http_req& req, const std::string& model_name, const std::string& model_path, const std::string& devices) {

    if (!meta) {
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);
        model_info_tmp.set_service_kind(HLIEModelType::WHISPER);
        const auto model_key = register_model_instance(model_info_tmp);
        LOG_INFO("Service Starting Add WHISPER Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    const std::string model_key = model_inst->key;
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
        std::string json_info_file = generate_launch_manifest(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            rollback_model_start(model_key, "generate_launch_manifest_failed");
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--launch_manifest"), json_info_file};
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
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);
        model_info_tmp.set_service_kind(HLIEModelType::COSYVOICE);
        const auto model_key = register_model_instance(model_info_tmp);
        LOG_INFO("Service Starting Add COSYVOICE Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    const std::string model_key = model_inst->key;
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
        std::string json_info_file = generate_launch_manifest(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            rollback_model_start(model_key, "generate_launch_manifest_failed");
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--launch_manifest"), json_info_file};
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

int HLIEWorkflow::start_qwen3_tts(std::optional<server_model_meta>& meta,
                                  std::unique_ptr<server_res_generator_light>& res,
                                  const server_http_req& req,
                                  const std::string& model_name,
                                  const std::string& model_path,
                                  const std::string& devices) {
    (void)req;
    if (!meta) {
        HLIEModelInfo model_info_tmp;
        if (model_info_tmp.parse_from_gguf_filename(model_path) < 0) {
            res->error(format_error_response_ex("invalid Qwen3-TTS model filename",
                                                ERROR_TYPE_INVALID_REQUEST, 80007));
            return -1;
        }

        model_info_tmp.set_service_kind(HLIEModelType::QWEN3_TTS);
        const auto model_key = register_model_instance(model_info_tmp);
        LOG_INFO("Service Starting Add Qwen3-TTS Model: {}", model_key);
    }

    const auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    const std::string model_key = model_inst->key;
    auto model_info_opt = model_registry_.prepare_model_info_with_devices(model_key, devices);
    if (!model_info_opt) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }
    auto model_info = std::move(*model_info_opt);
    if (model_info.model_version_.empty() ||
        !HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
        res->error(format_error_response_ex("build_version must be >= v1.0.0",
                                            ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    model_registry_.mark_loading(model_key);
    if (!model_service_manager_.qwen3_tts()) {
        if (!model_service_manager_.start_qwen3_tts(model_dir_, model_info, device_service_)) {
            LOG_ERROR("failed to init the Qwen3-TTS model service");
            res->error(format_error_response_ex("failed to init the Qwen3-TTS model service",
                                                ERROR_TYPE_SERVER, 80001));
            rollback_model_start(model_key, "qwen3_tts_init_failed");
            return -1;
        }

        mark_local_model_started(model_key,
                                 model_service_manager_.qwen3_tts()->get_model_key(), false);
        return 0;
    }

    if (port_ != master_port_) {
        LOG_ERROR("slave cannot start new slave");
        res->error(format_error_response_ex("slave cannot start new slave",
                                            ERROR_TYPE_SERVER, 80001));
        rollback_model_start(model_key, "slave_cannot_start_new_slave");
        return -1;
    }

    const std::string json_info_file = generate_launch_manifest(model_info, model_dir_, res);
    if (json_info_file.empty()) {
        rollback_model_start(model_key, "generate_launch_manifest_failed");
        return -1;
    }

    const auto instance_id = generate_instance_id();
    std::vector args_for_sub_process = {
        std::string("--instance_id"), std::to_string(instance_id),
        std::string("--device_id"), devices,
        std::string("--launch_manifest"), json_info_file,
    };
    const auto& args = Singleton<HLIEConfigService>::instance()->get_args();
    for (const auto& [key, val] : args) {
        args_for_sub_process.push_back(key);
        args_for_sub_process.push_back(val);
    }

    const auto pid = Singleton<HLIEProcessManager>::instance()->create_process(
        program_path_, args_for_sub_process, {model_name}, instance_id);
    if (pid == 0 || pid == -1) {
        LOG_ERROR("Failed to start Qwen3-TTS process for instance_id[{}]", instance_id);
        res->error(format_error_response_ex("failed to start model subprocess",
                                            ERROR_TYPE_SERVER, 80001));
        rollback_model_start(model_key, "create_process_failed");
        return -1;
    }

    LOG_INFO("Started Qwen3-TTS process PID[{}] instance_id[{}]", pid, instance_id);
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

    if (meta->meta.status != SERVER_MODEL_STATUS_LOADED) {
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
            req.should_stop,
            build_proxy_timeout_options());
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
    if (!req.body.empty()) {
        body = json::parse(req.body);
        if (name.empty()) {
            name = json_value(body, "model", std::string());
        }
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

    auto inst = get_model_inst(name);
    if (!inst.has_value()) {
        wait_for_model_discovery(request_model_discovery_scan());
        inst = get_model_inst(name);
        if (!inst.has_value()) {
            LOG_ERROR("LLM request model lookup failed after discovery: path={}, model={}, normalized_model={}",
                      req.path, json(name).dump(), json(normalize_model_name(name)).dump());
            res->error(format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    LOG_INFO("LLM Request route info, model [{}] status [{}] port [{}]" , name, static_cast<int>(inst->meta.status), inst->meta.port);
    if (inst->meta.status == SERVER_MODEL_STATUS_LOADED && (inst->meta.port == master_port || master_port != port_)) {
        const auto model_key = inst->key;
        ModelRequestLease lease(model_registry_, model_key);
        if (!lease.active()) {
            res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_local_llm_or_document(model_key, lease);
    }

    constexpr auto model_load_timeout = std::chrono::minutes(5);
    const auto model_load_deadline = std::chrono::steady_clock::now() + model_load_timeout;
    const bool auto_load = Singleton<HLIEConfigService>::instance()->get_bool("model", "auto_load", true, true);
    // If the model is unloaded, try to auto-load it.
    if (inst->meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        if (!auto_load) {
            res->error(format_error_response("model not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (method != "POST") {
            res->error(format_error_response(
                "model is not loaded; load it before using this GET endpoint", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // The registry grants exactly one request the right to start this model.
        // Loading runs in a managed background task so the initiating HTTP request
        // observes the same deadline and cancellation behavior as other waiters.
        if (model_registry_.try_begin_loading(inst->key)) {
            json start_body = {{"model", name}, {"model_type", LLM}};
            for (const auto* field : {"model_path", "devices", "params"}) {
                if (body.contains(field)) {
                    start_body[field] = body.at(field);
                }
            }

            const std::string model_key = inst->key;
            const std::string request_path = req.path;
            const std::string serialized_start_body = start_body.dump();
            const bool launched = auto_load_tasks_.launch(
                [this, model_key, request_path, serialized_start_body] {
                    const std::function<bool()> never_stop = [] { return false; };
                    const server_http_req start_req{
                        {}, {}, request_path, {}, serialized_start_body, {}, never_stop};
                    try {
                        const auto start_res = start_service(start_req, model_key);
                        if (start_res->status != 200) {
                            LOG_ERROR("Automatic model start failed: key={}, status={}, response={}",
                                      json(model_key).dump(), start_res->status, json(start_res->data).dump());
                        }
                    } catch (const std::exception& error) {
                        LOG_ERROR("Automatic model start threw: key={}, error={}",
                                  json(model_key).dump(), json(error.what()).dump());
                    } catch (...) {
                        LOG_ERROR("Automatic model start threw an unknown exception: key={}", json(model_key).dump());
                    }
                });
            if (!launched) {
                rollback_model_start(model_key, "auto_load_task_launch_failed");
                res->error(format_error_response_ex(
                    "failed to schedule model loading", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
        }
    }

    auto inst1 = get_model_inst(name);
    if (inst1 && inst1->meta.status == SERVER_MODEL_STATUS_LOADING) {
        std::unique_lock<std::mutex> lk(slave_mutex_);
        const auto wait_result = wait_for_model_load(slave_cv_, lk, [this, &name]() {
            const auto inst_inner = get_model_inst(name);
            return inst_inner ? std::optional(inst_inner->meta.status) : std::nullopt;
        }, model_load_deadline, req.should_stop);

        if (wait_result == ModelLoadWaitResult::TIMED_OUT) {
            LOG_WARN("Timed out waiting for model load: model={}, timeout_sec={}",
                     json(name).dump(),
                     std::chrono::duration_cast<std::chrono::seconds>(model_load_timeout).count());
            res->error(format_error_response_ex(
                "timed out waiting for model to load", ERROR_TYPE_UNAVAILABLE, 80004));
            return res;
        }
        if (wait_result == ModelLoadWaitResult::CANCELLED) {
            LOG_INFO("Stopped waiting for model load after client cancellation: model={}", json(name).dump());
            auto error = format_error_response("request cancelled while waiting for model to load",
                                               ERROR_TYPE_UNAVAILABLE);
            error["code"] = 499;
            res->error(error);
            return res;
        }
        if (wait_result == ModelLoadWaitResult::MISSING) {
            res->error(format_error_response("model not found while loading", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        inst1 = get_model_inst(name);
    }

    if (!inst1) {
        res->error(format_error_response("model not found after load", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    if (inst1->meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        // 加载失败
        res->error("failed to load model: " + name);
        return res;
    }

    // 如果已加载则直接调用
    if (inst1->meta.port == port_) {
        const auto model_key = inst1->key;
        ModelRequestLease lease(model_registry_, model_key);
        if (!lease.active()) {
            res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_local_llm_or_document(model_key, lease);
    }

    const auto model_key = inst1->key;
    ModelRequestLease lease(model_registry_, model_key);
    if (!lease.active()) {
        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    return attach_model_request_lease(proxy_request(req, method, name, inst1->meta.port), model_registry_, model_key, lease);
}

void HLIEWorkflow::handle_speech(const httplib::Request& req, httplib::Response& res) {
    const auto fail = [&](const std::string& message, const error_type type) {
        const auto error_data = format_error_response_ex(message, type, 80004);
        const json final_response{{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };
    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        fail("request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
        return;
    }

    if (body.contains("model") && !body.at("model").is_null() &&
        !body.at("model").is_string()) {
        fail("model must be a string", ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    const std::string requested_model =
        body.contains("model") && body.at("model").is_string()
            ? body.at("model").get<std::string>() : std::string();

    // A concrete model id can resolve to a child process even when a different
    // TTS model is loaded in the primary process. Keep the historical aliases
    // (qwen3_tts, cosyvoice and tts) local so their old default behavior does
    // not change.
    if (!requested_model.empty()) {
        auto instance = get_model_inst(requested_model);
        if (instance && instance->meta.status == SERVER_MODEL_STATUS_LOADING) {
            constexpr auto model_load_timeout = std::chrono::minutes(5);
            const auto model_load_deadline =
                std::chrono::steady_clock::now() + model_load_timeout;
            std::unique_lock<std::mutex> lock(slave_mutex_);
            const auto wait_result = wait_for_model_load(
                slave_cv_, lock,
                [this, &requested_model]() {
                    const auto current = get_model_inst(requested_model);
                    return current
                        ? std::optional(current->meta.status) : std::nullopt;
                },
                model_load_deadline, req.is_connection_closed);

            if (wait_result == ModelLoadWaitResult::TIMED_OUT) {
                fail("timed out waiting for speech model to load",
                     ERROR_TYPE_UNAVAILABLE);
                return;
            }
            if (wait_result == ModelLoadWaitResult::CANCELLED) {
                auto error_data = format_error_response(
                    "request cancelled while waiting for speech model to load",
                    ERROR_TYPE_UNAVAILABLE);
                error_data["code"] = 499;
                res.set_content(
                    safe_json_to_str(json{{"error", error_data}}), MIMETYPE_JSON);
                res.status = 499;
                return;
            }
            if (wait_result == ModelLoadWaitResult::MISSING) {
                fail("speech model not found while loading", ERROR_TYPE_NOT_FOUND);
                return;
            }
            instance = get_model_inst(requested_model);
        }
        if (instance) {
            const auto service_kind = HLIEModelServiceFactory::service_kind(instance->model_info);
            const bool speech_instance =
                service_kind == ModelServiceKind::Qwen3TTS ||
                service_kind == ModelServiceKind::CosyVoice;
            if (speech_instance && instance->meta.port != port_) {
                if (instance->meta.status != SERVER_MODEL_STATUS_LOADED ||
                    instance->meta.port <= 0) {
                    fail("requested speech model is not loaded", ERROR_TYPE_UNAVAILABLE);
                    return;
                }

                auto lease = std::make_shared<ModelRequestLease>(model_registry_, instance->key);
                if (!lease->active()) {
                    fail("model is not loaded", ERROR_TYPE_INVALID_REQUEST);
                    return;
                }

                std::map<std::string, std::string> headers;
                for (const auto& [key, value] : req.headers) {
                    headers[key] = value;
                }

                std::string proxy_path = req.path;
                if (!req.params.empty()) {
                    proxy_path += "?";
                    bool first = true;
                    for (const auto& [key, value] : req.params) {
                        if (!first) {
                            proxy_path += "&";
                        }
                        first = false;
                        proxy_path += httplib::encode_query_component(key) + "=" +
                                      httplib::encode_query_component(value);
                    }
                }

                LOG_INFO("Proxying speech request for model {} to port {} (stream={})",
                         requested_model, instance->meta.port,
                         json_boolish(body, "stream"));

                std::shared_ptr<ServerHttpProxy> proxy;
                try {
                    proxy = std::make_shared<ServerHttpProxy>(
                        "POST", "127.0.0.1", instance->meta.port, proxy_path,
                        headers, req.body, std::map<std::string, uploaded_file>{},
                        req.is_connection_closed, build_proxy_timeout_options());
                } catch (const std::exception& error) {
                    LOG_ERROR("Failed to create speech proxy for model {} on port {}: {}",
                              requested_model, instance->meta.port, error.what());
                    fail("failed to connect to speech model instance", ERROR_TYPE_UNAVAILABLE);
                    return;
                }

                res.status = proxy->status;
                for (const auto& [key, value] : proxy->headers) {
                    res.set_header(key, value);
                }

                if (!json_boolish(body, "stream")) {
                    std::string complete_body;
                    std::string chunk;
                    while (proxy->next(chunk)) {
                        complete_body += chunk;
                        chunk.clear();
                    }
                    proxy->cancel();
                    res.set_content(std::move(complete_body), proxy->content_type);
                    return;
                }

                res.set_chunked_content_provider(
                    proxy->content_type,
                    [proxy, lease](size_t, httplib::DataSink& sink) {
                        (void)lease;
                        if (!sink.is_writable()) {
                            proxy->cancel();
                            return false;
                        }

                        std::string chunk;
                        const bool has_next = proxy->next(chunk);
                        if (!chunk.empty() &&
                            !sink.write(chunk.data(), chunk.size())) {
                            proxy->cancel();
                            return false;
                        }

                        if (!has_next) {
                            if (!sink.is_writable()) {
                                proxy->cancel();
                                return false;
                            }
                            const auto upstream_trailers = proxy->response_trailers();
                            if (upstream_trailers.empty()) {
                                sink.done();
                            } else {
                                httplib::Headers trailers;
                                for (const auto& [key, value] : upstream_trailers) {
                                    trailers.emplace(key, value);
                                }
                                sink.done_with_trailer(trailers);
                            }
                            proxy->cancel();
                        }
                        return true;
                    },
                    [proxy, lease](bool) {
                        (void)lease;
                        proxy->cancel();
                    });
                return;
            }
        }
    }

    enum class SpeechBackend { NONE, COSYVOICE, QWEN3_TTS };
    SpeechBackend backend = SpeechBackend::NONE;
    std::shared_ptr<Qwen3TTSModelService> qwen_service;
    std::shared_ptr<CosyVoiceModelService> cosy_service;
    std::shared_ptr<ModelRequestLease> lease;
    {
        std::lock_guard workflow_lock(mutex_);
        qwen_service = model_service_manager_.qwen3_tts();
        cosy_service = model_service_manager_.cosyvoice();
        const bool cosy_ready = cosy_service &&
            cosy_service->get_state() == SERVER_STATE_READY;
        const bool qwen_ready = qwen_service &&
            qwen_service->get_state() == SERVER_STATE_READY;

        if (!requested_model.empty()) {
            std::string requested_type;
            std::string requested_key;
            if (const auto instance = get_model_inst(requested_model)) {
                requested_type = normalize_model_name(
                    instance->model_info.get_service_kind());
                requested_key = normalize_model_name(instance->key);
            }
            const std::string normalized = normalize_model_name(requested_model);
            const std::string loaded_cosy_key = cosy_service
                ? normalize_model_name(cosy_service->get_model_key())
                : std::string();
            const std::string loaded_qwen_key = qwen_service
                ? normalize_model_name(qwen_service->get_model_key())
                : std::string();
            const bool cosy_alias =
                normalized == HLIEModelType::COSYVOICE ||
                normalized == HLIEModelType::TTS;
            const bool qwen_alias =
                normalized == HLIEModelType::QWEN3_TTS ||
                normalized == "qwen3-tts";
            const bool cosy_model_match = !loaded_cosy_key.empty() &&
                (normalized == loaded_cosy_key ||
                 requested_key == loaded_cosy_key);
            const bool qwen_model_match = !loaded_qwen_key.empty() &&
                (normalized == loaded_qwen_key ||
                 requested_key == loaded_qwen_key);
            if (cosy_alias || cosy_model_match) {
                backend = SpeechBackend::COSYVOICE;
            } else if (qwen_alias || qwen_model_match) {
                backend = SpeechBackend::QWEN3_TTS;
            } else {
                LOG_WARN(
                    "Speech request rejected because requested model is not the "
                    "loaded speech instance: requested={}, requested_type={}, "
                    "loaded_cosy={}, loaded_qwen={}",
                    requested_model, requested_type, loaded_cosy_key,
                    loaded_qwen_key);
                fail("requested speech model is not loaded",
                     ERROR_TYPE_INVALID_REQUEST);
                return;
            }
        } else if (cosy_ready) {
            // Backward compatibility: /v1/audio/speech without a model has
            // always meant CosyVoice.
            backend = SpeechBackend::COSYVOICE;
        } else if (qwen_ready) {
            fail("model is required when using Qwen3-TTS",
                 ERROR_TYPE_INVALID_REQUEST);
            return;
        }

        if (backend == SpeechBackend::QWEN3_TTS && !qwen_ready) {
            fail("Qwen3-TTS model is not available", ERROR_TYPE_UNAVAILABLE);
            return;
        }
        if (backend == SpeechBackend::COSYVOICE && !cosy_ready) {
            fail("CosyVoice model is not available", ERROR_TYPE_UNAVAILABLE);
            return;
        }
        if (backend == SpeechBackend::NONE) {
            fail("No speech model is available", ERROR_TYPE_UNAVAILABLE);
            return;
        }
        const std::string model_key = backend == SpeechBackend::QWEN3_TTS
            ? qwen_service->get_model_key() : cosy_service->get_model_key();
        lease = std::make_shared<ModelRequestLease>(model_registry_, model_key);
        if (!lease->active()) {
            fail("model is not loaded", ERROR_TYPE_INVALID_REQUEST);
            return;
        }
    }

    const char* backend_name = backend == SpeechBackend::QWEN3_TTS
        ? HLIEModelType::QWEN3_TTS : HLIEModelType::COSYVOICE;
    res.set_header("X-HLIE-TTS-Backend", backend_name);
    LOG_INFO("Speech request routed to {} (requested_model={})",
             backend_name,
             requested_model.empty() ? "<legacy-default>" : requested_model);
    if (backend == SpeechBackend::QWEN3_TTS) {
        qwen_service->handle_speech(req, res, lease);
    } else {
        cosy_service->handle_speech(req, res);
    }
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
    auto route_qwen3_tts = [this](const server_http_req& req, auto handler) -> server_http_res_ptr {
        std::shared_ptr<Qwen3TTSModelService> service;
        std::unique_ptr<ModelRequestLease> lease;
        std::string model_key;
        {
            std::lock_guard workflow_lock(mutex_);
            service = model_service_manager_.qwen3_tts();
            if (!service || service->get_state() != SERVER_STATE_READY) {
                auto res = std::make_unique<server_res_generator_light>();
                res->error(format_error_response("Model is not available for now.",
                                                 ERROR_TYPE_UNAVAILABLE));
                return res;
            }
            model_key = service->get_model_key();
            lease = std::make_unique<ModelRequestLease>(model_registry_, model_key);
            if (!lease->active()) {
                auto res = std::make_unique<server_res_generator_light>();
                res->error(format_error_response("model is not loaded",
                                                 ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        return attach_model_request_lease(
            (service.get()->*handler)(req), model_registry_, model_key, *lease);
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
                {"/hlie/v1/models/refresh", "/models/refresh"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return handle_model_discovery_refresh(req);
                },
                false
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
                false,
                HLIEModelType::TTS,
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_speech(req, res);
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
                {"/hlie/v1/qwen3_tts/prefill_debug"},
                POST,
                [route_qwen3_tts](const server_http_req& req) -> server_http_res_ptr {
                    return route_qwen3_tts(req, &Qwen3TTSModelService::handle_prefill_debug);
                },
                true,
                HLIEModelType::QWEN3_TTS
            },
            {
                {"/hlie/v1/qwen3_tts/codec_frames_debug"},
                POST,
                [route_qwen3_tts](const server_http_req& req) -> server_http_res_ptr {
                    return route_qwen3_tts(req, &Qwen3TTSModelService::handle_codec_frames_debug);
                },
                true,
                HLIEModelType::QWEN3_TTS
            },
            {
                {"/hlie/v1/qwen3_tts/reference_debug"},
                POST,
                [route_qwen3_tts](const server_http_req& req) -> server_http_res_ptr {
                    return route_qwen3_tts(req, &Qwen3TTSModelService::handle_reference_debug);
                },
                true,
                HLIEModelType::QWEN3_TTS
            },
            {
                {"/hlie/v1/qwen3_tts/speech_debug"},
                POST,
                [route_qwen3_tts](const server_http_req& req) -> server_http_res_ptr {
                    return route_qwen3_tts(req, &Qwen3TTSModelService::handle_speech_debug);
                },
                true,
                HLIEModelType::QWEN3_TTS
            },
            {
                {"/hlie/v1/qwen3_tts/stateful_decoder_debug"},
                POST,
                [route_qwen3_tts](const server_http_req& req) -> server_http_res_ptr {
                    return route_qwen3_tts(req, &Qwen3TTSModelService::handle_stateful_decoder_debug);
                },
                true,
                HLIEModelType::QWEN3_TTS
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

    if (model_service_manager_.qwen3_tts()) {
        model_registry_.update_port(model_service_manager_.qwen3_tts()->get_model_key(), port);
    }
}

server_http_res_ptr HLIEWorkflow::stop_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
    if (HLIEReplicaConfig::load().enabled()) {
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
        std::shared_ptr<Qwen3TTSModelService> qwen3_tts_to_stop;
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

            if (model_service_manager_.qwen3_tts()) {
                const auto model_key = model_service_manager_.qwen3_tts()->get_model_key();
                if (model_registry_.try_begin_eviction(model_key)) {
                    eviction_keys.push_back(model_key);
                } else {
                    LOG_WARN("Model stop_service skipped busy Qwen3-TTS model: key={}",
                             json(model_key).dump());
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
                if (model_service_manager_.qwen3_tts()) {
                    qwen3_tts_to_stop = model_service_manager_.detach_qwen3_tts();
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
        stop_outside_workflow_lock(qwen3_tts_to_stop, HLIEModelType::QWEN3_TTS);
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
        if (qwen3_tts_to_stop) {
            model_registry_.mark_unloaded(qwen3_tts_to_stop->get_model_key());
        }

        LOG_INFO("Succeed to stop service process for instance_id[{}][{}]", instance_id, process_id);
    } else if (process_id != 0) {
        const auto& process_manager = Singleton<HLIEProcessManager>::instance();
        const auto& models = process_manager->get_working_models(instance_id);

        for (const auto& model : models) {
            auto model_inst = get_model_inst(model);
            if (model_inst) {
                model_registry_.mark_unloaded(model_inst->key);
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
            const std::string public_name = model_inst ? model_inst->model_info.get_model_name() : name;
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
            model_inst ? model_inst->model_info.get_model_name() : model_service_manager_.llm()->get_model_key();
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

    if (model_service_manager_.qwen3_tts()) {
        const auto model_inst = get_model_inst(model_service_manager_.qwen3_tts()->get_model_key());
        const std::string model_name = model_inst
                                           ? model_inst->model_info.get_model_name()
                                           : model_service_manager_.qwen3_tts()->get_model_key();
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
        const auto resolved_key = inst->key;
        const auto resolved_model_name = inst->model_info.get_model_name();
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
    if (HLIEReplicaConfig::load().enabled()) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);

    // Prefer an explicit model identity when unloading.  Model type is only a
    // legacy selector and is ambiguous once multiple service kinds can host
    // audio models (for example qwen3-asr is served by the LLM backend).
    std::string requested_model = json_string_value(body, "model_id");
    if (requested_model.empty()) {
        requested_model = json_string_value(body, "model");
    }
    if (!requested_model.empty()) {
        const auto model_inst = get_model_inst(requested_model);
        if (!model_inst) {
            res->error(format_error_response("model not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        const std::string model_key = model_inst->key;
        if (!evict_model(*model_inst, "api_unload")) {
            LOG_WARN("Model unload refused or not matched: requested={}, key={}",
                     json(requested_model).dump(), json(model_key).dump());
            res->error(format_error_response_ex("model is busy or not loaded",
                                                 ERROR_TYPE_UNAVAILABLE, 80004));
            return res;
        }

        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, model_key, port_, SERVER_MODEL_STATUS_UNLOADED);
        }

        LOG_INFO("Succeed to uninstall model by identity: requested={}, key={}",
                 json(requested_model).dump(), json(model_key).dump());
        res->ok({{"code", 0}, {"model", model_key}});
        return res;
    }

    std::string model_type = body.value("model_type", LLM);
    // The document pipeline is owned by the LLM model lifecycle; there is no
    // independently unloadable document service instance.
    if (model_type == HLIEModelType::DOCUMENT) {
        model_type = LLM;
    }
    if (model_type == HLIEModelType::TTS) {
        model_type = HLIEModelType::COSYVOICE;
    }
    if (!HLIEModelServiceFactory::is_supported_model_type(model_type)) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    std::shared_ptr<LLModelService> llm_to_stop;
    std::shared_ptr<DocumentPipelineService> document_pipeline_to_stop;
    std::shared_ptr<SDModelService> sd_to_stop;
    std::shared_ptr<WhisperModelService> whisper_to_stop;
    std::shared_ptr<CosyVoiceModelService> cosy_voice_to_stop;
    std::shared_ptr<Qwen3TTSModelService> qwen3_tts_to_stop;
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
        } else if (model_type == HLIEModelType::QWEN3_TTS && model_service_manager_.qwen3_tts()) {
            notify_model_key = model_service_manager_.qwen3_tts()->get_model_key();
            if (model_registry_.try_begin_eviction(notify_model_key)) {
                qwen3_tts_to_stop = model_service_manager_.detach_qwen3_tts();
            } else {
                unload_refused = true;
            }
        }
    }

    if (notify_model_key.empty() && !unload_refused) {
        LOG_WARN("Model unload requested for a type with no loaded service: type={}", model_type);
        res->error(format_error_response_ex("model is busy or not loaded",
                                             ERROR_TYPE_UNAVAILABLE, 80004));
        return res;
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
    stop_outside_workflow_lock(qwen3_tts_to_stop, HLIEModelType::QWEN3_TTS);
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

    bool evicted = false;
    std::string local_service_type;
    std::shared_ptr<LLModelService> llm_to_stop;
    std::shared_ptr<DocumentPipelineService> document_pipeline_to_stop;
    std::shared_ptr<SDModelService> sd_to_stop;
    std::shared_ptr<WhisperModelService> whisper_to_stop;
    std::shared_ptr<CosyVoiceModelService> cosy_voice_to_stop;
    std::shared_ptr<Qwen3TTSModelService> qwen3_tts_to_stop;
    {
        std::unique_lock lock(mutex_);
        // Keep the lock order consistent with the explicit unload and
        // stop_service paths: workflow mutex before registry mutex.
        if (!model_registry_.try_begin_eviction(model_inst.key)) {
            return false;
        }

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
        } else if (model_service_manager_.qwen3_tts() && matches(model_service_manager_.qwen3_tts()->get_model_key())) {
            qwen3_tts_to_stop = model_service_manager_.detach_qwen3_tts();
            local_service_type = HLIEModelType::QWEN3_TTS;
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
    stop_evicted_service(qwen3_tts_to_stop);

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

bool HLIEWorkflow::initialize_llm_preset_catalogs() {
    const auto& config = Singleton<HLIEConfigService>::instance();
    llm_presets_enabled_ = config->get_bool("model", "llm_presets_enabled", true, true);
    llm_preset_strict_ = config->get_bool("model", "llm_preset_strict", true, true);
    if (!llm_presets_enabled_) {
        LOG_INFO("LLM preset catalogs are disabled");
        return true;
    }

    const auto bundle = config->get_config_bundle();
    const auto resolve_catalog_path = [&](const std::string& configured_path, const std::filesystem::path& bundle_path) {
        if (!bundle_path.empty()) {
            return bundle_path.string();
        }
        std::filesystem::path path(configured_path);
        if (path.is_relative()) {
            const std::filesystem::path config_path(config->get_config_path());
            path = config_path.parent_path() / path;
        }
        return path.lexically_normal().string();
    };
    const auto load_catalog = [&](HLIELlmPresetCatalog& catalog, const std::string& path,
                                  const std::string& source, const bool required) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) {
            if (required) {
                LOG_WARN("LLM {} catalog not found, continuing without it: {}", source, path);
            } else {
                LOG_INFO("LLM {} catalog not found, continuing without local overrides: {}", source, path);
            }
            return true;
        }

        std::string error;
        if (catalog.load_file(path, source, &error)) {
            LOG_INFO("Loaded LLM {} catalog: path={}, version={}, rules={}",
                     source, path, catalog.catalog_version(), catalog.rules().size());
            return true;
        }

        if (llm_preset_strict_) {
            LOG_ERROR("Failed to load LLM {} catalog in strict mode: path={}, error={}", source, path, error);
            return false;
        }
        LOG_WARN("Ignoring invalid LLM {} catalog: path={}, error={}", source, path, error);
        return true;
    };

    const std::string preset_path = resolve_catalog_path(config->get_string(
        "model", "llm_preset_catalog", "model_presets.json", true), bundle.model_presets);
    const std::string override_path = resolve_catalog_path(config->get_string(
        "model", "llm_override_catalog", "model_overrides.json", true), bundle.model_overrides);
    return load_catalog(llm_preset_catalog_, preset_path, "official", true) &&
           load_catalog(llm_override_catalog_, override_path, "local", false);
}

bool HLIEWorkflow::resolve_llm_launch_params(HLIEModelInfo& model_info) {
    if (HLIEModelServiceFactory::service_kind(model_info) != ModelServiceKind::LLM) {
        return true;
    }
    if (model_info.has_effective_llama_params()) {
        return true;
    }

    HLIELlmParamSet base;
    base.set("--parallel", static_cast<int64_t>(model_info.get_batch_num()));
    base.set("--ctx-size", static_cast<int64_t>(model_info.get_context_size()));
    base.set("--no-context-shift", true);
    base.set("--no-warmup", true);

    const std::vector<const HLIELlmPresetCatalog*> catalogs = llm_presets_enabled_
        ? std::vector<const HLIELlmPresetCatalog*>{&llm_preset_catalog_, &llm_override_catalog_}
        : std::vector<const HLIELlmPresetCatalog*>{};
    auto resolution = HLIELlmPresetResolver::resolve(model_info, base, catalogs);

    const auto& config = Singleton<HLIEConfigService>::instance();
    HLIELlmParamSet cli_layer;
    for (const auto& [key, value] : config->get_args()) {
        if (key.rfind("--", 0) != 0 || HLIELlmParamSet::is_framework_owned(key) ||
            key == "--device_id" || key == "--instance_id" || key == "--json_info" ||
            key == "--launch_manifest" || key == "--port") {
            continue;
        }
        if (value.empty()) {
            cli_layer.set(key, true);
        } else {
            cli_layer.set(key, value);
        }
    }
    if (!cli_layer.empty()) {
        resolution.params.merge_layer(cli_layer);
        resolution.sources.push_back("cli");
    }

    HLIELlmParamSet sys_layer;
    if (const auto reader = config->get_reader()) {
        for (const auto& key : reader->Keys("llm_params")) {
            if (HLIELlmParamSet::is_framework_owned(key)) {
                LOG_WARN("Ignoring framework-owned [llm_params] option: {}", key);
                continue;
            }
            const std::string value = reader->Get("llm_params", key, "");
            if (value.empty()) {
                sys_layer.set(key, true);
            } else {
                sys_layer.set(key, value);
            }
        }
    }
    if (!sys_layer.empty()) {
        resolution.params.merge_layer(sys_layer);
        resolution.sources.push_back("sys_cfg");
    }
    resolution.params_hash = resolution.params.stable_hash();

    model_info.set_effective_llama_params(std::move(resolution.params),
                                          std::move(resolution.matched_presets),
                                          std::move(resolution.sources),
                                          std::move(resolution.params_hash));
    if (!model_info.model_key_.empty()) {
        model_registry_.update_model_info(model_info.model_key_, model_info);
    }
    LOG_INFO("Resolved llama params: model_key={}, role={}, presets={}, hash={}",
             model_info.model_key_, model_info.get_role_name(),
             model_info.get_matched_llama_presets().size(), model_info.get_effective_llama_params_hash());
    return true;
}

void HLIEWorkflow::initialize_model_discovery_snapshot() {
    if (is_slave_) {
        return;
    }

    std::lock_guard lk(model_discovery_mutex_);
    if (const auto snapshot = build_model_directory_snapshot(model_dir_)) {
        model_discovery_snapshot_ = *snapshot;
        model_discovery_pending_ = false;
        LOG_INFO("Model discovery baseline initialized: directory={}", model_dir_);
    }
}

int HLIEWorkflow::discover_new_models(const bool force) {
    if (is_slave_ || model_dir_.empty()) {
        return 0;
    }

    std::lock_guard lk(model_discovery_mutex_);
    const auto observed_snapshot = build_model_directory_snapshot(model_dir_);
    if (!observed_snapshot) {
        return 0;
    }

    const auto& config = Singleton<HLIEConfigService>::instance();
    const int stability_sec = std::clamp(
        config->get_int("model", "discovery_stability_sec", 10, true), 0, 3600);
    const auto now = std::chrono::steady_clock::now();

    if (!force) {
        if (*observed_snapshot != model_discovery_snapshot_) {
            model_discovery_snapshot_ = *observed_snapshot;
            model_discovery_pending_ = true;
            model_discovery_pending_since_ = now;
            LOG_INFO("Model directory change detected; waiting for stable files: directory={}, stability_sec={}",
                     model_dir_, stability_sec);
            if (stability_sec > 0) {
                return 0;
            }
        } else if (!model_discovery_pending_) {
            return 0;
        } else if (std::chrono::duration_cast<std::chrono::seconds>(now - model_discovery_pending_since_).count() <
                   stability_sec) {
            return 0;
        }
    }

    std::vector<HLIEModelInfo> model_infos = load_sd_model_manifests_from_directory(model_dir_);
    HLIEModelInfo::ScanLocalModelsFromDir(model_infos, model_dir_);

    // Do not publish anything if a downloader changed the directory while
    // GGUF metadata was being inspected. The next stable pass will retry.
    const auto verified_snapshot = build_model_directory_snapshot(model_dir_);
    if (!verified_snapshot || *verified_snapshot != *observed_snapshot) {
        if (verified_snapshot) {
            model_discovery_snapshot_ = *verified_snapshot;
        }
        model_discovery_pending_ = true;
        model_discovery_pending_since_ = std::chrono::steady_clock::now();
        LOG_INFO("Model directory changed during discovery; deferring registration: directory={}", model_dir_);
        return 0;
    }

    int discovered_count = 0;
    for (auto& model_info : model_infos) {
        if (!is_discovered_model_complete(model_dir_, model_info)) {
            continue;
        }
        if (model_registry_.register_model_if_absent(model_info, model_info.get_estimated_mem())) {
            ++discovered_count;
        }
    }

    model_discovery_snapshot_ = *verified_snapshot;
    model_discovery_pending_ = false;
    if (discovered_count > 0) {
        LOG_INFO("Model discovery registered {} new model instance(s) from {}", discovered_count, model_dir_);
    } else {
        LOG_INFO("Model discovery completed with no new model instances: directory={}", model_dir_);
    }
    return discovered_count;
}

uint64_t HLIEWorkflow::request_model_discovery_scan(const bool force) {
    if (is_slave_ || (!force &&
        !Singleton<HLIEConfigService>::instance()->get_bool(
            "model", "discovery_scan_on_models_request", false, true))) {
        return 0;
    }

    const int cooldown_sec = std::clamp(
        Singleton<HLIEConfigService>::instance()->get_int(
            "model", "discovery_request_cooldown_sec", 5, true),
        0, 3600);
    uint64_t generation = 0;
    {
        std::lock_guard wait_lock(model_discovery_wait_mutex_);
        if (!model_discovery_thread_ || model_discovery_stop_.load()) {
            return 0;
        }
        if (model_discovery_requested_.load()) {
            return model_discovery_sync_.latest_requested();
        }
        const auto now = std::chrono::steady_clock::now();
        if (cooldown_sec > 0 && model_discovery_last_request_.time_since_epoch().count() > 0 &&
            std::chrono::duration_cast<std::chrono::seconds>(now - model_discovery_last_request_).count() < cooldown_sec) {
            return 0;
        }

        generation = model_discovery_sync_.request();
        model_discovery_last_request_ = now;
        model_discovery_requested_.store(true);
    }
    model_discovery_cv_.notify_all();
    return generation;
}

bool HLIEWorkflow::wait_for_model_discovery(const uint64_t generation) {
    if (generation == 0) {
        return false;
    }

    const int wait_sec = std::clamp(
        Singleton<HLIEConfigService>::instance()->get_int(
            "model", "discovery_request_wait_sec", 15, true),
        0, 120);
    if (wait_sec <= 0) {
        return false;
    }

    const bool completed = model_discovery_sync_.wait_for(generation, std::chrono::seconds(wait_sec));
    if (!completed) {
        LOG_WARN("Timed out waiting for model discovery: generation={}, timeout_sec={}", generation, wait_sec);
    }
    return completed;
}

void HLIEWorkflow::start_model_discovery_worker() {
    if (is_slave_ || model_discovery_thread_) {
        return;
    }

    const int interval_sec = Singleton<HLIEConfigService>::instance()->get_int(
        "model", "discovery_scan_interval_sec", 30, true);
    const bool scan_on_request = Singleton<HLIEConfigService>::instance()->get_bool(
        "model", "discovery_scan_on_models_request", false, true);
    if (interval_sec <= 0 && !scan_on_request) {
        LOG_INFO("Model discovery worker is disabled");
        return;
    }

    {
        std::lock_guard wait_lock(model_discovery_wait_mutex_);
        model_discovery_stop_.store(false);
        model_discovery_requested_.store(false);
        model_discovery_last_request_ = {};
    }
    model_discovery_sync_.reset();
    model_discovery_thread_ = std::make_unique<std::thread>(&HLIEWorkflow::model_discovery_loop, this);
    LOG_INFO("Model discovery worker started: directory={}, interval_sec={}, scan_on_request={}",
             model_dir_, interval_sec > 0 ? std::clamp(interval_sec, 1, 3600) : 0, scan_on_request);
}

void HLIEWorkflow::stop_model_discovery_worker() {
    {
        std::lock_guard wait_lock(model_discovery_wait_mutex_);
        model_discovery_stop_.store(true);
    }
    model_discovery_sync_.stop();
    model_discovery_cv_.notify_all();
    if (!model_discovery_thread_) {
        return;
    }

    if (model_discovery_thread_->joinable()) {
        model_discovery_thread_->join();
    }
    model_discovery_thread_.reset();
    LOG_INFO("Periodic model discovery worker stopped");
}

void HLIEWorkflow::model_discovery_loop() {
    while (!model_discovery_stop_.load()) {
        const int configured_interval = Singleton<HLIEConfigService>::instance()->get_int(
            "model", "discovery_scan_interval_sec", 30, true);
        int wait_sec = configured_interval > 0 ? std::clamp(configured_interval, 1, 3600) : 3600;
        bool discovery_pending = false;
        {
            std::lock_guard discovery_lock(model_discovery_mutex_);
            if (model_discovery_pending_) {
                discovery_pending = true;
                const int stability_sec = std::clamp(
                    Singleton<HLIEConfigService>::instance()->get_int(
                        "model", "discovery_stability_sec", 10, true),
                    0, 3600);
                const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - model_discovery_pending_since_).count();
                const int remaining_sec = std::max(1, stability_sec - static_cast<int>(elapsed_sec));
                wait_sec = std::min(wait_sec, remaining_sec);
            }
        }

        std::unique_lock wait_lock(model_discovery_wait_mutex_);
        model_discovery_cv_.wait_for(wait_lock, std::chrono::seconds(wait_sec), [this] {
            return model_discovery_stop_.load() || model_discovery_requested_.load();
        });
        if (model_discovery_stop_.load()) {
            break;
        }
        const bool requested = model_discovery_requested_.exchange(false);
        wait_lock.unlock();

        if (configured_interval > 0 || requested || discovery_pending) {
            try {
                discover_new_models();
            } catch (const std::exception& ex) {
                LOG_ERROR("Model discovery failed: {}", ex.what());
            } catch (...) {
                LOG_ERROR("Model discovery failed with an unknown exception");
            }
            bool pending = false;
            {
                std::lock_guard discovery_lock(model_discovery_mutex_);
                pending = model_discovery_pending_;
            }
            if (!pending) {
                // Read the latest generation after the scan so requests that
                // arrive while a scan is in flight are coalesced into it.
                model_discovery_sync_.complete_through(model_discovery_sync_.latest_requested());
            }
        }
    }
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

        const auto model_copy = *inst;
        evict_model(model_copy, "idle_ttl");
    }
}

std::vector<server_model_meta> HLIEWorkflow::get_all_model_metas() {
    return model_registry_.get_all_metas();
}

std::string HLIEWorkflow::register_model_instance(HLIEModelInfo& model_info) {
    model_registry_.register_model(model_info, model_info.get_estimated_mem());
    return model_info.model_key_;
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

std::optional<HLIEWorkflow::ModelInstance> HLIEWorkflow::get_model_inst(const std::string& model_name) {
    return model_registry_.get_instance_snapshot(model_name);
}

HLIE_NAMESPACE_END
