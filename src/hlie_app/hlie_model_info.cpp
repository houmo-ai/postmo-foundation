#include "hlie_app/hlie_model_info.h"

#include <fstream>

#include "hlie_base/hlie_logger.h"
#include "json.hpp"
#include "log.h"


HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
        return result;
    }
} // namespace

int HLIEModelInfo::parse_from_file(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Error: Could not open file: {}", file_path);
            return -3;
        }

        nlohmann::json config;
        file >> config;

        model_path_ = config.value("ModelPath", "");
        model_type_ = config.value("ModelType", "");
        model_type_ = to_lower(model_type_);
        context_size_ = config.value("ContextSize", 0);
        cores_ = config.value("Cores", 0);
        batch_num_ = config.value("BatchNum", 1);
        xh_version_ = config.value("XHVersion", 1);
        is_multi_model_ = config.value("IsMM", false);
        params_ = config.value("Params", "");
        mm_proj_path_ = config.value("MMProj", "");
        model_version_ = config.value("Version", "");
        clip_model_ = config.value("ClipModel", "");
        clip_l_model_ = config.value("ClipLModel", "");
        t5_model_ = config.value("T5Model", "");
        devices_ = config.value("Devices", "");
        mmdit_model_ = config.value("MMDitModel", "");
        vae_model_ = config.value("VaeModel", "");
    } catch (const nlohmann::json::exception& ex) {
        LOG_ERROR("JSON Error: {}", ex.what());
        return -2;
    } catch (const std::exception& ex) {
        LOG_ERROR("General Error: {}", ex.what());
        return -1;
    }

    return 0;
}

void HLIEModelInfo::build_llm(const std::string& model_path, const std::string& model_type, const int context_size, const int cores,
                              const int batch_num, const int xh_version, const bool is_multi_model, const std::string& params,
                              const std::string& mm_proj_path, const std::string& devices) {
    model_path_ = model_path;
    model_type_ = to_lower(model_type);
    context_size_ = context_size;
    cores_ = cores;
    batch_num_ = batch_num;
    xh_version_ = xh_version;
    is_multi_model_ = is_multi_model;
    params_ = params;
    mm_proj_path_ = mm_proj_path;
    devices_ = devices;
}

void HLIEModelInfo::build_sd(const std::string& model_type, const std::string& vae_model, const std::string& t5_model,
                             const std::string& clip_model, const std::string& clip_l_model, const std::string& mmdit_model,
                             const int xh_version, const std::string& devices) {
    model_type_ = to_lower(model_type);
    vae_model_ = vae_model;
    t5_model_ = t5_model;
    clip_model_ = clip_model;
    clip_l_model_ = clip_l_model;
    mmdit_model_ = mmdit_model;
    xh_version_ = xh_version;
    devices_ = devices;
}

int HLIEModelInfo::to_json_file(const std::string& file_path) const {
    const nlohmann::json config = {
        {"ModelPath", model_path_},
        {"ModelType", model_type_},
        {"ContextSize", context_size_},
        {"Cores", cores_},
        {"BatchNum", batch_num_},
        {"XHVersion", xh_version_},
        {"IsMM", is_multi_model_},
        {"Params", params_},
        {"MMProj", mm_proj_path_},
        {"Version", model_version_},
        {"ClipModel", clip_model_},
        {"ClipLModel", clip_l_model_},
        {"T5Model", t5_model_},
        {"Devices", devices_},
        {"MMDitModel", mmdit_model_},
        {"VaeModel", vae_model_}
    };

    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Could not open file for writing: {} ", file_path);
            return -1;
        }
        file << config.dump(4); // 缩进 4 格，方便阅读
    } catch (const std::exception& ex) {
        LOG_ERROR("Error writing JSON file {}: {}", file_path, ex.what());
        return -1;
    }

    return 0;
}

HLIE_NAMESPACE_END
