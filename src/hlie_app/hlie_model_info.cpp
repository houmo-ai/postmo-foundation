#include "hlie_app/hlie_model_info.h"

#include <fstream>

#include "hlie_base/hlie_logger.h"
#include "json.hpp"


HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return result;
    }
}

int HLIEModelInfo::parse_from_file(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Error: Could not open file: {}", file_path);
            return -3;
        }

        nlohmann::json config;
        file >> config;

        model_name_ = config.value("ModelName", "");
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
        mmdit_model_ = config.value("MMDitModel", "");
        vae_model_  = config.value("VaeModel", "");
    } catch (const nlohmann::json::exception& ex) {
        LOG_ERROR("JSON Error: {}", ex.what());
        return -2;
    } catch (const std::exception& ex) {
        LOG_ERROR("General Error: {}", ex.what());
        return -1;
    }

    return 0;
}

HLIE_NAMESPACE_END
