#ifndef HLIE_ASR_REALTIME_CONTRACT_H
#define HLIE_ASR_REALTIME_CONTRACT_H

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

inline std::string normalize_whisper_language(std::string language) {
    const auto is_space = [](const unsigned char ch) {
        return std::isspace(ch) != 0;
    };
    language.erase(language.begin(),
                   std::find_if_not(language.begin(), language.end(), is_space));
    language.erase(
        std::find_if_not(language.rbegin(), language.rend(), is_space).base(),
        language.end());
    std::transform(language.begin(), language.end(), language.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });

    if (language.empty() || language == "auto" || language == "automatic") {
        return "auto";
    }
    if (language == "chinese" || language == "mandarin" || language == "cmn" ||
        language.rfind("zh-", 0) == 0 || language.rfind("zh_", 0) == 0) {
        return "zh";
    }
    if (language == "english" || language.rfind("en-", 0) == 0 ||
        language.rfind("en_", 0) == 0) {
        return "en";
    }
    return language;
}

struct WhisperStreamRequest {
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    int sample_rate = 16000;
    std::string model;
    std::string prompt;
    std::string language;
    std::string response_format = "json";
    std::string path;
    std::function<bool()> should_stop;
};

using TranscriptCallback = std::function<void(const std::string&)>;
using AudioHandler = std::function<std::string(
    const WhisperStreamRequest&, const TranscriptCallback&)>;

HLIE_NAMESPACE_END

#endif // HLIE_ASR_REALTIME_CONTRACT_H
