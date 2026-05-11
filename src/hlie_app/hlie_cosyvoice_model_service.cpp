#include "hlie_app/hlie_cosyvoice_model_service.h"

#include <algorithm>
#include <array>
#include <regex>
#include <string>


#include <miniaudio.h>
#include <samplerate.h>
#include <whisper.h>
#include "json.hpp"

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/models/hlie_cosyvoice_model.h"
#include "hlie_base/hlie_audio.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"
#include "hlie_base/hlie_util.h"

#define _USE_MATH_DEFINES

using json = nlohmann::ordered_json;

namespace {

    // =========================================================================
    // CosyVoice3 Text Processing Utilities
    // =========================================================================

    // Punctuation marks for text splitting - using std::string to support UTF-8 Chinese characters
    const std::vector<std::string> kChinesePunctuation = {"。", "？", "！", "；", "：", "、", ".", "?", "!", ";"};
    const std::vector<std::string> kEnglishPunctuation = {".", "?", "!", ";", ":"};

    bool ContainsChinese(const std::string& text) {
        // Check for Chinese characters (Unicode range: \u4e00-\u9fff)
        // In UTF-8, Chinese characters are 3 bytes starting with 0xE4-0xE9

        for (size_t i = 0; i < text.length();) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // Check for 3-byte UTF-8 characters in Chinese range
            if ((c & 0xF0) == 0xE0) {
                if (i + 2 < text.length()) {
                    // Chinese characters: \u4e00-\u9fff
                    // UTF-8 encoding:
                    // \u4e00: E4 B8 80
                    // \u9fff: E9 BF BF
                    unsigned char b1 = static_cast<unsigned char>(text[i + 1]);
                    unsigned char b2 = static_cast<unsigned char>(text[i + 2]);

                    // Check if in range E4-E9 with appropriate second byte range
                    if (c >= 0xE4 && c <= 0xE9) {
                        if (c == 0xE4 && b1 >= 0xB8)
                            return true; // \u4e00-\u4fff
                        if (c >= 0xE5 && c <= 0xE8)
                            return true; // \u5000-\u8fff
                        if (c == 0xE9 && b1 <= 0xBF)
                            return true; // \u9000-\u9fff
                    }
                }
                i += 3;
            } else if (c < 128) {
                i++;
            } else if ((c & 0xE0) == 0xC0) {
                i += 2;
            } else if ((c & 0xF8) == 0xF0) {
                i += 4;
            } else {
                i++;
            }
        }
        return false;
    }


    std::string TrimAsciiWhitespace(const std::string& text) {
        size_t start = text.find_first_not_of(" \t\n\r\f\v");
        if (start == std::string::npos) {
            return "";
        }

        size_t end = text.find_last_not_of(" \t\n\r\f\v");
        return text.substr(start, end - start + 1);
    }

    void ReplaceAll(std::string* text, const std::string& from, const std::string& to) {
        if (text == nullptr || from.empty()) {
            return;
        }

        size_t pos = 0;
        while ((pos = text->find(from, pos)) != std::string::npos) {
            text->replace(pos, from.length(), to);
            pos += to.length();
        }
    }

    bool StartsWithAt(const std::string& text, size_t pos, const std::string& token) {
        return pos + token.length() <= text.length() && text.compare(pos, token.length(), token) == 0;
    }

    size_t Utf8CharLen(const std::string& text, size_t pos) {
        if (pos >= text.size()) {
            return 0;
        }

        unsigned char c = static_cast<unsigned char>(text[pos]);
        if (c < 0x80) {
            return 1;
        }
        if ((c & 0xE0) == 0xC0) {
            return 2;
        }
        if ((c & 0xF0) == 0xE0) {
            return 3;
        }
        if ((c & 0xF8) == 0xF0) {
            return 4;
        }
        return 1;
    }


    std::string ReplaceBlank(const std::string& text) {
        // Keep spaces only when they separate adjacent ASCII segments

        std::string result;
        for (size_t i = 0; i < text.length(); ++i) {
            char c = text[i];

            if (c == ' ') {
                // Check if space is between two ASCII non-space characters
                bool prev_is_ascii = (i > 0) && (static_cast<unsigned char>(text[i - 1]) < 128) && (text[i - 1] != ' ');
                bool next_is_ascii = (i + 1 < text.length()) && (static_cast<unsigned char>(text[i + 1]) < 128) && (text[i + 1] != ' ');

                if (prev_is_ascii && next_is_ascii) {
                    result += c;
                }
            } else {
                result += c;
            }
        }

        return result;
    }


    std::string ReplaceCornerMark(const std::string& text) {
        std::string result = text;

        // Replace superscript 2 ("²") with Chinese word for "square"
        // UTF-8: ² = \xc2\xb2, Chinese "square" = \xe5\xb9\xb3\xe6\x96\xb9
        size_t pos = 0;
        while ((pos = result.find("\xc2\xb2", pos)) != std::string::npos) {
            result.replace(pos, 2, "\xe5\xb9\xb3\xe6\x96\xb9"); // Chinese "square"
            pos += 6;
        }

        // Replace superscript 3 ("³") with Chinese word for "cube"
        // UTF-8: ³ = \xc2\xb3, Chinese "cube" = \xe7\xab\x8b\xe6\x96\xb9
        pos = 0;
        while ((pos = result.find("\xc2\xb3", pos)) != std::string::npos) {
            result.replace(pos, 2, "\xe7\xab\x8b\xe6\x96\xb9"); // Chinese "cube"
            pos += 6;
        }

        return result;
    }


    std::string RemoveBracket(const std::string& text) {
        std::string result = text;

        // Remove Chinese full-width brackets: （ ） 【 】
        // UTF-8 codes:
        // （ = \xef\xbc\x88
        // ） = \xef\xbc\x89
        // 【 = \xe3\x80\x90
        // 】 = \xe3\x80\x91

        std::vector<std::pair<std::string, std::string>> replacements = {
            {"\xef\xbc\x88", ""}, // Remove "（"
            {"\xef\xbc\x89", ""}, // Remove "）"
            {"\xe3\x80\x90", ""}, // Remove "【"
            {"\xe3\x80\x91", ""}, // Remove "】"
            {"`", ""}, // Remove backtick
            {"\xe2\x80\x94\xe2\x80\x94", " "} // Replace "——" with space
        };

        for (const auto& rep : replacements) {
            size_t pos = 0;
            while ((pos = result.find(rep.first, pos)) != std::string::npos) {
                result.replace(pos, rep.first.length(), rep.second);
                pos += rep.second.length();
            }
        }

        return result;
    }


    std::string NormalizeChinese(const std::string& text) {
        // Simplified implementation without wetext library

        std::string result = text;

        // Remove newline characters
        result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());

        // Replace blank (keep spaces only between ASCII)
        result = ReplaceBlank(result);

        // Replace corner marks (superscript symbols)
        result = ReplaceCornerMark(result);

        // Replace " - " with Chinese comma "，".
        ReplaceAll(&result, " - ", "\xef\xbc\x8c");

        // Remove brackets
        result = RemoveBracket(result);

        // Note: Removed std::regex for trailing punctuation as it corrupts UTF-8
        // Chinese text. The trailing punctuation handling is done in split_paragraph.

        return result;
    }


    bool IsOnlyPunctuation(const std::string& text) {
        // Check if string contains only punctuation/symbols

        // Common ASCII punctuation
        const std::string ascii_punct = ".,!?;:'\"()-[]{}<>@#$%^&*_+=|\\~/`";

        // Chinese punctuation (UTF-8)
        const std::vector<std::string> chinese_punct = {
            "\xe3\x80\x82", // 。
            "\xef\xbc\x8c", // ，
            "\xef\xbc\x9f", // ？
            "\xef\xbc\x81", // ！
            "\xef\xbc\x9b", // ；
            "\xef\xbc\x9a", // ：
            "\xe3\x80\x90", // 【
            "\xe3\x80\x91", // 】
            "\xef\xbc\x88", // （
            "\xef\xbc\x89", // ）
            "\xe3\x80\x8c", // "
            "\xe3\x80\x8d", // "
            "\xe2\x80\x9c", // "
            "\xe2\x80\x9d", // "
            "\xef\xbc\x8e", // ．
        };

        if (text.empty()) {
            return true;
        }

        size_t i = 0;
        while (i < text.length()) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // Check ASCII punctuation
            if (c < 128) {
                if (ascii_punct.find(c) == std::string::npos && !std::isspace(c)) {
                    return false;
                }
                i++;
                continue;
            }

            // Check for multi-byte punctuation
            bool found = false;
            for (const auto& p : chinese_punct) {
                if (i + p.length() <= text.length() && text.compare(i, p.length(), p) == 0) {
                    i += p.length();
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Not a known punctuation character
                return false;
            }
        }

        return true;
    }

    /**
     * Check if text contains any Chinese characters (U+4E00 - U+9FFF).
     */
    bool contains_chinese(const std::string& text) {
        for (size_t i = 0; i < text.size();) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // 检测 UTF-8 编码的中文字符 (3字节: 0xE0 开头)
            if ((c & 0xF0) == 0xE0) {
                // 检查是否是中文范围
                if (i + 2 < text.size()) {
                    unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
                    unsigned char c3 = static_cast<unsigned char>(text[i + 2]);

                    // 解析 UTF-8 中文字符的码点
                    int codepoint = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

                    // 中文范围: U+4E00 - U+9FFF
                    if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
                        return true;
                    }
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        return false;
    }

    /**
     * Replace superscript symbols with spoken Chinese equivalents.
     */
    std::string replace_corner_mark(std::string text) {
        text = std::regex_replace(text, std::regex("²"), "平方");
        text = std::regex_replace(text, std::regex("³"), "立方");
        return text;
    }

    /**
     * Remove decorative brackets and normalize special separators.
     */
    std::string remove_bracket(std::string text) {
        text = std::regex_replace(text, std::regex(R"(（)"), "");
        text = std::regex_replace(text, std::regex(R"(）)"), "");
        text = std::regex_replace(text, std::regex(R"(\[)"), "");
        text = std::regex_replace(text, std::regex(R"(\])"), "");
        text = std::regex_replace(text, std::regex("`"), "");
        text = std::regex_replace(text, std::regex("——"), " ");
        return text;
    }

    /**
     * Keep spaces only when they separate adjacent ASCII segments.
     */
    std::string replace_blank(const std::string& text) {
        if (text.empty())
            return text;

        std::string out_str;
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == ' ') {
                bool prev_ascii = (i > 0) && isascii(static_cast<unsigned char>(text[i - 1])) && text[i - 1] != ' ';
                bool next_ascii = (i + 1 < text.size()) && isascii(static_cast<unsigned char>(text[i + 1])) && text[i + 1] != ' ';
                if (prev_ascii && next_ascii) {
                    out_str += c;
                }
            } else {
                out_str += c;
            }
        }
        return out_str;
    }

    /**
     * Check if the string contains only punctuation or symbols.
     */
    bool is_only_punctuation(const std::string& text) {
        // Match strings that consist only of punctuation marks or are empty
        static const std::regex punctuation_pattern(R"(^[\p{P}\p{S}]*$)");
        return std::regex_match(text, punctuation_pattern);
    }

    /**
     * Convert Arabic numerals in text into spoken-word expressions.
     * NOTE: This requires an inflect library (e.g., Python's inflect engine).
     * Currently returns the number as string without conversion.
     * TODO: Implement number_to_words or integrate with inflect library.
     */
    std::string spell_out_number(const std::string& text) {
        std::string new_text;
        size_t st = std::string::npos;

        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (!std::isdigit(c)) {
                if (st != std::string::npos) {
                    // Convert number substring
                    std::string num_str = text.substr(st, i - st);
                    // TODO: Replace with actual number-to-words conversion
                    // For now, keep the numeric string as-is
                    // Example: inflect_parser.number_to_words(num_str)
                    new_text += num_str;
                    st = std::string::npos;
                }
                new_text += text[i];
            } else {
                if (st == std::string::npos) {
                    st = i;
                }
            }
        }
        if (st != std::string::npos && st < text.size()) {
            std::string num_str = text.substr(st);
            // TODO: Replace with actual number-to-words conversion
            new_text += num_str;
        }
        return new_text;
    }

    /**
     * Calculate utterance length based on language.
     * For Chinese: character count (UTF-8 aware).
     * For English: token count (simplified to character count).
     */
    size_t calc_utt_length(const std::string& text, const std::string& lang) {
        if (lang == "zh") {
            // Count UTF-8 characters properly
            size_t count = 0;
            for (size_t i = 0; i < text.size();) {
                unsigned char c = static_cast<unsigned char>(text[i]);
                if (c < 0x80) {
                    // ASCII
                    ++count;
                    ++i;
                } else if ((c & 0xE0) == 0xC0) {
                    // 2-byte UTF-8
                    ++count;
                    i += 2;
                } else if ((c & 0xF0) == 0xE0) {
                    // 3-byte UTF-8 (including Chinese)
                    ++count;
                    i += 3;
                } else if ((c & 0xF8) == 0xF0) {
                    // 4-byte UTF-8
                    ++count;
                    i += 4;
                } else {
                    ++i;
                }
            }
            return count;
        }
        // TODO: Use actual tokenizer for English
        // return tokenizer.encode(text).size();
        return text.size();
    }

    /**
     * Check if the tail utterance is short enough to merge.
     */
    bool should_merge(const std::string& text, const std::string& lang, size_t merge_len = 20) {
        return calc_utt_length(text, lang) < merge_len;
    }

    /**
     * Get punctuation marks based on language and comma_split flag.
     */
    std::vector<std::string> get_punctuations(const std::string& lang, bool comma_split = false) {
        std::vector<std::string> pounc;
        if (lang == "zh") {
            pounc = kChinesePunctuation;
        } else {
            pounc = kEnglishPunctuation;
        }
        if (comma_split) {
            pounc.push_back("，");
            pounc.push_back(",");
        }
        return pounc;
    }

    /**
     * Check if a character at position i is a punctuation mark.
     */
    bool is_punct_at(const std::string& text, size_t i, const std::vector<std::string>& pounc) {
        if (i >= text.size())
            return false;

        // Check multi-byte UTF-8 punctuation first
        for (const auto& p : pounc) {
            if (p.size() > 1) {
                if (text.substr(i, p.size()) == p) {
                    return true;
                }
            }
        }

        // Check single-byte punctuation
        unsigned char c = static_cast<unsigned char>(text[i]);
        for (const auto& p : pounc) {
            if (p.size() == 1) {
                if (c == static_cast<unsigned char>(p[0])) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Split text into utterances that satisfy token-length constraints.
     *
     * Split logic:
     * 1. Per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
     * 2. Calculate sentence len according to lang
     * 3. Split sentence according to punctuation
     */
    std::vector<std::string> split_paragraph(const std::string& text, const std::string& lang = "zh", const size_t token_max_n = 80,
                                             const size_t token_min_n = 60, const size_t merge_len = 20, const bool comma_split = false) {

        if (text.empty())
            return {};

        std::string processed_text = text;
        auto pounc = get_punctuations(lang, comma_split);

        // Ensure text ends with punctuation
        char last_char = processed_text.back();
        bool ends_with_punct = false;
        for (const auto& p : pounc) {
            if (p.size() == 1 && last_char == p[0]) {
                ends_with_punct = true;
                break;
            } else if (p.size() > 1) {
                if (processed_text.size() >= p.size() && processed_text.substr(processed_text.size() - p.size()) == p) {
                    ends_with_punct = true;
                    break;
                }
            }
        }

        if (!ends_with_punct) {
            processed_text += (lang == "zh" ? "。" : ".");
        }

        // Split by punctuation
        std::vector<std::string> utts;
        size_t st = 0;
        for (size_t i = 0; i < processed_text.size(); ++i) {
            if (is_punct_at(processed_text, i, pounc)) {
                if (i > st) {
                    std::string sentence = processed_text.substr(st, i - st + 1);
                    utts.push_back(sentence);
                }
                // Handle quote following punctuation
                if (i + 1 < processed_text.size() && (processed_text[i + 1] == '"' || processed_text[i + 1] == '\"')) {
                    if (!utts.empty()) {
                        utts.back() += processed_text[i + 1];
                        st = i + 2;
                    } else {
                        st = i + 1;
                    }
                } else {
                    st = i + 1;
                }
            }
        }

        // Merge utterances based on length constraints
        std::vector<std::string> final_utts;
        std::string cur_utt;
        for (const auto& utt : utts) {
            size_t cur_len = calc_utt_length(cur_utt + utt, lang);
            size_t cur_utt_len = calc_utt_length(cur_utt, lang);

            if (cur_len > token_max_n && cur_utt_len > token_min_n) {
                final_utts.push_back(cur_utt);
                cur_utt.clear();
            }
            cur_utt += utt;
        }

        if (!cur_utt.empty()) {
            if (should_merge(cur_utt, lang, merge_len) && !final_utts.empty()) {
                final_utts.back() += cur_utt;
            } else {
                final_utts.push_back(cur_utt);
            }
        }

        return final_utts;
    }


    bool read_audio_data(const std::string& fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
        std::vector<uint8_t> audio_data; // used for pipe input from stdin or ffmpeg decoding output

        ma_result result;
        ma_decoder_config decoder_config;
        ma_decoder decoder;

        decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);

        if (fname == "-") {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif

            uint8_t buf[1024];
            while (true) {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                audio_data.insert(audio_data.end(), buf, buf + n);
            }

            if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {

                fprintf(stderr, "Error: failed to open audio data from stdin (%s)\n", ma_result_description(result));

                return false;
            }

            fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, audio_data.size());
        } else if (((result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder)) != MA_SUCCESS)) {
#if defined(WHISPER_FFMPEG)
            if (ffmpeg_decode_audio(fname, audio_data) != 0) {
                fprintf(stderr, "error: failed to ffmpeg decode '%s'\n", fname.c_str());

                return false;
            }

            if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
                fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

                return false;
            }
#else
            if ((result = ma_decoder_init_memory(fname.c_str(), fname.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
                fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

                return false;
            }
#endif
        }

        ma_uint64 frame_count;
        ma_uint64 frames_read;

        if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
            fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));

            return false;
        }

        pcmf32.resize(stereo ? frame_count * 2 : frame_count);

        if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
            fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));

            return false;
        }

        if (stereo) {
            std::vector<float> stereo_data = pcmf32;
            pcmf32.resize(frame_count);

            for (uint64_t i = 0; i < frame_count; i++) {
                pcmf32[i] = (stereo_data[2 * i] + stereo_data[2 * i + 1]);
            }

            pcmf32s.resize(2);
            pcmf32s[0].resize(frame_count);
            pcmf32s[1].resize(frame_count);
            for (uint64_t i = 0; i < frame_count; i++) {
                pcmf32s[0][i] = stereo_data[2 * i];
                pcmf32s[1][i] = stereo_data[2 * i + 1];
            }
        }

        ma_decoder_uninit(&decoder);

        return true;
    }

    struct AudioData {
        std::vector<float> samples; // Mono audio [n_samples]
        std::vector<std::vector<float>> stereo_samples; // Stereo if needed
    };


    /**
     * Resample audio data to target sample rate.
     */
    std::vector<float> resample_audio(const std::vector<float>& input, const int input_sr, const int output_sr) {
        if (input_sr == output_sr) {
            return input;
        }

        double ratio = static_cast<double>(output_sr) / input_sr;
        int output_length = static_cast<int>(input.size() * ratio);

        std::vector<float> output(output_length);

        SRC_DATA src_data;
        src_data.data_in = input.data();
        src_data.input_frames = static_cast<long>(input.size());
        src_data.data_out = output.data();
        src_data.output_frames = static_cast<long>(output_length);
        src_data.src_ratio = ratio;
        src_data.end_of_input = 1;

        if (const int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1)) {
            throw std::runtime_error("Resampling error: " + std::string(src_strerror(error)));
        }

        // Resize to actual output frames
        output.resize(src_data.output_frames_gen);
        LOG_INFO("resample audio done");
        return output;
    }


    struct wav_header {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t chunk_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_chunk_size = 16;
        uint16_t audio_format = 1; // PCM
        uint16_t num_channels = 1; // Mono
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    };

    bool save_wav(const std::string& fname, const std::vector<float>& data, const int sample_rate) {
        std::ofstream file(fname, std::ios::binary);
        if (!file) {
            LOG_ERROR("save_wav failed to open file {} for writing.", fname.c_str());
            return false;
        }

        wav_header header;
        header.sample_rate = sample_rate;
        header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.data_size = data.size() * (header.bits_per_sample / 8);
        header.chunk_size = 36 + header.data_size;

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto& sample : data) {
            int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
            file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
        }

        return file.good();
    }

    std::string wav_to_bytes(const std::vector<float>& data, const int sample_rate) {
        wav_header header;
        header.sample_rate = sample_rate;
        header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.data_size = static_cast<uint32_t>(data.size() * (header.bits_per_sample / 8));
        header.chunk_size = 36 + header.data_size;

        std::string bytes;
        bytes.reserve(sizeof(header) + data.size() * sizeof(int16_t));
        bytes.append(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto& sample : data) {
            const int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
            bytes.append(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
        }

        return bytes;
    }

    AudioData load_wav(const std::string& wav_path, int target_sr = 16000) {
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        // read_audio_data always outputs WHISPER_SAMPLE_RATE (16000)
        if (!read_audio_data(wav_path, pcmf32, pcmf32s, false)) {
            throw std::runtime_error("Failed to load audio: " + wav_path);
        }

        // Resample to target_sr if needed
        if (constexpr int source_sr = WHISPER_SAMPLE_RATE; target_sr != source_sr) {
            LOG_INFO("Resampling from {} to {}", source_sr, target_sr);
            pcmf32 = resample_audio(pcmf32, source_sr, target_sr);
        }

        LOG_INFO("Loaded wav audio to {}", target_sr);
        return AudioData{.samples = std::move(pcmf32), .stereo_samples = std::move(pcmf32s)};
    }

} // namespace

HLIE_NAMESPACE_BEGIN

int CosyVoiceModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info,
                                const std::shared_ptr<HLIEDeviceService>& device_service) {
    blocking_queue_ = std::make_shared<HLIEBlockingQueue<QueueItem>>();
    cosyvoice_model_ = std::make_shared<HLIECosyVoiceModel>();
    device_service_ = device_service;

    auto devices = model_info.get_devices();

    if (const int replicas = Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1); replicas > 1) {
        is_multi_instance_ = true;
        if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(HLIEModelType::COSYVOICE); !val) {
            LOG_ERROR("CosyVoiceModelService cannot acquire available device");
            return -1;
        } else {
            devices_.push_back(*val);
            LOG_INFO("CosyVoiceModelService using device: {}", val.value());
        }
    } else { // 单实例模式可以定制Model Devices
        if (!devices.empty()) {
            if (!HlieUtil::is_valid_devices(devices)) {
                LOG_ERROR("CosyVoiceModelService cannot acquire model specific device \"{}\"", devices);
                return -1;
            }

            if (HlieUtil::parse_device(devices, devices_) < 0) {
                LOG_ERROR("CosyVoiceModelService parse_device specific device \"{}\" failed", devices);
                return -1;
            }

            if (device_service_->acquire_mem_on_devices(devices_, 1200) < 0) {
                return -1;
            }

            LOG_INFO("CosyVoiceModelService using model specific device: {}", devices_.at(0));
        } else {
            uint32_t device;
            if (device_service_->acquire_mem(4000, device) < 0) {
                return -1;
            }

            devices_.push_back(device);
            LOG_INFO("CosyVoiceModelService using model specific device: {}", device);
        }
    }

    cosyvoice_model_->set_devices(devices_);
    const auto& model = model_info.get_model_path();


    cosyvoice_model_->load_model(model);
    LOG_INFO("load_model done");
    cosyvoice_model_->init();

    audio_processor_ = std::make_shared<HLIEAudio>();

    model_key_ = model_info.model_key_;
    /*
    websocket_service_ = std::make_unique<HLIEWebSocketService>();
    websocket_service_->init();
    */
    return 0;
}

void CosyVoiceModelService::run() {
    if (worker_thread_) {
        LOG_WARN("CosyVoiceModelService thread is already running.");
        return;
    }
    /*
        websocket_service_->run([this](const WhisperStreamRequest& req) { return this->handle_stream(req); }, 7998);
    */
    state_.store(SERVER_STATE_READY);
    should_stop_ = false;
    worker_thread_ = std::make_unique<std::thread>(&CosyVoiceModelService::process, this);
    LOG_INFO("CosyVoiceModelService thread started.");
}

void CosyVoiceModelService::process() {
    try {
        while (!should_stop_) {
            const auto [request, promise] = blocking_queue_->pop(); // 阻塞直到有数据或队列关闭
            if (should_stop_) {
                break; // 防止 stop() 后仍有残留任务
            }

            try {
                auto start = std::chrono::steady_clock::now();
                auto speech_result = speech(request);
                auto end = std::chrono::steady_clock::now();
                // 构造结果
                CosyVoiceResult result;
                result.wav = speech_result;
                result.success = !speech_result.empty();
                result.predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                LOG_INFO("speech time: {} ms", result.predicted_ms);

                // 通知调用者
                promise->set_value(result);
            } catch (const std::exception& e) {
                // 错误处理
                CosyVoiceResult result;
                result.success = false;
                result.error_message = e.what();
                promise->set_value(result);
            }
        }
    } catch (const std::runtime_error&) {
        LOG_WARN("CosyVoiceModelService process exited");
    }
}

void CosyVoiceModelService::stop() {
    if (!worker_thread_) {
        return;
    }

    should_stop_ = true;
    if (blocking_queue_) {
        blocking_queue_->close(); // 通知队列关闭，唤醒可能阻塞的 pop()
    }

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();

    if (websocket_service_) {
        websocket_service_->stop();
    }

    if (is_multi_instance_) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
    } else {
        device_service_->release_mem_on_devices(devices_, 2000);
    }
    devices_.clear();
}

int CosyVoiceModelService::get_state() {
    return state_;
}

int CosyVoiceModelService::handle_health(nlohmann::ordered_json& resp) {
    return 0;
}

void CosyVoiceModelService::handle_models(nlohmann::ordered_json& resp) const {
    resp["models"].push_back({
        {"name", "CosyVoice"},
        {"model", "CosyVoice"},
    });

    resp["data"].push_back({
        {"id", "CosyVoice"},
        {"object", "model"},
        {"created", std::time(0)},
        {"owned_by", "HLIECpp"},
    });
}

std::vector<float> CosyVoiceModelService::handle_stream(const WhisperStreamRequest& request) {

    CosyVoiceRequest req;
    req.content = "stream";

    const auto start = std::chrono::steady_clock::now();
    auto speech_result = speech(req);
    const auto end = std::chrono::steady_clock::now();
    auto predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG_INFO("tts stream cost: {} ms", predicted_ms);
    return speech_result;
}

void CosyVoiceModelService::llm_job(const std::vector<int>& text, const std::vector<int>& prompt_text,
                                    const std::vector<int>& llm_prompt_speech_token, const std::vector<float>& llm_embedding,
                                    const std::string& uuid) {
    LOG_INFO("LLM job started for uuid: {}", uuid);

    const int max_silent_token_num = 5;
    int cur_silent_token_num = 0;

    LLMPerf llm_perf;

    try {
        // Run LLM inference
        auto generated_tokens =
            cosyvoice_model_->llm_inference(text, static_cast<int>(text.size()), prompt_text, static_cast<int>(prompt_text.size()),
                                            llm_prompt_speech_token, static_cast<int>(llm_prompt_speech_token.size()), llm_embedding,
                                            25, // sampling
                                            20.0f, // max_token_text_ratio
                                            2.0f, // min_token_text_ratio
                                            &llm_perf);

        // Process generated tokens and filter silent tokens
        for (int token_id : generated_tokens) {
            if (std::find(kSilentTokens.begin(), kSilentTokens.end(), token_id) != kSilentTokens.end()) {
                cur_silent_token_num++;
                if (cur_silent_token_num > max_silent_token_num) {
                    LOG_DEBUG("skip current silent_token {}", cur_silent_token_num);
                    continue;
                }
            } else {
                cur_silent_token_num = 0;
            }

            // Store the token
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_[uuid].push_back(token_id);
            }
        }

        // Mark LLM as ended
        {
            std::lock_guard<std::mutex> lock(lock_);
            llm_end_dict_[uuid] = true;
            perf_by_uuid_[uuid] = {llm_perf.llm_total_s};
        }

        LOG_INFO("LLM job completed for uuid: {}, generated {} tokens", uuid, generated_tokens.size());

    } catch (const std::exception& e) {
        LOG_ERROR("LLM job failed for uuid: {}, error: {}", uuid, e.what());
        {
            std::lock_guard<std::mutex> lock(lock_);
            llm_end_dict_[uuid] = true;
        }
        throw;
    }
}

std::future<CosyVoiceResult> CosyVoiceModelService::speech_async(const CosyVoiceRequest& request) const {
    const auto promise = std::make_shared<std::promise<CosyVoiceResult>>();
    auto future = promise->get_future();

    // 创建队列项
    QueueItem item;
    item.request = request;
    item.promise = promise;

    if (blocking_queue_) {
        blocking_queue_->push(item);
    }

    return future;
}

CosyVoiceResult CosyVoiceModelService::speech_sync(const CosyVoiceRequest& request) const {
    auto future = speech_async(request);
    return future.get();
}

std::vector<float> CosyVoiceModelService::speech(const CosyVoiceRequest& request) {
    std::vector<float> final_audio;
    try {
        LOG_INFO("cosyvoice3 ready to run {}", request.content);

        const std::string prompt = "You are a helpful assistant.<|endofprompt|>" +
            (request.prompt_text.empty() ? "希望你以后能够做的比我还好呦。" : request.prompt_text);

        auto prompt_text_normalized = text_normalize(prompt, false);
        std::string prompt_text_str = prompt_text_normalized.empty() ? prompt : prompt_text_normalized[0];

        LOG_INFO("text_normalize prompt done, prompt_text_str: {}", prompt_text_str);
        auto tts_text = text_normalize(request.content, true);
        if (tts_text.empty()) {
            tts_text.push_back(request.content);
        }

        LOG_INFO("TTS text_normalize request done, segment count: {}", tts_text.size());
#ifdef WIN32
        char exePath[260];
        GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
        const auto program_path = std::string(exePath);
        const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
        const std::string default_prompt_wav = HlieUtil::join_paths(program_dir, "..", R"(samples/zero_shot_prompt.wav)");
#else
        const std::string default_prompt_wav = R"(samples/zero_shot_prompt.wav)";
#endif
        const std::string prompt_wav = request.prompt_wav_path.empty() ? default_prompt_wav : request.prompt_wav_path;

        FrontendInput prompt_cache;
        bool has_prompt_cache = false;

        for (auto text : tts_text) {

            if (text.length() < 0.5 * prompt_text_str.length()) {
                LOG_WARN("WARNING: Synthesis text is shorter than prompt text. This may lead to bad performance.");
            }

            // Get frontend input (tokens, embeddings, etc.)
            FrontendInput model_input;
            if (!has_prompt_cache) {
                model_input = frontend_zero_shot(text, prompt_text_str, prompt_wav);
                prompt_cache.prompt_tokens = model_input.prompt_tokens;
                prompt_cache.prompt_text_len = model_input.prompt_text_len;
                prompt_cache.llm_prompt_speech_token = model_input.llm_prompt_speech_token;
                prompt_cache.llm_prompt_speech_token_len = model_input.llm_prompt_speech_token_len;
                prompt_cache.flow_prompt_speech_token = model_input.flow_prompt_speech_token;
                prompt_cache.flow_prompt_speech_token_len = model_input.flow_prompt_speech_token_len;
                prompt_cache.prompt_speech_feat = model_input.prompt_speech_feat;
                prompt_cache.embedding = model_input.embedding;
                has_prompt_cache = true;
            } else {
                model_input.prompt_tokens = prompt_cache.prompt_tokens;
                model_input.prompt_text_len = prompt_cache.prompt_text_len;
                model_input.llm_prompt_speech_token = prompt_cache.llm_prompt_speech_token;
                model_input.llm_prompt_speech_token_len = prompt_cache.llm_prompt_speech_token_len;
                model_input.flow_prompt_speech_token = prompt_cache.flow_prompt_speech_token;
                model_input.flow_prompt_speech_token_len = prompt_cache.flow_prompt_speech_token_len;
                model_input.prompt_speech_feat = prompt_cache.prompt_speech_feat;
                model_input.embedding = prompt_cache.embedding;
                model_input.text_len = cosyvoice_model_->extract_text_token(text, model_input.text_tokens);
            }

            LOG_INFO("synthesis text {}", text);

            // Generate unique ID for this request
            std::string this_uuid = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

            // Initialize shared state
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_[this_uuid] = {};
                llm_end_dict_[this_uuid] = false;
            }

            // Run LLM job (blocking)
            auto llm_start = std::chrono::steady_clock::now();
            llm_job(model_input.text_tokens, model_input.prompt_tokens, model_input.llm_prompt_speech_token, model_input.embedding,
                    this_uuid);
            auto llm_end = std::chrono::steady_clock::now();
            float llm_time = std::chrono::duration<float>(llm_end - llm_start).count();

            // Get generated tokens
            std::vector<int> this_tts_speech_token;
            {
                std::lock_guard<std::mutex> lock(lock_);
                this_tts_speech_token = tts_speech_token_dict_[this_uuid];
            }

            LOG_INFO("LLM generated {} speech tokens in {:.2f}s", this_tts_speech_token.size(), llm_time);

            // Run token2wav (flow decoder)
            std::vector<float> utterance_audio = cosyvoice_model_->token2wav(this_tts_speech_token, model_input.flow_prompt_speech_token,
                                                                             model_input.prompt_speech_feat, model_input.embedding, 1.0f);

            // Clean up
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_.erase(this_uuid);
                llm_end_dict_.erase(this_uuid);
                hift_cache_dict_.erase(this_uuid);
            }

            final_audio.insert(final_audio.end(), utterance_audio.begin(), utterance_audio.end());
        }

        LOG_INFO("TTS inference completed");
        return final_audio;
    } catch (const std::exception& e) {
        LOG_ERROR("speech exception: {}", e.what());
        return final_audio;
    }
    LOG_INFO("speech done");
    return final_audio;
}

void CosyVoiceModelService::handle_speech(const httplib::Request& req, httplib::Response& res) {
    try {
        const json body = json::parse(req.body);
        std::string content = json_value(body, "input", std::string());
        if (TrimAsciiWhitespace(content).empty()) {
            res.status = 400;
            res.set_content(R"({"error":"input is required"})", "application/json");
            return;
        }

        CosyVoiceRequest cosy_voice_request;
        cosy_voice_request.content = content;
        cosy_voice_request.prompt_text = json_value(body, "prompt_text", std::string());
        cosy_voice_request.prompt_wav_path = json_value(body, "prompt_wav_path", std::string());

        if (auto [success, wav, error_message, predicted_ms] = speech_sync(cosy_voice_request); !success) {
            // handle failure or early abort
            if (req.is_connection_closed()) {
                // log client disconnect
                fprintf(stderr, "client disconnected, aborted processing\n");
                res.status = 499; // Client Closed Request (nginx convention)
                res.set_content(R"({"error":"client disconnected"})", "application/json");
                return;
            }
            LOG_ERROR("failed to process audio");
            res.status = 500; // Internal Server Error
            const std::string error_resp = R"({"error":"failed to process audio"})";
            res.set_content(error_resp, "application/json");
            return;
        } else {
            auto wav_bytes = wav_to_bytes(wav, 24000);
            res.set_header("Content-Disposition", R"(attachment; filename="speech.wav")");
            res.set_content(std::move(wav_bytes), "audio/wav");
        }

        LOG_INFO("process audio done");
    } catch (const json::parse_error& ex) {
        LOG_ERROR("handle_speech parse ex: {}", ex.what());
        res.status = 400;
        res.set_content(R"({"error":"invalid json body"})", "application/json");
    } catch (const std::exception& ex) {
        LOG_ERROR("handle_speech ex: {}", ex.what());
        res.status = 500;
        res.set_content(R"({"error":"failed to process audio"})", "application/json");
    }
}

std::vector<std::string> CosyVoiceModelService::text_normalize(const std::string& text, bool split) const {
    // Python: skip text_frontend when SSML/special-token marker exists.
    if (text.find("<|") != std::string::npos && text.find("|>") != std::string::npos) {
        return split ? std::vector<std::string>{text} : std::vector<std::string>{text};
    }

    if (text.empty()) {
        return split ? std::vector<std::string>{""} : std::vector<std::string>{text};
    }

    std::string normalized = TrimAsciiWhitespace(text);
    if (normalized.empty()) {
        return split ? std::vector<std::string>{""} : std::vector<std::string>{""};
    }

    // Check if contains Chinese
    bool is_chinese = ContainsChinese(normalized);

    if (is_chinese) {
        normalized = NormalizeChinese(normalized);
        if (split) {
            // Speech rate: 3 chars/sec, max 15s per chunk => 45 chars max
            return SplitParagraph(normalized, 60, 40);
        }
        return {normalized};
    }

    if (split) {
        // For English, we use token count for length calculation
        // Simplified: use same split logic with token-based length
        std::vector<std::string> segments = SplitParagraph(normalized, 80, 60);
        return segments;
    }
    return {normalized};
}

int CosyVoiceModelService::CalcUttLength(const std::string& text, bool is_chinese) const {
    if (is_chinese) {
        // For Chinese, use character count
        // Need to count UTF-8 characters properly
        int count = 0;
        for (size_t i = 0; i < text.length();) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c < 128) {
                count++;
                i++;
            } else if ((c & 0xE0) == 0xC0) {
                count++;
                i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                count++;
                i += 3;
            } else if ((c & 0xF8) == 0xF0) {
                count++;
                i += 4;
            } else {
                i++;
            }
        }
        return count;
    }

    // For English, use token count
    std::vector<int32_t> text_tokens;
    cosyvoice_model_->extract_text_token(text, text_tokens);
    return static_cast<int>(text_tokens.size());
}

bool CosyVoiceModelService::ShouldMerge(const std::string& text, const bool is_chinese, const int merge_len) const {
    return CalcUttLength(text, is_chinese) < merge_len;
}


std::vector<std::string> CosyVoiceModelService::SplitParagraph(const std::string& text, const int max_tokens, const int min_tokens) const {
    // merge_len: threshold for merging short tail segment (proportional to
    // min_tokens)
    const int merge_len = min_tokens / 2;

    bool is_chinese = ContainsChinese(text);

    // Define punctuation for splitting
    std::vector<std::string> punc;
    if (is_chinese) {
        punc = {"\xe3\x80\x82",
                "\xef\xbc\x9f",
                "\xef\xbc\x81", // "。", "？", "！"
                "\xef\xbc\x9b",
                "\xef\xbc\x9a",
                "\xe3\x80\x81",
                "\xef\xbc\x8c", // "；", "：", "、", "，"
                ".",
                "?",
                "!",
                ";"}; // ASCII punctuation
    } else {
        punc = {".", "?", "!", ";", ":"};
    }

    // Add ending punctuation if not present
    std::string work_text = text;
    bool has_ending_punc = false;
    for (const auto& p : punc) {
        if (work_text.length() >= p.length() && work_text.compare(work_text.length() - p.length(), p.length(), p) == 0) {
            has_ending_punc = true;
            break;
        }
    }

    if (!has_ending_punc && !work_text.empty()) {
        if (is_chinese) {
            work_text += "\xe3\x80\x82"; // "。"
        } else {
            work_text += ".";
        }
    }

    // Split by punctuation, following demo.py semantics.
    std::vector<std::string> utts;
    size_t st = 0;

    for (size_t i = 0; i < work_text.length();) {
        bool matched = false;
        for (const auto& p : punc) {
            if (StartsWithAt(work_text, i, p)) {
                // Extract segment before punctuation
                if (i > st) {
                    utts.push_back(work_text.substr(st, i - st) + p);
                }

                // Handle quote after punctuation: Python checks '"' and '"'.
                size_t next_pos = i + p.length();
                if (next_pos < work_text.length()) {
                    if (StartsWithAt(work_text, next_pos, "\"") || StartsWithAt(work_text, next_pos, "\xe2\x80\x9d")) {
                        if (!utts.empty()) {
                            size_t quote_len = Utf8CharLen(work_text, next_pos);
                            utts.back() += work_text.substr(next_pos, quote_len);
                            st = next_pos + quote_len;
                        } else {
                            st = next_pos;
                        }
                    } else {
                        st = next_pos;
                    }
                } else {
                    st = next_pos;
                }
                i += p.length();
                matched = true;
                break;
            }
        }

        if (!matched) {
            i += Utf8CharLen(work_text, i);
        }
    }

    // Merge segments according to token length constraints
    std::vector<std::string> final_utts;
    std::string cur_utt;

    for (const auto& utt : utts) {
        int cur_len = CalcUttLength(cur_utt, is_chinese);
        int combined_len = CalcUttLength(cur_utt + utt, is_chinese);

        if (combined_len > max_tokens && cur_len > min_tokens) {
            final_utts.push_back(cur_utt);
            cur_utt = "";
        }
        cur_utt += utt;
    }

    // Handle remaining segment
    if (!cur_utt.empty()) {
        if (ShouldMerge(cur_utt, is_chinese, merge_len) && !final_utts.empty()) {
            final_utts.back() += cur_utt;
        } else {
            final_utts.push_back(cur_utt);
        }
    }

    // Filter out punctuation-only segments
    std::vector<std::string> filtered_utts;
    for (const auto& utt : final_utts) {
        if (!IsOnlyPunctuation(utt)) {
            filtered_utts.push_back(utt);
        }
    }

    return filtered_utts;
}

FrontendInput CosyVoiceModelService::frontend_zero_shot(const std::string& tts_text, const std::string& prompt_text,
                                                        const std::string& prompt_wav, const int resample_rate) {
    static constexpr int kSampleRate16k = 16000; ///< Sample rate for tokenizer/campplus
    static constexpr int kSampleRate24k = 24000; ///< Sample rate for vocoder
    FrontendInput input;
    input.text_len = cosyvoice_model_->extract_text_token(tts_text, input.text_tokens);

    // TODO: Cache the other repeated pre-calc logic

    input.prompt_text_len = cosyvoice_model_->extract_text_token(prompt_text, input.prompt_tokens);

    LOG_INFO("Start to extract_speech_token");
    const AudioData audio_16k = load_wav(prompt_wav, kSampleRate16k);
    const auto speech_token = extract_speech_token(audio_16k.samples);
    input.flow_prompt_speech_token = speech_token.first.data;
    input.flow_prompt_speech_token_len = speech_token.second;

    input.embedding = extract_spk_embedding(audio_16k.samples);

    input.llm_prompt_speech_token = speech_token.first.data;
    input.llm_prompt_speech_token_len = speech_token.second;

    LOG_INFO("Start to extract_speech_feat");
    const AudioData audio_24k = load_wav(prompt_wav, kSampleRate24k);
    auto speech_feat = audio_processor_->ComputeMelSpectrogram80(audio_24k.samples);
    input.prompt_speech_feat = std::move(speech_feat.data);

    if (resample_rate == 24000) { // cosyvoice2, force speech_feat % speech_token = 2
        // cosyvoice2, force speech_feat % speech_token = 2
        // speech_feat: [1, n_len, n_mel=80] -> flattened size = n_len * 80
        // speech_token: [1, token_len]
        /**
        speech_feat 是 mel 频谱特征，基于 24kHz 音频计算
        speech_token 是 speech tokenizer 的输出，基于 16kHz 音频计算
        关键点：
        24kHz 的 hop_length = 480，所以 30秒 = 1500 帧
        16kHz 的 hop_length = 160，所以 30秒 = 3000 帧
        speech tokenizer 内部有 4:1 的下采样，3000/4 = 750 tokens
        所以：
        24kHz speech_feat: 1500 帧
        16kHz speech_token: 750 tokens
        1500 / 750 = 2，所以确实是 2:1 的比例。
        但为什么需要这个比例？这是因为 CosyVoice2 模型在处理时，speech_feat 和 speech_token 需要对齐。speech tokenizer 是基于 16kHz 的 mel
        特征训练的，而 speech_feat 可能是基于 24kHz 提取的。 从物理上理解： 同一个 30 秒音频，用 24kHz 提取得到 1500 帧 mel 用 16kHz
        提取得到 3000 帧 mel，下采样 4 倍得到 750 tokens 如果要保证它们能对应上，需要： 1500 / 750 = 2 这个 2:1
        的比例反映了两种不同采样率下特征提取的差异。实际上是采样率和帧移配置导致了不同的帧数，而模型需要这个对齐才能正确处理。
        **/
        constexpr int kN_mel = 80;
        int token_len = std::min(speech_feat.n_frames / 2, input.flow_prompt_speech_token_len);
        // Truncate speech_feat to 2 * token_len frames
        input.prompt_speech_feat.resize(2 * token_len * kN_mel);
        // Truncate speech_token to token_len
        input.flow_prompt_speech_token_len = token_len;
        input.flow_prompt_speech_token.resize(token_len);
        LOG_INFO("handled for resample_rate 24000");
    }

    LOG_INFO("frontend_zero_shot done");

    return input;
}

std::pair<SpeechToken, int> CosyVoiceModelService::extract_speech_token(const std::vector<float>& pcm_data_16k) const {

    static constexpr int kMaxMelFrames = 3000; ///< Max mel frames (30s audio at 16kHz)
    static constexpr int kNMels = 128; ///< Number of mel bins
    static constexpr int kTokenRatio = 4; ///< Token/mel frame ratio (n_tokens = n_frames/4)
    static constexpr int kMaxTokens = 750; ///< Max tokens (3000/4)
    static constexpr int kMaskDim2 = 20; ///< mask shape dim 2
    static constexpr int kMask1Dim2 = 1280; ///< mask1 shape dim 2

    constexpr float kMaxDurationSec = 30.0f;

    // Float16 minimum value for attention mask
    static constexpr float kMaskMinValue = -65504.0f; ///< float16 min for invalid attention

    auto t_start = std::chrono::high_resolution_clock::now();

    // Check audio duration (max 30s)
    float duration = static_cast<float>(pcm_data_16k.size()) / HLIEAudio::kSampleRate16k;
    if (duration > kMaxDurationSec) {
        throw std::runtime_error("do not support extract speech token for audio longer than 30s");
    }

    // Compute 128-bin mel spectrogram

    MelFeatures128 mel = audio_processor_->ComputeMelSpectrogram128(pcm_data_16k);
    int feat_len = mel.n_frames;
    // last_feat_len_ = feat_len; // cache todo


    // Prepare input tensors
    // PrepareInput(mel, feat_len);

    // Prepare padded mel input [1, 128, 3000]
    std::vector padded_mel(kNMels * kMaxMelFrames, static_cast<TensorFP16>(0.0f));

    const auto& mel_data = mel.data;

    for (int t = 0; t < feat_len; t++) {
        for (int m = 0; m < kNMels; m++) {
            // Source index in mel_features.data
            int src_idx = m * feat_len + t;
            // Target index in padded_mel
            int dst_idx = m * kMaxMelFrames + t;
            if (src_idx < static_cast<int>(mel_data.size())) {
                padded_mel[dst_idx] = mel_data[src_idx];
            }
        }
    }


    // Calculate valid token length
    int valid_len = feat_len / kTokenRatio;

    // Create attention mask [1, 20, 750, 750]
    // Total elements: 1 * 20 * 750 * 750 = 11250000
    int total_elements = kMaskDim2 * kMaxTokens * kMaxTokens;
    std::vector<TensorFP16> mask(total_elements, static_cast<TensorFP16>(kMaskMinValue));

    for (int d2 = 0; d2 < kMaskDim2; d2++) {
        for (int row = 0; row < kMaxTokens; row++) {
            for (int col = 0; col < valid_len; col++) {
                // Index in flattened array
                // Layout assumption: [d2, row, col] -> d2 * 750 * 750 + row * 750 + col
                int idx = d2 * kMaxTokens * kMaxTokens + row * kMaxTokens + col;
                mask[idx] = static_cast<TensorFP16>(0.0f);
            }
        }
    }


    // Create encoder mask [1, 750, 1280]
    // Total elements: 1 * 750 * 1280 = 960000

    total_elements = kMaxTokens * kMask1Dim2;
    std::vector<TensorFP16> mask1(total_elements, static_cast<TensorFP16>(0.0f));

    // Set valid region to 1.0
    // Python: mask1[:, 0:feat_len//4, :] = 1.0
    // This means for rows < valid_len, all columns are 1.0

    for (int row = 0; row < valid_len; row++) {
        for (int col = 0; col < kMask1Dim2; col++) {
            // Index in flattened array
            // Layout: [row, col] -> row * 1280 + col
            int idx = row * kMask1Dim2 + col;
            mask1[idx] = static_cast<TensorFP16>(1.0f);
        }
    }


    // Step 6: Run speech tokenizer
    auto [fst, snd] = cosyvoice_model_->speech_tokenizer(padded_mel, mask, mask1, feat_len);
    LOG_INFO("Extracted speech token: shape=[1, {}], len={}", snd, snd);
    return {SpeechToken{std::move(fst), snd}, snd};
}

std::vector<float> CosyVoiceModelService::extract_spk_embedding(const std::vector<float>& pcm_data_16k) const {
    static constexpr int kFixedT = 1000; // Fixed time length for CampPlus

    FbankFeatures fbank = audio_processor_->ComputeFbank(pcm_data_16k);

    if (fbank.n_frames == 0 || fbank.data.empty()) {
        LOG_ERROR("Error: Empty FBANK features");
        return std::vector<float>();
    }

    // Step 2: Prepare features for model input
    // std::vector<TensorFP16> prepared_features = PrepareFeatures(fbank);

    // feat = feat - feat.mean(dim=0, keepdim=True)
    int n_frames = fbank.n_frames;
    int n_mels = fbank.n_mels;

    std::vector<float> features_transposed(n_frames * n_mels);
    std::vector<float> features_mean(n_mels, 0.0f);

    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < n_mels; j++) {
            features_mean[j] += fbank.data[i * n_mels + j];
        }
    }
    for (int j = 0; j < n_mels; j++) {
        features_mean[j] /= static_cast<float>(n_frames);
    }

    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < n_mels; j++) {
            features_transposed[i * n_mels + j] = fbank.data[i * n_mels + j] - features_mean[j];
        }
    }

    // Step 3: Pad or trim to fixed length (kFixedT = 1000)
    std::vector<float> features_fixed(kFixedT * n_mels,
                                      0.0f); // Initialize with zeros for padding

    if (n_frames < kFixedT) {
        // Copy all frames, rest will be zeros (padding)
        std::copy(features_transposed.begin(), features_transposed.end(), features_fixed.begin());
    } else {
        // Copy only first kFixedT frames (trim)
        std::copy(features_transposed.begin(), features_transposed.begin() + kFixedT * n_mels, features_fixed.begin());
    }


    // Step 4: Convert to float16
    std::vector<TensorFP16> prepared_features(kFixedT * n_mels);
    for (int t = 0; t < kFixedT; t++) {
        for (int m = 0; m < n_mels; m++) {
            prepared_features[t * n_mels + m] = static_cast<TensorFP16>(features_fixed[t * n_mels + m]);
        }
    }


    // Step 3: Run campplus model inference
    auto result = cosyvoice_model_->campplus(prepared_features);

    LOG_INFO("Extracted speaker embedding: size={}", result.size());
    return result;
}


HLIE_NAMESPACE_END
