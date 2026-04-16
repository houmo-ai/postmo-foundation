#include "hlie_base/hlie_tokenizer.h"

#include <fstream>
#include <gguf.h>
#include <regex>
#include <sstream>

#include "hlie_base/hlie_gguf_util.h"
#include "hlie_base/hlie_logger.h"


HLIE_NAMESPACE_BEGIN

bool HLIETokenizer::load(const std::string& tokenizer_dir) {
    std::string vocab_path = tokenizer_dir + "/vocab.json";

    if (!load_vocab(vocab_path)) {
        LOG_ERROR("Failed to load vocab.json from: {}", vocab_path);
        return false;
    }

    LOG_INFO("Tokenizer loaded: {} tokens", token_to_id_.size());
    return true;
}

bool HLIETokenizer::load_from_gguf(const std::string& gguf_path, size_t vocab_offset, size_t vocab_size) {
    if (!load_vocab_from_gguf(gguf_path, vocab_offset, vocab_size)) {
        LOG_ERROR("Failed to load vocab from GGUF at offset {}, size {}", vocab_offset, vocab_size);
        return false;
    }

    LOG_INFO("CosyVoice3Tokenizer loaded from GGUF: {} tokens", token_to_id_.size());
    return true;
}

bool HLIETokenizer::load_from_gguf_ctx(gguf_context* ctx) {
    int32_t n_vocab = 0;
    HLIEGgufUtil::get_key(ctx, "vocab_size", n_vocab);
    if (0 == n_vocab) {
        LOG_ERROR("%s: n_vocab is 0", __func__);
        return false;
    }

    const auto token_idx = gguf_find_key(ctx, "vocab");
    if (token_idx == -1) {
        LOG_ERROR("cannot find tokenizer vocab in model file");
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();
    const uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);
    for (int32_t i = 0; i < static_cast<int32_t>(n_tokens); i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        if (word.empty()) {
            LOG_WARN("%s: empty token at index %u\n", __func__, i);
        }

        token_to_id_[word] = i;
        id_to_token_[i] = word;
    }

    LOG_INFO("Tokenizer loaded, size: {}", token_to_id_.size());

    return true;
}

std::vector<int> HLIETokenizer::encode(const std::string& text) {
    if (text.empty()) {
        LOG_ERROR("empty text when encode");
        return {};
    }

    LOG_INFO("Ready to encode the text: {}", text);

    // Step 1: Normalize text (similar to GPT-2)
    std::string normalized = normalize_text(text);
    LOG_INFO("normalized text: {}", normalized);

    // Step 2: Split into word tokens using regex
    std::vector<std::string> words = split_into_words(normalized);
    LOG_INFO("split_into_words words: {}", words.size());
    for (auto word : words) {
        LOG_INFO("of word: {}", word);
    }

    // Step 3: Convert words to token IDs using BPE
    return words_to_ids(words);
}

std::string HLIETokenizer::decode(const std::vector<int>& tokens) {
    if (tokens.empty())
        return "";

    std::string result;
    for (int token_id : tokens) {
        if (skip_special_tokens_ && is_special_token(token_id)) {
            continue;
        }

        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            result += it->second;
        }
    }

    // Post-process: remove leading space for certain tokens
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }

    return result;
}

void HLIETokenizer::set_skip_special_tokens(bool skip) {
    skip_special_tokens_ = skip;
}

size_t HLIETokenizer::vocab_size() const {
    return token_to_id_.size();
}


bool HLIETokenizer::load_vocab_from_gguf(const std::string& path, size_t offset, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open GGUF file: {}", path);
        return false;
    }

    // Seek to vocab offset
    file.seekg(offset, std::ios::beg);
    if (!file.good()) {
        LOG_ERROR("Failed to seek to vocab offset {}", offset);
        return false;
    }

    // Read vocab data
    std::string content(size, '\0');
    file.read(&content[0], size);
    if (!file.good()) {
        LOG_ERROR("Failed to read vocab data, expected {} bytes", size);
        return false;
    }

    // Parse vocabulary
    return parse_vocab_content(content);
}

bool HLIETokenizer::parse_vocab_content(const std::string& content) {
    LOG_INFO("Parsing vocab content, length: {}", content.size());

    // Parse JSON - format: {"token": id, ...}
    std::string pattern_str = "\"([^\"]+)\"";
    std::regex pattern(pattern_str);
    std::smatch match;
    std::string::const_iterator search_start(content.cbegin());

    int count = 0;
    while (std::regex_search(search_start, content.cend(), match, pattern)) {
        std::string token = match[1].str();

        // Handle Unicode escape sequences
        token = decode_unicode_escapes(token);

        int id;
        // Try to get the number after the closing quote
        std::string remainder = match.suffix().str();
        if (!remainder.empty() && remainder[0] == ':') {
            size_t num_start = 1;
            while (num_start < remainder.size() && (remainder[num_start] == ' ' || remainder[num_start] == '\t')) {
                num_start++;
            }
            size_t num_end = num_start;
            while (num_end < remainder.size() && std::isdigit(remainder[num_end])) {
                num_end++;
            }
            if (num_end > num_start) {
                id = std::stoi(remainder.substr(num_start, num_end - num_start));
            } else {
                search_start = match.suffix().first;
                continue;
            }
        } else {
            search_start = match.suffix().first;
            continue;
        }

        token_to_id_[token] = id;
        id_to_token_[id] = token;

        count++;
        if (count <= 3) {
            LOG_INFO("Loaded token: '{}' (len={}) -> {}", token, token.size(), id);
        }

        search_start = match.suffix().first;
    }

    LOG_INFO("Parsed {} tokens from vocab", count);
    return count > 0;
}

std::string HLIETokenizer::normalize_text(const std::string& text) {
    std::string result = text;

    // Replace multiple spaces with single space
    result = std::regex_replace(result, std::regex(R"(\s+)"), " ");

    // TODO: Add more normalization as needed
    // - NFC normalization
    // - Strip leading/trailing whitespace

    return result;
}

std::vector<std::string> HLIETokenizer::split_into_words(const std::string& text) {
    std::vector<std::string> words;

    if (text.empty()) {
        return words;
    }

    // 检测是否包含中文
    bool has_chinese = false;
    for (unsigned char c : text) {
        if ((c & 0xF0) == 0xE0) {  // UTF-8 中文范围
            has_chinese = true;
            break;
        }
    }

    LOG_INFO("has_chinese: {}", has_chinese);

    if (has_chinese) {
        // 中文分词：按字符分割
        size_t i = 0;
        size_t n = text.size();

        while (i < n) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // 跳过空白
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++i;
                continue;
            }

            // 计算字符长度
            size_t char_len = 1;
            if ((c & 0xF0) == 0xE0) {
                char_len = 3;  // 中文 3字节
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;  // 2字节字符
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;  // emoji 4字节
            }

            if (i + char_len <= n) {
                words.push_back(text.substr(i, char_len));
            }
            i += char_len;
        }

        LOG_DEBUG("Chinese split result: {} words", words.size());
        for (size_t i = 0; i < std::min(words.size(), size_t(5)); ++i) {
            LOG_DEBUG("  word[{}]: '{}'", i, words[i]);
        }
    } else {
        // 英文：使用原有的正则分词
        std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        try {
            std::regex re(pattern);
            std::smatch m;
            std::string str = text;

            while (std::regex_search(str, m, re)) {
                for (auto& x : m) {
                    std::string word = x.str();
                    if (!word.empty() && word != " ") {
                        words.push_back(word);
                    }
                }
                str = m.suffix();
            }
        } catch (const std::regex_error& e) {
            LOG_ERROR("Regex error: {}", e.what());
        }
    }

    return words;
}

std::vector<int> HLIETokenizer::words_to_ids(const std::vector<std::string>& words) {
    std::vector<int> tokens;
    LOG_INFO("=== words_to_ids START ===");
    LOG_INFO("Input words count: {}", words.size());

    // 统计未知字符
    std::map<std::string, int> unknown_counts;
    int unknown_total = 0;
    for (size_t w = 0; w < words.size(); ++w) {
        const auto& word = words[w];

        if (word.empty())
            continue;

        LOG_DEBUG("Processing word[{}]: '{}' (len={})", w, word, word.size());

        size_t i = 0;
        size_t n = word.size();

        while (i < n) {
            size_t j = n;
            bool found = false;

            // Find longest matching token (from end to start)
            while (j > i) {
                std::string sub = word.substr(i, j - i);

                LOG_TRACE("  Trying sub='{}' (pos={}, len={})", sub, i, sub.size());

                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    LOG_TRACE("    FOUND! token_id={}", it->second);
                    tokens.push_back(it->second);
                    i = j;
                    found = true;
                    break;
                }
                --j;
            }

            if (!found) {
                // Unknown token
                std::string unknown = word.substr(i, 1);
                unsigned char c = static_cast<unsigned char>(word[i]);

                // 统计未知字符
                unknown_counts[unknown]++;
                unknown_total++;

                LOG_WARN("Unknown token[{}] '{}' (hex={:02x}, byte_len={})",
                         unknown_total, unknown, static_cast<unsigned int>(c),
                         (c & 0xF0) == 0xE0 ? 3 : (c & 0xE0) == 0xC0 ? 2 : 1);

                ++i;
            }
        }

        LOG_DEBUG("  Word[{}] done, tokens so far: {}", w, tokens.size());
    }

    // 打印未知字符统计
    LOG_INFO("=== Unknown Token Summary ===");
    LOG_INFO("Total unknown tokens: {}", unknown_total);
    for (const auto& [char_str, count] : unknown_counts) {
        unsigned char c = static_cast<unsigned char>(char_str[0]);
        LOG_INFO("  '{}' (hex={:02x}): {} times",
                 char_str, static_cast<unsigned int>(c), count);
    }

    LOG_INFO("=== words_to_ids END, total tokens: {} ===", tokens.size());

    // 打印前20个 token
    if (!tokens.empty()) {
        std::string token_str;
        for (size_t i = 0; i < std::min(tokens.size(), size_t(20)); ++i) {
            token_str += std::to_string(tokens[i]) + " ";
        }
        LOG_INFO("First 20 tokens: {}", token_str);
    }

    return tokens;
}

bool HLIETokenizer::is_special_token(int token_id) {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end())
        return false;

    const std::string& token = it->second;
    // Special tokens usually start with <| or end with |>
    return token.find("<|") != std::string::npos || token.find("|>") != std::string::npos;
}


bool HLIETokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open vocab file: {}", path);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Parse JSON - format: {"token": id, ...}
    std::string pattern_str = "\"([^\"]+)\":\\s*(\\d+)";
    std::regex pattern(pattern_str);
    std::smatch match;
    std::string::const_iterator search_start(content.cbegin());

    while (std::regex_search(search_start, content.cend(), match, pattern)) {
        std::string token = match[1];
        int id = std::stoi(match[2]);

        // Handle Unicode escape sequences
        token = decode_unicode_escapes(token);

        token_to_id_[token] = id;
        id_to_token_[id] = token;

        search_start = match.suffix().first;
    }

    LOG_INFO("Loaded {} tokens from vocabulary", token_to_id_.size());
    return !token_to_id_.empty();
}

std::string HLIETokenizer::decode_unicode_escapes(const std::string& text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            if (text[i + 1] == 'u' && i + 5 < text.size()) {
                std::string hex = text.substr(i + 2, 4);
                int codepoint = std::stoi(hex, nullptr, 16);
                result += codepoint_to_utf8(codepoint);
                i += 6;
            } else {
                result += text[i + 1];
                i += 2;
            }
        } else {
            result += text[i];
            ++i;
        }
    }
    return result;
}

/**
 * Convert Unicode codepoint to UTF-8 bytes
 */
std::string HLIETokenizer::codepoint_to_utf8(int codepoint) {
    std::string result;
    if (codepoint < 0x80) {
        result += static_cast<char>(codepoint);
    } else if (codepoint < 0x800) {
        result += static_cast<char>(0xC0 | (codepoint >> 6));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
        result += static_cast<char>(0xE0 | (codepoint >> 12));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (codepoint >> 18));
        result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    return result;
}

HLIE_NAMESPACE_END
