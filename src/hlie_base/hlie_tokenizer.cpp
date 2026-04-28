#include "hlie_base/hlie_tokenizer.h"

#include <fstream>
#include <gguf.h>
#include <queue>
#include <regex>

#include "hlie_base/hlie_gguf_util.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/unicode.h"


namespace {
    std::vector<hlie_token> cache_special_tokens{151646};
    void append(const hlie_token token_id, std::vector<hlie_token>& output) {
        output.push_back(token_id);
    }

    std::vector<std::string> regex_exprs = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
        "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    bool append_bos(std::vector<hlie_token>& output) {
        return false;
    }

    bool append_eos(std::vector<hlie_token>& output) {
        return false;
    }

    void check_double_bos_eos(const std::vector<hlie_token>& output) {
    }
} // namespace

HLIE_NAMESPACE_BEGIN

void HLIETokenizer::tokenizer_st_partition(std::forward_list<fragment_buffer_variant>& buffer) const {
    // for each special token
    for (const hlie_token special_id : cache_special_tokens) {
        const auto& text = std::string("<|endofprompt|>");// get_token_data(special_id);

        // for each text fragment
        auto it = buffer.begin();
        while (it != buffer.end()) {
            auto& fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto& raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    //  string_view begins at pos 0 for the same reason
                    auto match =
                        std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos)
                        break;


                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        // if (data.attr & TOKEN_ATTR_LSTRIP) {
                        //     while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                        //         left_reminder_length--;
                        //     }
                        // }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        // if (data.attr & TOKEN_ATTR_RSTRIP) {
                        //     while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                        //         right_reminder_offset++;
                        //         right_reminder_length--;
                        //     }
                        // }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

const std::string& HLIETokenizer::get_token_data(const hlie_token id) const {
    return id_to_token_.at(id);
}

std::vector<hlie_token> HLIETokenizer::tokenize(const std::string& raw_text) {

    std::vector<hlie_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer);
    }

    for (const auto& fragment : fragment_buffer) {
        if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
            std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);
            tokenize_i(text, output);
        } else {
            append(fragment.token, output);
            LOG_WARN("Add specail token {}", fragment.token);
        }
    }

    return output;
}

void HLIETokenizer::tokenize_i(const std::string& text, std::vector<hlie_token>& output) {
    int final_prev_index = -1;
    const auto word_collection = unicode_regex_split(text, regex_exprs);

    symbols_final.clear();

    for (const auto& word : word_collection) {
        work_queue = llm_bigram_bpe::queue();
        symbols.clear();

        LOG_TRACE("tokenize_i process word ={}",word);

        int index = 0;
        size_t offset = 0;

        while (offset < word.size()) {
            llm_symbol sym;
            size_t char_len = std::min(word.size() - offset, (size_t)unicode_len_utf8(word[offset]));
            sym.text = word.c_str() + offset;
            sym.n = char_len;
            offset += sym.n;
            sym.prev = index - 1;
            sym.next = offset == word.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }
        for (int i = 1; i < (int)symbols.size(); ++i) {
            add_new_bigram(i - 1, i);
        }

        // build token(s)
        while (!work_queue.empty()) {
            auto bigram = work_queue.pop_move();

            auto& left_symbol = symbols[bigram.left];
            auto& right_symbol = symbols[bigram.right];

            if (left_symbol.n == 0 || right_symbol.n == 0) {
                continue;
            }
            std::string left_token = std::string(left_symbol.text, left_symbol.n);
            std::string right_token = std::string(right_symbol.text, right_symbol.n);
            if (left_token + right_token != bigram.text) {
                continue; // Skip this bigram if it's outdated
            }

            // merge the right sym into the left one
            left_symbol.n += right_symbol.n;
            right_symbol.n = 0;

            // remove the right sym from the chain
            left_symbol.next = right_symbol.next;
            if (right_symbol.next >= 0) {
                symbols[right_symbol.next].prev = bigram.left;
            }

            add_new_bigram(left_symbol.prev, bigram.left); // left side of current symbol
            add_new_bigram(bigram.left, left_symbol.next); // right side of current symbol
        }

        // add the finished tokens to the final list keeping correct order for next and prev
        for (auto& sym : symbols) {
            if (sym.n > 0) {
                sym.prev = final_prev_index;
                sym.next = -1;
                if (final_prev_index != -1) {
                    symbols_final[final_prev_index].next = symbols_final.size();
                }
                symbols_final.emplace_back(sym);
                final_prev_index = symbols_final.size() - 1;
            }
        }
    }

    symbols = symbols_final;

    if (!symbols.empty()) {
        for (int i = 0; i != -1; i = symbols[i].next) {
            auto& symbol = symbols[i];
            if (symbol.n == 0) {
                continue;
            }

            const std::string str = std::string(symbol.text, symbol.n);
            const auto token = text_to_token(str);

            if (token == TOKEN_NULL) {
                for (auto j = str.begin(); j != str.end(); ++j) {
                    std::string byte_str(1, *j);
                    auto token_multibyte = text_to_token(byte_str);
                    if (token_multibyte != TOKEN_NULL) {
                        LOG_TRACE("tokenize_i push_back token_multibyte ={}",token_multibyte);
                        output.push_back(token_multibyte);
                    }
                }
            } else {
                LOG_TRACE("tokenize_i push_back token ={}",token);
                output.push_back(token);
            }
        }
    }
}

hlie_token HLIETokenizer::text_to_token(const std::string& text) const {
    auto it = token_to_id_.find(text);
    if (it != token_to_id_.end()) {
        return (*it).second;
    }

    return TOKEN_NULL;
}

void HLIETokenizer::add_new_bigram(int left, int right) {
    if (left == -1 || right == -1) {
        return;
    }
    std::string left_token = std::string(symbols[left].text, symbols[left].n);
    std::string right_token = std::string(symbols[right].text, symbols[right].n);

    int rank_found = -1;

    rank_found = find_bpe_rank(left_token, right_token);

    if (rank_found < 0) {
        return;
    }

    llm_bigram_bpe bigram;

    bigram.left = left;
    bigram.right = right;
    bigram.text = left_token + right_token;
    bigram.size = left_token.size() + right_token.size();
    bigram.rank = rank_found;

    work_queue.push(bigram);
}

int HLIETokenizer::find_bpe_rank(const std::string& token_left, const std::string& token_right) const {
    GGML_ASSERT(token_left.find(' ') == std::string::npos);
    GGML_ASSERT(token_left.find('\n') == std::string::npos);
    GGML_ASSERT(token_right.find(' ') == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    const auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
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

    const auto merges_idx = gguf_find_key(ctx, "merges");
    if (merges_idx == -1) {
        LOG_ERROR("cannot find tokenizer merges in model file");
        return false;
    }

    const auto n_merges = gguf_get_arr_n(ctx, merges_idx);
    for (int i = 0; i < n_merges; i++) {
        const std::string word = gguf_get_arr_str(ctx, merges_idx, i);

        std::string first;
        std::string second;

        const size_t pos = word.find(' ', 1);

        if (pos != std::string::npos) {
            first = word.substr(0, pos);
            second = word.substr(pos + 1);
        }

        bpe_ranks.emplace(std::make_pair(first, second), i);
    }

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

    /*
    // Step 2: Split into word tokens using regex
    std::vector<std::string> words = split_into_words(normalized);
    LOG_INFO("split_into_words words: {}", words.size());
    for (auto word : words) {
        LOG_INFO("of word: {}", word);
    }*/

    // Step 3: Convert words to token IDs using BPE
    return tokenize(normalized);
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
        if ((c & 0xF0) == 0xE0) { // UTF-8 中文范围
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
                char_len = 3; // 中文 3字节
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2; // 2字节字符
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4; // emoji 4字节
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

                LOG_WARN("Unknown token[{}] '{}' (hex={:02x}, byte_len={})", unknown_total, unknown, static_cast<unsigned int>(c),
                         (c & 0xF0) == 0xE0       ? 3
                             : (c & 0xE0) == 0xC0 ? 2
                                                  : 1);

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
        LOG_INFO("  '{}' (hex={:02x}): {} times", char_str, static_cast<unsigned int>(c), count);
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
