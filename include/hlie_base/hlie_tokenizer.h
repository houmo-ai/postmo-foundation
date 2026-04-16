#ifndef HLIE_TOKENIZER_H
#define HLIE_TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_base/hlie_base.h"

struct gguf_context;

HLIE_NAMESPACE_BEGIN

// =========================================================================
// HLIE BPE Tokenizer
// =========================================================================

/**
 * BPE Tokenizer.
 * Based on GPT-2 style byte-level BPE tokenization.
 */
class HLIE_API HLIETokenizer {
public:
    HLIETokenizer() = default;
    ~HLIETokenizer() = default;

    /**
     * Load tokenizer from directory containing:
     * - vocab.json: Vocabulary mapping (token -> id)
     */
    bool load(const std::string& tokenizer_dir);

    /**
     * Load tokenizer from GGUF file using offset and size.
     * @param gguf_path Path to GGUF file
     * @param vocab_offset Offset of vocab.json data in GGUF file
     * @param vocab_size Size of vocab.json data
     */
    bool load_from_gguf(const std::string& gguf_path, size_t vocab_offset, size_t vocab_size);

    bool load_from_gguf_ctx(gguf_context* ctx);

    /**
     * Encode text to token IDs.
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text);

    /**
     * Decode token IDs to text.
     * @param tokens Vector of token IDs
     * @return Decoded text string
     */
    std::string decode(const std::vector<int>& tokens);

    void set_skip_special_tokens(bool skip);

    [[nodiscard]] size_t vocab_size() const;

private:
    std::unordered_map<std::string, int> token_to_id_; // token -> id
    std::unordered_map<int, std::string> id_to_token_; // id -> token (reverse lookup)
    bool skip_special_tokens_ = true;

    /**
     * Load vocabulary from GGUF file at specific offset and size.
     */
    bool load_vocab_from_gguf(const std::string& path, size_t offset, size_t size);

    /**
     * Parse vocabulary content (shared by file and GGUF loading).
     */
    bool parse_vocab_content(const std::string& content);


    /**
     * Normalize text (simplified version)
     * TODO: Match exact normalization from Python tokenizer
     */
    static std::string normalize_text(const std::string& text);

    /**
     * Split text into word tokens using regex.
     * Similar to GPT-2 regex pattern.
     *
     * Pattern matches:
     * - 's, 't, 're, 've, 'm, 'll, 'd (contractions)
     * - Words (alphabetic)
     * - Numbers
     * - Symbols
     * - Whitespace
     */
    static std::vector<std::string> split_into_words(const std::string& text);

    /**
     * Convert word tokens to token IDs using BPE.
     * Uses greedy longest-match algorithm.
     */
    std::vector<int> words_to_ids(const std::vector<std::string>& words);

    /**
     * Check if token ID is a special token
     */
    bool is_special_token(int token_id);

    /**
     * Load vocabulary from vocab.json
     */
    bool load_vocab(const std::string& path);

    /**
     * Decode Unicode escape sequences in JSON (\uXXXX)
     */
    static std::string decode_unicode_escapes(const std::string& text);

    /**
     * Convert Unicode codepoint to UTF-8 bytes
     */
    static std::string codepoint_to_utf8(int codepoint);
};

HLIE_NAMESPACE_END

#endif
