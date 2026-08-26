#ifndef HLIE_TOKENIZER_H
#define HLIE_TOKENIZER_H

#include <forward_list>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "hlie_base/hlie_base.h"

struct gguf_context;

#define TOKEN_NULL (-1)

typedef int32_t hlie_token;

//
// enum llama_token_attr {
//     TOKEN_ATTR_UNDEFINED = 0,
//     TOKEN_ATTR_UNKNOWN = 1 << 0,
//     TOKEN_ATTR_UNUSED = 1 << 1,
//     TOKEN_ATTR_NORMAL = 1 << 2,
//     TOKEN_ATTR_CONTROL = 1 << 3, // SPECIAL?
//     TOKEN_ATTR_USER_DEFINED = 1 << 4,
//     TOKEN_ATTR_BYTE = 1 << 5,
//     TOKEN_ATTR_NORMALIZED = 1 << 6,
//     TOKEN_ATTR_LSTRIP = 1 << 7,
//     TOKEN_ATTR_RSTRIP = 1 << 8,
//     TOKEN_ATTR_SINGLE_WORD = 1 << 9,
// };

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(hlie_token _token) :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN), token(_token), raw_text(_dummy), offset(0), length(0) {
    }

    fragment_buffer_variant(const std::string& _raw_text, int64_t _offset, int64_t _length) :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT), token((hlie_token)-1), raw_text(_raw_text), offset(_offset), length(_length) {
        GGML_ASSERT(_offset >= 0);
        GGML_ASSERT(_length >= 1);
        GGML_ASSERT(offset + length <= raw_text.length());
    }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const hlie_token token;
    const std::string _dummy;
    const std::string& raw_text;
    const uint64_t offset;
    const uint64_t length;
};


struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char* text;
    size_t n;
};

template <typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() = delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe& l, const llm_bigram_bpe& r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left{};
    llm_symbol::index right{};
    std::string text;
    int rank{};
    size_t size{};
};

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

    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant>& buffer) const;
    [[nodiscard]] const std::string& get_token_data(int32_t id) const;
    [[nodiscard]] int32_t text_to_token(const std::string& text) const;
    void add_new_bigram(int left, int right);
    [[nodiscard]] int find_bpe_rank(const std::string& token_left, const std::string& token_right) const;

    [[nodiscard]] std::vector<hlie_token> tokenize(const std::string & raw_text);

    bool load_from_gguf_ctx(gguf_context* ctx);

    /**
     * Register tokens that must bypass byte-level BPE.
     * IDs are assigned consecutively from first_token_id.
     * Existing token/ID mappings must match exactly; conflicts are rejected.
     */
    bool add_special_tokens(const std::string_view* tokens, size_t count,
                            hlie_token first_token_id);

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

    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^ // create some hash for pair
                (std::hash<std::string>{}(p.second) << 1);
        }
    };

private:
    std::unordered_map<std::string, int> token_to_id_; // token -> id
    std::unordered_map<int, std::string> id_to_token_; // id -> token (reverse lookup)
    // Special tokens must bypass byte-level BPE. This list is populated from
    // the GGUF vocabulary and sorted longest-first to handle overlapping names.
    std::vector<hlie_token> special_token_ids_;
    bool skip_special_tokens_ = true;


    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

    void tokenize_i(const std::string& text, std::vector<int32_t>& output);

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
    void sort_special_tokens();

    /**
     * Decode Unicode escape sequences in JSON (\uXXXX)
     */
    static std::string decode_unicode_escapes(const std::string& text);

    /**
     * Convert Unicode codepoint to UTF-8 bytes
     */
    static std::string codepoint_to_utf8(int codepoint);

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

HLIE_NAMESPACE_END

#endif
