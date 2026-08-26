#include "hlie_base/hlie_gguf_util.h"

#include <cstdarg>
#include <gguf.h>
#include <sstream>

#include "hlie_base/hlie_logger.h"


namespace {
    std::string format(const char * fmt, ...) {
        va_list ap;
        va_list ap2;
        va_start(ap, fmt);
        va_copy(ap2, ap);
        int size = vsnprintf(NULL, 0, fmt, ap);
        GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
        std::vector<char> buf(size + 1);
        int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
        GGML_ASSERT(size2 == size);
        va_end(ap2);
        va_end(ap);
        return std::string(buf.data(), size);
    }
}


HLIE_NAMESPACE_BEGIN

namespace GGUFMeta {
    template<typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context *ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T>
    struct GKV_Base;

    template<>
    struct GKV_Base<bool> : GKV_Base_Type<bool, GGUF_TYPE_BOOL, gguf_get_val_bool> {
    };

    template<>
    struct GKV_Base<uint8_t> : GKV_Base_Type<uint8_t, GGUF_TYPE_UINT8, gguf_get_val_u8> {
    };

    template<>
    struct GKV_Base<uint16_t> : GKV_Base_Type<uint16_t, GGUF_TYPE_UINT16, gguf_get_val_u16> {
    };

    template<>
    struct GKV_Base<uint32_t> : GKV_Base_Type<uint32_t, GGUF_TYPE_UINT32, gguf_get_val_u32> {
    };

    template<>
    struct GKV_Base<uint64_t> : GKV_Base_Type<uint64_t, GGUF_TYPE_UINT64, gguf_get_val_u64> {
    };

    template<>
    struct GKV_Base<int8_t> : GKV_Base_Type<int8_t, GGUF_TYPE_INT8, gguf_get_val_i8> {
    };

    template<>
    struct GKV_Base<int16_t> : GKV_Base_Type<int16_t, GGUF_TYPE_INT16, gguf_get_val_i16> {
    };

    template<>
    struct GKV_Base<int32_t> : GKV_Base_Type<int32_t, GGUF_TYPE_INT32, gguf_get_val_i32> {
    };

    template<>
    struct GKV_Base<int64_t> : GKV_Base_Type<int64_t, GGUF_TYPE_INT64, gguf_get_val_i64> {
    };

    template<>
    struct GKV_Base<float> : GKV_Base_Type<float, GGUF_TYPE_FLOAT32, gguf_get_val_f32> {
    };

    template<>
    struct GKV_Base<double> : GKV_Base_Type<double, GGUF_TYPE_FLOAT64, gguf_get_val_f64> {
    };

    template<>
    struct GKV_Base<const char *> : GKV_Base_Type<const char *, GGUF_TYPE_STRING, gguf_get_val_str> {
    };

    template<>
    struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context *ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void *data;
    };

    template<>
    struct GKV_Base<ArrayInfo> {
    public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;

        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo{
                arr_type,
                size_t(gguf_get_arr_n(ctx, k)),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

    public:
        static T get_kv(const gguf_context *ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);
            if (kt != GKV::gt) {
                printf("key %s has wrong type %s but expected type %s", gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt));
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                                                gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static bool set(const gguf_context *ctx, const int k, T &target) {
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context *ctx, const char *key, T &target) {
            return set(ctx, gguf_find_key(ctx, key), target);
        }

        static bool set(const gguf_context *ctx, const std::string &key, T &target) {
            return set(ctx, key.c_str(), target);
        }
    };
}


template <typename T>
bool HLIEGgufUtil::get_key(gguf_context* meta, const std::string& key, T& result, bool required) {
    const bool found = GGUFMeta::GKV<T>::set(meta, key, result);

    if (required && !found) {
        throw std::runtime_error(format("key not found in model: %s", key.c_str()));
    }

    return found;
}

// Explicit template instantiations
template bool HLIEGgufUtil::get_key<int>(gguf_context*, const std::string&, int&, bool);
template bool HLIEGgufUtil::get_key<uint32_t>(gguf_context*, const std::string&, uint32_t&, bool);
template bool HLIEGgufUtil::get_key<int64_t>(gguf_context*, const std::string&, int64_t&, bool);
template bool HLIEGgufUtil::get_key<float>(gguf_context*, const std::string&, float&, bool);
template bool HLIEGgufUtil::get_key<double>(gguf_context*, const std::string&, double&, bool);
template bool HLIEGgufUtil::get_key<bool>(gguf_context*, const std::string&, bool&, bool);
template bool HLIEGgufUtil::get_key<std::string>(gguf_context*, const std::string&, std::string&, bool);


void HLIEGgufUtil::replace_all(std::string &s, const std::string &search, const std::string &replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string HLIEGgufUtil::gguf_data_to_str(enum gguf_type type, const void *data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8: return std::to_string(((const uint8_t *) data)[i]);
        case GGUF_TYPE_INT8: return std::to_string(((const int8_t *) data)[i]);
        case GGUF_TYPE_UINT16: return std::to_string(((const uint16_t *) data)[i]);
        case GGUF_TYPE_INT16: return std::to_string(((const int16_t *) data)[i]);
        case GGUF_TYPE_UINT32: return std::to_string(((const uint32_t *) data)[i]);
        case GGUF_TYPE_INT32: return std::to_string(((const int32_t *) data)[i]);
        case GGUF_TYPE_UINT64: return std::to_string(((const uint64_t *) data)[i]);
        case GGUF_TYPE_INT64: return std::to_string(((const int64_t *) data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float *) data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double *) data)[i]);
        case GGUF_TYPE_BOOL: return ((const bool *) data)[i] ? "true" : "false";
        default: return format("unknown type %d", type);
    }
}

std::string HLIEGgufUtil::gguf_kv_to_str(const struct gguf_context *ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY: {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
            int arr_n = gguf_get_arr_n(ctx_gguf, i);
            const void *data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
            std::stringstream ss;
            ss << "[";
            for (int j = 0; j < arr_n; j++) {
                if (arr_type == GGUF_TYPE_STRING) {
                    std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                    // escape quotes
                    replace_all(val, "\\", "\\\\");
                    replace_all(val, "\"", "\\\"");
                    ss << '"' << val << '"';
                } else if (arr_type == GGUF_TYPE_ARRAY) {
                    ss << "???";
                } else {
                    ss << gguf_data_to_str(arr_type, data, j);
                }
                if (j < arr_n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
            return ss.str();
        }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

HLIE_NAMESPACE_END
