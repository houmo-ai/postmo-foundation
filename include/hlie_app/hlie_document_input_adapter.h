#ifndef HLIE_DOCUMENT_INPUT_ADAPTER_H
#define HLIE_DOCUMENT_INPUT_ADAPTER_H

#include <string>
#include <utility>

#include <server/server-http.h>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct DocumentInputInspection {
    bool has_pdf = false;
    std::string source;
    std::string filename;
    std::string content_type;
};

struct DocumentInput {
    bool has_pdf = false;
    std::string source;
    std::string filename;
    std::string content_type;
    raw_buffer data;
};

class DocumentInputAdapter {
public:
    static DocumentInputInspection inspect(const server_http_req& req);
    static DocumentInput extract_pdf(const server_http_req& req);
    static bool is_pdf_bytes(const raw_buffer& data);
    static bool is_pdf_bytes(const std::string& data);
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_INPUT_ADAPTER_H
