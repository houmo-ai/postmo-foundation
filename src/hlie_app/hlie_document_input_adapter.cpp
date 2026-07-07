#include "hlie_app/hlie_document_input_adapter.h"

#include <algorithm>
#include <cctype>

#include "base64.hpp"
#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_lower_copy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    bool has_pdf_content_type(const std::string& content_type) {
        return to_lower_copy(content_type).find("application/pdf") != std::string::npos;
    }

    bool has_pdf_filename(const std::string& filename) {
        const auto lower = to_lower_copy(filename);
        return lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".pdf") == 0;
    }

    std::string strip_data_url_prefix(const std::string& data) {
        const auto comma = data.find(',');
        if (comma == std::string::npos) {
            return data;
        }
        return data.substr(comma + 1);
    }

    bool data_url_is_pdf(const std::string& data) {
        if (data.rfind("data:", 0) != 0) {
            return false;
        }
        const auto comma = data.find(',');
        if (comma == std::string::npos) {
            return false;
        }
        return has_pdf_content_type(data.substr(0, comma));
    }

    raw_buffer string_to_raw_buffer(const std::string& data) {
        return raw_buffer(data.begin(), data.end());
    }

    bool inspect_openai_file_object(const nlohmann::json& item, DocumentInputInspection& result, const std::string& source) {
        const nlohmann::json* file_obj = nullptr;
        if (item.contains("file") && item["file"].is_object()) {
            file_obj = &item["file"];
        } else if (item.contains("input_file") && item["input_file"].is_object()) {
            file_obj = &item["input_file"];
        } else {
            file_obj = &item;
        }

        const std::string filename = file_obj->value("filename", std::string());
        const std::string content_type = file_obj->value("media_type", file_obj->value("content_type", std::string()));

        std::string encoded;
        if (file_obj->contains("file_data") && (*file_obj)["file_data"].is_string()) {
            encoded = (*file_obj)["file_data"].get<std::string>();
        } else if (item.contains("file_data") && item["file_data"].is_string()) {
            encoded = item["file_data"].get<std::string>();
        }

        bool is_pdf = has_pdf_content_type(content_type) || has_pdf_filename(filename) || data_url_is_pdf(encoded);
        if (!encoded.empty()) {
            try {
                const auto decoded = base64::decode(strip_data_url_prefix(encoded));
                is_pdf = is_pdf || DocumentInputAdapter::is_pdf_bytes(decoded);
            } catch (const std::exception& ex) {
                LOG_WARN("Failed to decode OpenAI file_data while sniffing document input: {}", ex.what());
            }
        }

        if (!is_pdf) {
            return false;
        }

        result.has_pdf = true;
        result.source = source;
        result.filename = filename;
        result.content_type = content_type;
        return true;
    }

    bool extract_openai_file_object(const nlohmann::json& item, DocumentInput& result, const std::string& source) {
        const nlohmann::json* file_obj = nullptr;
        if (item.contains("file") && item["file"].is_object()) {
            file_obj = &item["file"];
        } else if (item.contains("input_file") && item["input_file"].is_object()) {
            file_obj = &item["input_file"];
        } else {
            file_obj = &item;
        }

        const std::string filename = file_obj->value("filename", std::string());
        const std::string content_type = file_obj->value("media_type", file_obj->value("content_type", std::string()));

        std::string encoded;
        if (file_obj->contains("file_data") && (*file_obj)["file_data"].is_string()) {
            encoded = (*file_obj)["file_data"].get<std::string>();
        } else if (item.contains("file_data") && item["file_data"].is_string()) {
            encoded = item["file_data"].get<std::string>();
        }

        raw_buffer decoded;
        if (!encoded.empty()) {
            try {
                decoded = string_to_raw_buffer(base64::decode(strip_data_url_prefix(encoded)));
            } catch (const std::exception& ex) {
                LOG_WARN("Failed to decode OpenAI file_data while extracting document input: {}", ex.what());
            }
        }

        const bool is_pdf = has_pdf_content_type(content_type) || has_pdf_filename(filename) ||
                            data_url_is_pdf(encoded) || DocumentInputAdapter::is_pdf_bytes(decoded);
        if (!is_pdf || decoded.empty()) {
            return false;
        }

        result.has_pdf = true;
        result.source = source;
        result.filename = filename;
        result.content_type = content_type;
        result.data = std::move(decoded);
        return true;
    }

    bool inspect_content_array(const nlohmann::json& content, DocumentInputInspection& result, const std::string& source) {
        if (!content.is_array()) {
            return false;
        }

        for (const auto& item : content) {
            if (!item.is_object()) {
                continue;
            }

            const std::string type = item.value("type", std::string());
            if ((type == "file" || type == "input_file") && inspect_openai_file_object(item, result, source)) {
                return true;
            }
        }
        return false;
    }

    bool extract_content_array(const nlohmann::json& content, DocumentInput& result, const std::string& source) {
        if (!content.is_array()) {
            return false;
        }

        for (const auto& item : content) {
            if (!item.is_object()) {
                continue;
            }

            const std::string type = item.value("type", std::string());
            if ((type == "file" || type == "input_file") && extract_openai_file_object(item, result, source)) {
                return true;
            }
        }
        return false;
    }

    bool inspect_responses_input(const nlohmann::json& body, DocumentInputInspection& result) {
        if (!body.contains("input")) {
            return false;
        }

        const auto& input = body["input"];
        if (input.is_array()) {
            for (const auto& item : input) {
                if (!item.is_object()) {
                    continue;
                }
                if (inspect_content_array(item.value("content", nlohmann::json::array()), result, "responses.input")) {
                    return true;
                }
                const std::string type = item.value("type", std::string());
                if (type == "input_file" && inspect_openai_file_object(item, result, "responses.input")) {
                    return true;
                }
            }
        }

        return false;
    }

    bool extract_responses_input(const nlohmann::json& body, DocumentInput& result) {
        if (!body.contains("input")) {
            return false;
        }

        const auto& input = body["input"];
        if (input.is_array()) {
            for (const auto& item : input) {
                if (!item.is_object()) {
                    continue;
                }
                if (extract_content_array(item.value("content", nlohmann::json::array()), result, "responses.input")) {
                    return true;
                }
                const std::string type = item.value("type", std::string());
                if (type == "input_file" && extract_openai_file_object(item, result, "responses.input")) {
                    return true;
                }
            }
        }

        return false;
    }

    bool inspect_chat_messages(const nlohmann::json& body, DocumentInputInspection& result) {
        if (!body.contains("messages") || !body["messages"].is_array()) {
            return false;
        }

        for (const auto& message : body["messages"]) {
            if (!message.is_object() || !message.contains("content")) {
                continue;
            }
            if (inspect_content_array(message["content"], result, "chat.messages")) {
                return true;
            }
        }
        return false;
    }

    bool extract_chat_messages(const nlohmann::json& body, DocumentInput& result) {
        if (!body.contains("messages") || !body["messages"].is_array()) {
            return false;
        }

        for (const auto& message : body["messages"]) {
            if (!message.is_object() || !message.contains("content")) {
                continue;
            }
            if (extract_content_array(message["content"], result, "chat.messages")) {
                return true;
            }
        }
        return false;
    }
} // namespace

bool DocumentInputAdapter::is_pdf_bytes(const raw_buffer& data) {
    return data.size() >= 5 && data[0] == '%' && data[1] == 'P' && data[2] == 'D' && data[3] == 'F' && data[4] == '-';
}

bool DocumentInputAdapter::is_pdf_bytes(const std::string& data) {
    return data.size() >= 5 && data.compare(0, 5, "%PDF-") == 0;
}

DocumentInputInspection DocumentInputAdapter::inspect(const server_http_req& req) {
    DocumentInputInspection result;

    for (const auto& [field_name, file] : req.files) {
        if (has_pdf_content_type(file.content_type) || has_pdf_filename(file.filename) || is_pdf_bytes(file.data)) {
            result.has_pdf = true;
            result.source = "multipart:" + field_name;
            result.filename = file.filename;
            result.content_type = file.content_type;
            return result;
        }
    }

    if (req.body.empty()) {
        return result;
    }

    const auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        return result;
    }

    if (inspect_responses_input(body, result) || inspect_chat_messages(body, result)) {
        return result;
    }

    return result;
}

DocumentInput DocumentInputAdapter::extract_pdf(const server_http_req& req) {
    DocumentInput result;

    for (const auto& [field_name, file] : req.files) {
        if (has_pdf_content_type(file.content_type) || has_pdf_filename(file.filename) || is_pdf_bytes(file.data)) {
            result.has_pdf = true;
            result.source = "multipart:" + field_name;
            result.filename = file.filename;
            result.content_type = file.content_type;
            result.data = file.data;
            return result;
        }
    }

    if (req.body.empty()) {
        return result;
    }

    const auto body = nlohmann::json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        return result;
    }

    if (extract_responses_input(body, result) || extract_chat_messages(body, result)) {
        return result;
    }

    return result;
}

HLIE_NAMESPACE_END
