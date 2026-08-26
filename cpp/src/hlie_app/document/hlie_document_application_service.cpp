#include "hlie_app/document/hlie_document_application_service.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "hlie_app/document/hlie_document_input_adapter.h"

HLIE_NAMESPACE_BEGIN

namespace {

bool json_boolish(const nlohmann::json& body, const char* key) {
    if (!body.is_object() || !body.contains(key)) {
        return false;
    }
    const auto& value = body.at(key);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_string()) {
        auto text = value.get<std::string>();
        std::transform(text.begin(), text.end(), text.begin(),
                       [](const unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        const auto first = std::find_if_not(
            text.begin(), text.end(), [](const unsigned char ch) {
                return std::isspace(ch);
            });
        const auto last = std::find_if_not(
            text.rbegin(), text.rend(), [](const unsigned char ch) {
                return std::isspace(ch);
            }).base();
        if (first >= last) {
            return false;
        }
        const std::string normalized(first, last);
        return normalized == "1" || normalized == "true" ||
            normalized == "yes";
    }
    return false;
}

std::optional<HLIEInferenceOperation> select_document_operation(
    const bool has_pdf, const nlohmann::json& body) {
    if (!has_pdf) {
        return std::nullopt;
    }
    return json_boolish(body, "document_preprocess_only")
        ? HLIEInferenceOperation::DocumentPreprocess
        : HLIEInferenceOperation::DocumentOcr;
}

} // namespace

HLIEDocumentApplicationService::HLIEDocumentApplicationService(
    HLIEDocumentApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

std::optional<HLIEInferenceOperation>
HLIEDocumentApplicationService::operation_for(
    const server_http_req& request, const nlohmann::json& body) const {
    const auto input = DocumentInputAdapter::inspect(request);
    return select_document_operation(input.has_pdf, body);
}

server_http_res_ptr HLIEDocumentApplicationService::dispatch_or_route(
    const std::string& model_key, const HLIEInferenceOperation operation,
    const server_http_req& request, const nlohmann::json& body,
    const std::string& method) const {
    const auto input = DocumentInputAdapter::inspect(request);
    const auto selected =
        select_document_operation(input.has_pdf, body).value_or(operation);

    if (input.has_pdf) {
        if (dependencies_.route_observer) {
            dependencies_.route_observer({
                model_key, input.source, input.filename, input.content_type});
        }
    }

    if (!dependencies_.dispatch) {
        auto response = std::make_unique<server_http_res>();
        response->status = 503;
        response->data =
            R"({"error":{"message":"document dispatch service is unavailable","type":"unavailable_error","code":80004}})";
        return response;
    }
    return dependencies_.dispatch(model_key, selected, request, method);
}

void HLIEDocumentApplicationService::stop_document_pipeline() const {
    if (dependencies_.stop_pipeline) {
        dependencies_.stop_pipeline();
    }
}

HLIE_NAMESPACE_END
