#include <catch_amalgamated.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "hlie_app/document/hlie_document_application_service.h"

namespace {

using namespace hm_HLIE;

server_http_req make_request(
    const std::function<bool()>& should_stop,
    std::string body = "{}",
    std::map<std::string, uploaded_file> files = {}) {
    return {{}, {}, "/v1/responses", {}, std::move(body),
            std::move(files), should_stop};
}

} // namespace

TEST_CASE("Document application service selects document operations",
          "[p1][document][application-service]") {
    const std::function<bool()> should_stop = [] { return false; };
    HLIEDocumentApplicationService service({});

    auto normal = make_request(should_stop);
    CHECK_FALSE(service.operation_for(normal, nlohmann::json::object()));

    uploaded_file pdf;
    pdf.data = {'%', 'P', 'D', 'F', '-', '1', '.', '7'};
    pdf.filename = "sample.pdf";
    pdf.content_type = "application/pdf";
    auto document = make_request(
        should_stop, "{}", {{"file", std::move(pdf)}});

    const auto ocr = service.operation_for(
        document, nlohmann::json::object());
    REQUIRE(ocr);
    CHECK(*ocr == HLIEInferenceOperation::DocumentOcr);

    const auto preprocess = service.operation_for(
        document, {{"document_preprocess_only", "true"}});
    REQUIRE(preprocess);
    CHECK(*preprocess == HLIEInferenceOperation::DocumentPreprocess);
}

TEST_CASE("Document application service delegates dispatch and lifecycle",
          "[p1][document][application-service]") {
    const std::function<bool()> should_stop = [] { return false; };
    HLIEInferenceOperation observed_operation =
        HLIEInferenceOperation::Completions;
    std::string observed_model;
    std::string observed_method;
    int stop_count = 0;
    int route_observation_count = 0;

    HLIEDocumentApplicationService service({
        [&](const std::string& model,
            const HLIEInferenceOperation operation,
            const server_http_req&, const std::string& method) {
            observed_model = model;
            observed_operation = operation;
            observed_method = method;
            auto response = std::make_unique<server_http_res>();
            response->status = 202;
            return response;
        },
        [&] { ++stop_count; },
        [&](const HLIEDocumentRouteObservation&) {
            ++route_observation_count;
        },
    });

    uploaded_file pdf;
    pdf.data = {'%', 'P', 'D', 'F', '-', '1', '.', '7'};
    auto request = make_request(
        should_stop, "{}", {{"upload", std::move(pdf)}});
    const auto response = service.dispatch_or_route(
        "model-key", HLIEInferenceOperation::Responses, request,
        {{"document_preprocess_only", "true"}}, "POST");

    REQUIRE(response);
    CHECK(response->status == 202);
    CHECK(observed_model == "model-key");
    CHECK(observed_operation == HLIEInferenceOperation::DocumentPreprocess);
    CHECK(observed_method == "POST");
    CHECK(route_observation_count == 1);

    auto normal_request = make_request(should_stop);
    const auto normal_response = service.dispatch_or_route(
        "model-key", HLIEInferenceOperation::Responses, normal_request,
        nlohmann::json::object(), "POST");
    REQUIRE(normal_response);
    CHECK(observed_operation == HLIEInferenceOperation::Responses);
    CHECK(route_observation_count == 1);

    service.stop_document_pipeline();
    CHECK(stop_count == 1);
}
