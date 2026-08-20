#include "hlie_app/inference/hlie_inference_http_error_mapper.h"

#include <utility>

#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

server_http_res_ptr HLIEInferenceHttpErrorMapper::map(
    HLIEInferenceDispatchResult result) {
    if (result.dispatched()) {
        return std::move(result.response);
    }

    auto response = std::make_unique<server_res_generator_light>();
    switch (result.status) {
    case HLIEInferenceDispatchStatus::Draining:
        response->error(format_error_response_ex(
            "service is draining", ERROR_TYPE_UNAVAILABLE, 503));
        break;
    case HLIEInferenceDispatchStatus::ModelNotFound:
        response->error(format_error_response(
            "model not found", ERROR_TYPE_NOT_FOUND));
        break;
    case HLIEInferenceDispatchStatus::ModelNotReady:
        response->error(format_error_response(
            "model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        break;
    case HLIEInferenceDispatchStatus::RuntimeUnavailable:
    case HLIEInferenceDispatchStatus::EndpointUnavailable:
        LOG_WARN("Inference target unavailable: model={}, status={}, detail={}",
                 result.model_key, static_cast<int>(result.status), result.detail);
        response->error(format_error_response_ex(
            "model runtime is unavailable", ERROR_TYPE_UNAVAILABLE, 80004));
        break;
    case HLIEInferenceDispatchStatus::UnsupportedOperation:
        response->error(format_error_response_ex(
            "inference operation is not supported by this model",
            ERROR_TYPE_NOT_SUPPORTED, 80005));
        break;
    case HLIEInferenceDispatchStatus::EndpointFailure:
        LOG_ERROR("Inference endpoint failed: model={}, detail={}",
                  result.model_key, result.detail);
        response->error(format_error_response(
            "inference endpoint failed", ERROR_TYPE_SERVER));
        break;
    case HLIEInferenceDispatchStatus::Dispatched:
        response->error(format_error_response(
            "inference endpoint returned no response", ERROR_TYPE_SERVER));
        break;
    }
    return response;
}

HLIE_NAMESPACE_END
