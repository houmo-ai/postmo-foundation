#ifndef HLIE_APP_INFERENCE_HTTP_ERROR_MAPPER_H
#define HLIE_APP_INFERENCE_HTTP_ERROR_MAPPER_H

#include "hlie_app/inference/hlie_inference_dispatcher.h"

HLIE_NAMESPACE_BEGIN

class HLIEInferenceHttpErrorMapper {
public:
    [[nodiscard]] static server_http_res_ptr map(
        HLIEInferenceDispatchResult result);
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_INFERENCE_HTTP_ERROR_MAPPER_H
