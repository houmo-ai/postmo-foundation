#include "hlie_app/http/hlie_control_http_error_mapper.h"

HLIE_NAMESPACE_BEGIN

server_http_res_ptr HLIEControlHttpErrorMapper::map(
    const HLIEControlError& error) {
    auto response = std::make_unique<server_res_generator_light>();
    response->error(format_error_response_ex(
        error.message, error_type_for(error.code),
        error.native_code == 0 ? -1 : error.native_code));
    return response;
}

HLIE_NAMESPACE_END
