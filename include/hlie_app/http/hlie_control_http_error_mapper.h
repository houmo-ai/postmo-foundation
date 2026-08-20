#pragma once

#include "hlie_base/hlie_common_define.h"
#include "hlie_core/domain/hlie_control_result.h"

HLIE_NAMESPACE_BEGIN

// Maps protocol-neutral control-plane failures at the HTTP boundary.
class HLIEControlHttpErrorMapper {
public:
    [[nodiscard]] static constexpr error_type error_type_for(
        const HLIEControlErrorCode code) noexcept {
        switch (code) {
        case HLIEControlErrorCode::InvalidArgument:
        case HLIEControlErrorCode::AlreadyExists:
        case HLIEControlErrorCode::Conflict:
        case HLIEControlErrorCode::StaleOperation:
            return ERROR_TYPE_INVALID_REQUEST;
        case HLIEControlErrorCode::NotFound:
            return ERROR_TYPE_NOT_FOUND;
        case HLIEControlErrorCode::Unavailable:
        case HLIEControlErrorCode::CapacityExceeded:
        case HLIEControlErrorCode::DeadlineExceeded:
        case HLIEControlErrorCode::ReadyTimeout:
            return ERROR_TYPE_UNAVAILABLE;
        case HLIEControlErrorCode::Unsupported:
            return ERROR_TYPE_NOT_SUPPORTED;
        case HLIEControlErrorCode::PermissionDenied:
            return ERROR_TYPE_PERMISSION;
        case HLIEControlErrorCode::Cancelled:
        case HLIEControlErrorCode::BackendStartFailed:
        case HLIEControlErrorCode::Internal:
            return ERROR_TYPE_SERVER;
        }
        return ERROR_TYPE_SERVER;
    }

    [[nodiscard]] static server_http_res_ptr map(
        const HLIEControlError& error);
};

HLIE_NAMESPACE_END
