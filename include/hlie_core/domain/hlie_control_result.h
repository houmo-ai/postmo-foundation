#pragma once

#include <string>
#include <utility>
#include <variant>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Protocol- and SDK-neutral failure vocabulary used inside the control plane.
// HTTP status/error bodies and TCIM return codes are mapped at app boundaries.
enum class HLIEControlErrorCode {
    InvalidArgument,
    NotFound,
    AlreadyExists,
    Conflict,
    Unavailable,
    CapacityExceeded,
    Unsupported,
    PermissionDenied,
    DeadlineExceeded,
    Cancelled,
    StaleOperation,
    BackendStartFailed,
    ReadyTimeout,
    Internal,
};

enum class HLIEControlErrorStage {
    Validation,
    Catalog,
    Reservation,
    RuntimeStart,
    Readiness,
    Dispatch,
    Stop,
    Reconcile,
};

struct HLIEControlError {
    HLIEControlErrorCode code = HLIEControlErrorCode::Internal;
    HLIEControlErrorStage stage = HLIEControlErrorStage::Validation;
    std::string message;
    int native_code = 0;
    bool retryable = false;
};

// A deliberately small result type. It prevents the Controller/Router boundary
// from encoding failures as null pointers, magic integers, or HTTP JSON.
template <typename T>
class HLIEControlResult {
public:
    [[nodiscard]] static HLIEControlResult success(T value) {
        return HLIEControlResult(std::move(value));
    }

    [[nodiscard]] static HLIEControlResult failure(HLIEControlError error) {
        return HLIEControlResult(std::move(error));
    }

    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(storage_);
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return has_value();
    }

    [[nodiscard]] const T* value_if() const noexcept {
        return std::get_if<T>(&storage_);
    }

    [[nodiscard]] T* value_if() noexcept {
        return std::get_if<T>(&storage_);
    }

    [[nodiscard]] const HLIEControlError* error_if() const noexcept {
        return std::get_if<HLIEControlError>(&storage_);
    }

private:
    explicit HLIEControlResult(T value) : storage_(std::move(value)) {}
    explicit HLIEControlResult(HLIEControlError error) : storage_(std::move(error)) {}

    std::variant<T, HLIEControlError> storage_;
};

HLIE_NAMESPACE_END
