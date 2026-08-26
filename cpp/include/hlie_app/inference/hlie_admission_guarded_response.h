#ifndef HLIE_ADMISSION_GUARDED_RESPONSE_H
#define HLIE_ADMISSION_GUARDED_RESPONSE_H

#include <memory>
#include <optional>
#include <utility>

#include "hlie_core/control/hlie_admission_controller.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

class HLIEAdmissionGuardedResponse final : public server_http_res {
public:
    HLIEAdmissionGuardedResponse(
        server_http_res_ptr inner,
        std::shared_ptr<HLIEAdmissionController::Permit> permit)
        : permit_(std::move(permit)), inner_(std::move(inner)) {
        content_type = inner_->content_type;
        status = inner_->status;
        data = std::move(inner_->data);
        headers = std::move(inner_->headers);
        if (inner_->is_stream()) {
            next = [this](std::string& out) {
                if (!data.empty()) {
                    out = std::move(data);
                    data.clear();
                    return true;
                }
                return inner_->next(out);
            };
        }
    }

private:
    // Keep the permit until the upstream response has been destroyed.
    std::shared_ptr<HLIEAdmissionController::Permit> permit_;
    server_http_res_ptr inner_;
};

inline server_http_res_ptr attach_admission_permit(
    server_http_res_ptr response,
    std::optional<HLIEAdmissionController::Permit>& permit) {
    if (!response || !permit || !permit->active()) {
        return response;
    }
    auto shared_permit = std::make_shared<HLIEAdmissionController::Permit>(
        std::move(*permit));
    permit.reset();
    return std::make_unique<HLIEAdmissionGuardedResponse>(
        std::move(response), std::move(shared_permit));
}

HLIE_NAMESPACE_END

#endif // HLIE_ADMISSION_GUARDED_RESPONSE_H
