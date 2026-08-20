#include "hlie_app/asr/hlie_default_asr_realtime_transport.h"

#include <utility>

#include "hlie_app/hlie_ws_service.h"

HLIE_NAMESPACE_BEGIN

HLIEDefaultAsrRealtimeTransport::HLIEDefaultAsrRealtimeTransport() = default;

HLIEDefaultAsrRealtimeTransport::~HLIEDefaultAsrRealtimeTransport() {
    stop();
}

void HLIEDefaultAsrRealtimeTransport::start(
    HLIEAsrRealtimeHandler handler, const int port) {
    if (service_) {
        return;
    }
    service_ = std::make_unique<HLIEWebSocketService>();
    service_->init();
    service_->run(std::move(handler), port);
}

void HLIEDefaultAsrRealtimeTransport::stop() {
    if (!service_) {
        return;
    }
    service_->stop();
    service_.reset();
}

HLIE_NAMESPACE_END
