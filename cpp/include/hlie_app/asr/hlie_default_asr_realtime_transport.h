#ifndef HLIE_DEFAULT_ASR_REALTIME_TRANSPORT_H
#define HLIE_DEFAULT_ASR_REALTIME_TRANSPORT_H

#include <memory>

#include "hlie_app/asr/hlie_asr_application_service.h"

HLIE_NAMESPACE_BEGIN

class HLIEWebSocketService;

class HLIEDefaultAsrRealtimeTransport final
    : public IHLIEAsrRealtimeTransport {
public:
    HLIEDefaultAsrRealtimeTransport();
    ~HLIEDefaultAsrRealtimeTransport() override;

    void start(HLIEAsrRealtimeHandler handler, int port) override;
    void stop() override;

private:
    std::unique_ptr<HLIEWebSocketService> service_;
};

HLIE_NAMESPACE_END

#endif // HLIE_DEFAULT_ASR_REALTIME_TRANSPORT_H
