#ifndef HLIE_APP_CONTROLLER_H
#define HLIE_APP_CONTROLLER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "hlie_app/hlie_balancer_client.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEWorkflow;
class HLIEWebService;
class HLIEConfigService;

class HLIE_API HLIEAppController {
public:
    HLIEAppController() = default;
    virtual ~HLIEAppController() = default;

    virtual int initialize() = 0;

    virtual int run();

    virtual void stop();

protected:
    void request_stop_after_response();

    std::shared_ptr<HLIEWorkflow> workflow_;
    std::shared_ptr<HLIEWebService> web_service_;
    HLIEBalancerClient balancer_client_;
    std::atomic<bool> is_running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_started_{false};
    std::atomic<bool> stop_completed_{false};
    std::mutex stop_completion_mutex_;
    std::condition_variable stop_completion_cv_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> worker_shutdown_requested_{false};
    bool zero_load_{false};
};


#ifdef _WIN32

class HLIE_API HLIEWinAppController final : public HLIEAppController {
public:
    HLIEWinAppController() = default;
    ~HLIEWinAppController() override = default;

    int initialize() override;
};

#else

class HLIE_API HLIELinuxAppController final : public HLIEAppController {
public:
    HLIELinuxAppController() = default;
    ~HLIELinuxAppController() override = default;

    int initialize() override;
};

#endif

HLIE_NAMESPACE_END

#endif // HLIE_APP_CONTROLLER_H
