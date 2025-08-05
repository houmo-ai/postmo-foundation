#ifndef HLIE_APP_CONTROLLER_H
#define HLIE_APP_CONTROLLER_H

#include <atomic>
#include <memory>
#include <mutex>

#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_workflow.h"

HLIE_NAMESPACE_BEGIN

class HLIEWorkflow;
class HLIEWebService;
class HLIEConfigService;

class HLIE_API HLIEAppController {
public:
    HLIEAppController() = default;
    virtual ~HLIEAppController() = default;

    virtual int initialize() = 0;

    virtual int run() = 0;

    virtual void stop() {
        if (workflow_) {
            workflow_->stop();
        }
        is_running_.store(false);
        cv_.notify_one();
    }

protected:
    std::shared_ptr<HLIEWorkflow> workflow_;
    std::shared_ptr<HLIEWebService> web_service_;
    std::shared_ptr<HLIEConfigService> sys_config_;
    std::atomic<bool> is_running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};


class HLIE_API HLIEWinAppController final : public HLIEAppController {
public:
    HLIEWinAppController() = default;
    ~HLIEWinAppController() override = default;

    int initialize() override;
    int run() override;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_CONTROLLER_H
