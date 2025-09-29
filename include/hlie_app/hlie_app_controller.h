#ifndef HLIE_APP_CONTROLLER_H
#define HLIE_APP_CONTROLLER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

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
    std::shared_ptr<HLIEWorkflow> workflow_;
    std::shared_ptr<HLIEWebService> web_service_;
    std::atomic<bool> is_running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    bool zero_load_;
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
