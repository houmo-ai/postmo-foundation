#ifndef HLIE_APP_CONTROLLER_H
#define HLIE_APP_CONTROLLER_H

#include <memory>
#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_config_service.h"

HLIE_NAMESPACE_BEGIN

class HLIEWorkflow;
class HLIEConfigService;

class HLIEAppController {
public:
    HLIEAppController() = default;
    ~HLIEAppController() = default;

    void initialize();
    void run();

private:
    std::shared_ptr<HLIEWorkflow> workflow_;
    std::shared_ptr<HLIEConfigService> config_service_;
};

HLIE_NAMESPACE_END

#endif //HLIE_APP_CONTROLLER_H
