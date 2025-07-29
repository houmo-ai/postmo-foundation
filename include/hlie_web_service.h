#ifndef HLIE_WEB_SERVICE_H
#define HLIE_WEB_SERVICE_H

#include <memory>


#include "hlie_base/hlie_base.h"

namespace httplib {
    class Server;
}
HLIE_NAMESPACE_BEGIN

class HLIEAppController;

class HLIEWebService {
public:
    HLIEWebService(std::weak_ptr<HLIEAppController> app_controller);
    ~HLIEWebService() = default;

    void run();
    void register_route();

private:
    std::weak_ptr<HLIEAppController> app_controller_;
    httplib::Server* svr;
};

HLIE_NAMESPACE_END

#endif // HLIE_WEB_SERVICE_H
