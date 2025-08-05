#ifndef HLIE_WEB_SERVICE_H
#define HLIE_WEB_SERVICE_H

#include <functional>
#include <memory>


#include "hlie_base/hlie_base.h"
#include "httplib.h"

namespace httplib {
    class Server;
}
HLIE_NAMESPACE_BEGIN

class HLIEWorkflow;

class HLIE_API HLIEWebService {
public:
    explicit HLIEWebService(const std::shared_ptr<HLIEWorkflow>& workflow);
    ~HLIEWebService() = default;

    void init(int n_threads_http);

    int run(std::string host_name, int port);

    void register_routers() const;
    void register_post(const std::string& path, httplib::Server::Handler&& func) const;
    void register_get(const std::string& path, httplib::Server::Handler&& func) const;

    void stop();

private:
    std::shared_ptr<HLIEWorkflow> workflow_;
    std::shared_ptr<httplib::Server> web_server_;

    std::unique_ptr<std::thread> worker_thread_;  // 工作线程

    httplib::Server::HandlerResponse pre_routing_handler(const httplib::Request& req, httplib::Response& res) const;
    bool middleware_server_state(const httplib::Request& req, httplib::Response& res) const;
    bool middleware_validate_api_key(const httplib::Request &req, httplib::Response &res) const;
};

HLIE_NAMESPACE_END

#endif // HLIE_WEB_SERVICE_H
