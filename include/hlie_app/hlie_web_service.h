#ifndef HLIE_WEB_SERVICE_H
#define HLIE_WEB_SERVICE_H

#include <memory>


#include "hlie_base/hlie_base.h"
#include "hlie_app/http/hlie_http_route_provider.h"
#include "httplib.h"
#include "server/server-http.h"

namespace httplib {
    class Server;
}
HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEWebService {
public:
    HLIEWebService(std::shared_ptr<IHLIEHttpRouteProvider> route_provider,
                   std::shared_ptr<IHLIEWebServiceStateProvider> state_provider,
                   std::shared_ptr<IHLIEAuthPolicy> auth_policy);
    ~HLIEWebService() = default;

    void init(int n_threads_http);

    int run(const std::string& host_name = "0.0.0.0", int port = 7901);
    [[nodiscard]] int get_port() const { return port_; }
    void get(const std::string& path, const server_http_context::handler_t& handler) const;
    void post(const std::string& path, const server_http_context::handler_t& handler) const;

    void register_routers() const;
    void register_post(const std::string& path, httplib::Server::Handler&& func) const;
    void register_get(const std::string& path, httplib::Server::Handler&& func) const;

    void stop();

private:
    std::shared_ptr<IHLIEHttpRouteProvider> route_provider_;
    std::shared_ptr<IHLIEWebServiceStateProvider> state_provider_;
    std::shared_ptr<IHLIEAuthPolicy> auth_policy_;
    std::shared_ptr<httplib::Server> web_server_;
    int port_ = 0;

    std::unique_ptr<std::thread> worker_thread_;  // 工作线程
    std::atomic<bool> is_ready = false;

    httplib::Server::HandlerResponse pre_routing_handler(const httplib::Request& req, httplib::Response& res) const;
    bool middleware_server_state(const httplib::Request& req, httplib::Response& res) const;
    bool middleware_validate_api_key(const httplib::Request &req, httplib::Response &res) const;
};

HLIE_NAMESPACE_END

#endif // HLIE_WEB_SERVICE_H
