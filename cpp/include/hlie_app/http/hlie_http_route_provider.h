#ifndef HLIE_APP_HTTP_ROUTE_PROVIDER_H
#define HLIE_APP_HTTP_ROUTE_PROVIDER_H

#include <functional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "httplib.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

struct HLIEHttpRouteConfig {
    std::vector<std::string> paths;
    HttpType method;
    std::function<server_http_res_ptr(const server_http_req&)> handler = nullptr;
    bool require_model = false;
    std::string model_type;
    std::function<void(const httplib::Request&, httplib::Response&)>
        native_handler = nullptr;
};

class IHLIEHttpRouteProvider {
public:
    virtual ~IHLIEHttpRouteProvider() = default;
    [[nodiscard]] virtual std::vector<HLIEHttpRouteConfig>
    get_route_configs() = 0;
};

class IHLIEWebServiceStateProvider {
public:
    virtual ~IHLIEWebServiceStateProvider() = default;
    virtual void set_server_port(int port) = 0;
    [[nodiscard]] virtual bool is_draining() const = 0;
    [[nodiscard]] virtual int get_model_service_state() const = 0;
};

class IHLIEAuthPolicy {
public:
    virtual ~IHLIEAuthPolicy() = default;
    [[nodiscard]] virtual bool authentication_enabled() const = 0;
    [[nodiscard]] virtual bool authorize_api_key(
        const std::string& api_key) const = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_HTTP_ROUTE_PROVIDER_H
