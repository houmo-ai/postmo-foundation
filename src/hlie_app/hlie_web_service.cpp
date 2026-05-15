#include "hlie_app/hlie_web_service.h"

#include <string>

#include <server/index.html.hpp>
#include <server/loading.html.hpp>
#include <server/bundle.js.hpp>
#include <server/bundle.css.hpp>
#include <server/server-common.h>
#include <server/server-http.h>

#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string process_body(const std::string& body) {
        try {
            if (body.empty()) {
                return body;
            }
            auto body_json = nlohmann::json::parse(body);
            // request
            if (body_json.contains("messages") && body_json["messages"].is_array() && !body_json["messages"].empty()) {
                for (auto& message : body_json["messages"]) {
                    if (message.contains("content") && message["content"].is_array() && !message["content"].empty()) {
                        for (auto& content_item : message["content"]) {
                            if (content_item.contains("type") && content_item["type"] == "image_url" &&
                                content_item.contains("image_url") && content_item["image_url"].contains("url")) {
                                std::string b64_json_data = content_item["image_url"]["url"];
                                LOG_TRACE("Request b64_json: {}", b64_json_data);
                                if (b64_json_data.length() > 100) {
                                    content_item["image_url"]["url"] =
                                        b64_json_data.substr(0, 100) + "..., length=" + std::to_string(b64_json_data.length());
                                }
                            }
                        }
                    }
                }
            }
            // response
            if (body_json.contains("data") && body_json["data"].is_array() && !body_json["data"].empty()) {
                for (auto& item : body_json["data"]) {
                    if (item.contains("b64_json")) {
                        std::string b64_json_data = item["b64_json"];
                        LOG_TRACE("Response b64_json: {}", b64_json_data);
                        if (b64_json_data.length() > 100) {
                            item["b64_json"] = b64_json_data.substr(0, 100) + "..., length=" + std::to_string(b64_json_data.length());
                        }
                    }
                }
            }

            return body_json.dump();
        } catch (const json::exception& e) {
            LOG_WARN("Failed to parse body as JSON: {}", e.what());
            return body;
        }
    }

    void log_server_request(const httplib::Request& req, const httplib::Response& res) {
        // skip GH copilot requests when using default port
        if (req.path == "/v1/health" || req.path == "/health") {
            return;
        }

        const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "null";
        const auto access_logger = HLIELogger::get_logger(HLIE_ACCESS);
        HLIE_CHECK_NULL(access_logger);
        access_logger->info("{} - - \"{} {}\" {} [RequestID={}]", req.remote_addr, req.method, req.path, res.status, request_id);
        access_logger->flush();

        LOG_INFO("Request[{}]: {} {} {} {} \n{}", request_id, req.method, req.path, req.remote_addr, res.status, process_body(req.body));
        LOG_INFO("Response[{}]: {}", request_id, process_body(res.body));
    }

    void res_error(httplib::Response& res, const json& error_data) {
        const json final_response{{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    }

    void exception_handler(const httplib::Request&, httplib::Response& res, const std::exception_ptr& ep) {
        std::string message;
        int ret = -1;
        try {
            std::rethrow_exception(ep);
        } catch (const WebServerException& e) {
            message = e.get_err_msg();
            ret = e.get_ret_code();
        } catch (const std::exception& e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "null";
        const nlohmann::json formatted_error = format_error_response_ex(message, ERROR_TYPE_SERVER, ret);
        LOG_ERROR("[Request-{}] got server exception: {}", request_id, formatted_error.dump().c_str());
        res_error(res, formatted_error);
    }


    void set_headers(httplib::Response& res, const std::map<std::string, std::string>& headers) {
        for (const auto& [key, value] : headers) {
            res.set_header(key, value);
        }
    }

    std::map<std::string, std::string> get_params(const httplib::Request& req) {
        std::map<std::string, std::string> params;
        for (const auto& [key, value] : req.params) {
            params[key] = value;
        }
        for (const auto& [key, value] : req.path_params) {
            params[key] = value;
        }
        return params;
    }

    std::map<std::string, std::string> get_headers(const httplib::Request& req) {
        std::map<std::string, std::string> headers;
        for (const auto& [key, value] : req.headers) {
            headers[key] = value;
        }
        return headers;
    }

    std::string build_query_string(const httplib::Request & req) {
        std::string qs;
        for (const auto & [key, value] : req.params) {
            if (!qs.empty()) {
                qs += '&';
            }
            qs += httplib::encode_query_component(key) + "=" + httplib::encode_query_component(value);
        }
        return qs;
    }

    void process_handler_response(std::unique_ptr<server_http_req>&& request, server_http_res_ptr & response, httplib::Response & res) {
        if (response->is_stream()) {
            res.status = response->status;
            set_headers(res, response->headers);
            std::string content_type = response->content_type;
            // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
            std::shared_ptr q_ptr = std::move(request);
            std::shared_ptr r_ptr = std::move(response);
            const auto chunked_content_provider = [response = r_ptr](size_t, httplib::DataSink & sink) -> bool {
                std::string chunk;
                bool has_next = response->next(chunk);
                if (!chunk.empty()) {
                    if (!sink.write(chunk.data(), chunk.size())) {
                        return false;
                    }
                    LOG_DEBUG("http: streamed chunk: {}", chunk.c_str());
                }
                if (!has_next) {
                    sink.done();
                    LOG_DEBUG("http: stream ended");
                }
                return has_next;
            };
            const auto on_complete = [request = q_ptr, response = r_ptr](bool) mutable {
                response.reset(); // trigger the destruction of the response object
                request.reset();  // trigger the destruction of the request object
            };
            res.set_chunked_content_provider(content_type, chunked_content_provider, on_complete);
        } else {
            res.status = response->status;
            set_headers(res, response->headers);
            res.set_content(response->data, response->content_type);
        }
    }

    void res_ok(httplib::Response& res, const json& data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    }
} // namespace

HLIEWebService::HLIEWebService(const std::shared_ptr<HLIEWorkflow>& workflow) {
    workflow_ = workflow;
}

void HLIEWebService::init(int n_threads_http) {
    web_server_ = std::make_shared<httplib::Server>();
    web_server_->set_default_headers({{"Server", "HLIECpp"}});

    web_server_->Get("/version", [](const httplib::Request&, httplib::Response& res) {
        const std::string json_response = R"({"Version": "v1.7.0"})";
        res.set_content(json_response, "application/json");
    });

    web_server_->set_logger(log_server_request);
    web_server_->set_exception_handler(exception_handler);

    // set timeouts and change hostname and port
    web_server_->set_read_timeout(600);
    web_server_->set_write_timeout(600);


    web_server_->set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) { return this->pre_routing_handler(req, res); });

    // using embedded static index.html
    web_server_->Get("/", [](const httplib::Request & /*req*/, httplib::Response & res) {
        // COEP and COOP headers, required by pyodide (python interpreter)
        res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
        res.set_header("Cross-Origin-Opener-Policy", "same-origin");
        res.set_content(reinterpret_cast<const char*>(index_html), index_html_len, "text/html; charset=utf-8");
        return false;
    });
    web_server_->Get("/bundle.js", [](const httplib::Request & /*req*/, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char*>(bundle_js), bundle_js_len, "application/javascript; charset=utf-8");
        return false;
    });
    web_server_->Get("/bundle.css", [](const httplib::Request & /*req*/, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char*>(bundle_css), bundle_css_len, "text/css; charset=utf-8");
        return false;
    });

    web_server_->new_task_queue = [n_threads_http] { return new httplib::ThreadPool(n_threads_http); };
    LOG_INFO("Web Service init done, n_threads_http: {}", n_threads_http);
}

int HLIEWebService::run(const std::string& host_name, int port) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    const auto& args = config_services->get_args();
    if (const auto cfg_port = config_services->get_int("server", "port", -1); cfg_port < 0) {
        if (args.find("--port") != args.end()) {
            try {
                port = std::stoi(args.at("--port"));
                LOG_DEBUG("Using arg port: {}", port);
            } catch (...) {
                LOG_ERROR("Cannot found port config, default port will be used");
            }
        } else {
            LOG_DEBUG("Cannot found port config, default port will be used");
        }
    } else {
        port = cfg_port;
    }

    if (args.find("--instance_id") != args.end()) {
        port += std::stoi(args.at("--instance_id"));
        LOG_DEBUG("Instance[{}] port: {}", args.at("--instance_id"), port);
    }

    if (const bool was_bound = web_server_->bind_to_port(host_name, port); !was_bound) {
        LOG_ERROR("couldn't bind HTTP server socket, hostname: {}, port: %d\n", host_name, port);
        stop();

        return -1;
    }

    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    workflow_->set_server_port(port);

    worker_thread_ = std::make_unique<std::thread>([this] {
        int ret = this->web_server_->listen_after_bind();
        spdlog::debug("webserver listen ret: {}", ret);
    });
    web_server_->wait_until_ready();

    LOG_INFO("HTTP server is listening, hostname: {}, port: {}", host_name, port);
    is_ready.store(true);
    return 0;
}


void HLIEWebService::get(const std::string& path, const server_http_context::handler_t& handler) const {
    web_server_->Get(path, [handler](const httplib::Request& req, httplib::Response& res) {
        auto request = std::make_unique<server_http_req>(server_http_req{get_params(req),
                                                                         get_headers(req),
                                                                         req.path,
                                                                         build_query_string(req),
                                                                         req.body,
                                                                         {},
                                                                         req.is_connection_closed});
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);

    });
}

void HLIEWebService::post(const std::string& path, const server_http_context::handler_t& handler) const {
    web_server_->Post(path, [handler](const httplib::Request& req, httplib::Response& res) {
        if (const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "";
            !request_id.empty()) {
            LOG_DEBUG("[Request-{}] come in", request_id);
        }

        std::string body = req.body;
        std::map<std::string, uploaded_file> files;

        if (req.is_multipart_form_data()) {
            // translate text fields to a JSON object and use it as the body
            json form_json = json::object();
            for (const auto & [key, field] : req.form.fields) {
                if (form_json.contains(key)) {
                    // if the key already exists, convert it to an array
                    if (!form_json[key].is_array()) {
                        json existing_value = form_json[key];
                        form_json[key] = json::array({existing_value});
                    }
                    form_json[key].push_back(field.content);
                } else {
                    form_json[key] = field.content;
                }
            }
            body = form_json.dump();

            // populate files from multipart form
            for (const auto & [key, file] : req.form.files) {
                files[key] = uploaded_file{
                    raw_buffer(file.content.begin(), file.content.end()),
                    file.filename,
                    file.content_type,
                };
            }
        }

        auto request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            req.body,
            std::move(files),
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req& req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument& e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception& e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{"error", error_data}});
            LOG_WARN("got exception: {}", res->data.c_str());
        } catch (const std::exception& e) {
            LOG_ERROR("got another exception: {} | while handling exception: {}", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

void HLIEWebService::register_routers() const {
    HLIE_CHECK_NULL(workflow_);

    const auto& routes = workflow_->get_route_configs();

    for (const auto& [paths, method, handler, require_model, model_type, ori_handler] : routes) {
        // 检查是否需要特定模型服务
        if (require_model) {
            if (const auto model_service = workflow_->get_model_service(model_type); !model_service) {
                LOG_WARN("route {} and related - required model service not available", paths[0]);
            } else if (model_service->get_state() != SERVER_STATE_READY) {
                LOG_WARN("route {} and related - model service not ready", paths[0]);
            }
        }

        // 注册路由
        if (method == GET) {
            for (const auto& path : paths) {
                LOG_INFO("Registered GET route: {}", path);
                get(path, ex_wrapper(handler));
            }
        } else if (method == POST) {
            if (ori_handler) {
                for (const auto& path : paths) {
                    LOG_INFO("Registered POST ORI route: {}", path);
                    web_server_->Post(path, ori_handler);
                }
                continue;
            }

            for (const auto& path : paths) {
                LOG_INFO("Registered POST route: {}", path);
                post(path, ex_wrapper(handler));
            }
        }
    }
}


void HLIEWebService::register_post(const std::string& path, httplib::Server::Handler&& func) const {
    web_server_->Post(path, func);
}

void HLIEWebService::register_get(const std::string& path, httplib::Server::Handler&& func) const {
    web_server_->Get(path, func);
}

void HLIEWebService::stop() {
    if (web_server_) {
        web_server_->stop();
    }

    if (!worker_thread_) {
        return;
    }

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
}

httplib::Server::HandlerResponse HLIEWebService::pre_routing_handler(const httplib::Request& req, httplib::Response& res) const {
    LOG_TRACE("Pre-routing handler called for {} {}", req.method, req.path);
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
    if (req.method == "OPTIONS") {
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "GET, POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_content("", "text/html"); // blank response, no data
        return httplib::Server::HandlerResponse::Handled; // skip further processing
    }

    if (!middleware_server_state(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }

    if (!middleware_validate_api_key(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }

    return httplib::Server::HandlerResponse::Unhandled;
}

bool HLIEWebService::middleware_server_state(const httplib::Request& req, httplib::Response& res) const {
    LOG_TRACE("middleware_server_state called for path: {}", req.path);
    bool ready = is_ready.load();
    if (!ready) {
#ifdef LLAMA_BUILD_WEBUI
        auto tmp = string_split<std::string>(req.path, '.');
        if (req.path == "/" || tmp.back() == "html") {
            res.status = 503;
            res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
        } else
#endif
        {
            // no endpoints is allowed to be accessed when the server is not ready
            // this is to prevent any data races or inconsistent states
            res.status = 503;
            res.set_content(
                safe_json_to_str(json {
                    {"error", {
                        {"message", "Loading model"},
                        {"type", "unavailable_error"},
                        {"code", 503}
                    }}
                }),
                "application/json; charset=utf-8"
            );
        }
        return false;
    }
    return true;
}

bool HLIEWebService::middleware_validate_api_key(const httplib::Request& req, httplib::Response& res) const {
    HLIE_CHECK_NULL_RETURN(workflow_, 0);
    static const std::unordered_set<std::string> public_endpoints = {
        "/health",
        "/v1/health",
        "/models",
        "/v1/models",
        "/",
        "/index.html",
        "/bundle.js",
        "/bundle.css",
    };

    // If path is public or static file, skip validation
    if (public_endpoints.find(req.path) != public_endpoints.end()) {
        return true;
    }

    const int state = workflow_->get_model_service_state();
    const auto request_id = HlieUtil::uuid();
    res.set_header("X-Request-ID", request_id);
    LOG_INFO("Request[{}] come in, current state[0: LOADING_MODEL 1:READY]: {}", request_id, state);
    if (const auto logger = HLIELogger::get_logger(HLIE_ACCESS)) {
        logger->info("{} - - \"{} {}\" [RequestID={}]", req.remote_addr, req.method, req.path, request_id);
    }

    if (state == SERVER_STATE_LOADING_MODEL) {
        res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
        return false;
    }

    // If API key is not set, skip validation
    if (workflow_->get_api_keys().empty()) {
        return true;
    }

    // Check for API key in the Authorization header
    std::string req_api_key = req.get_header_value("Authorization");
    if (req_api_key.empty()) {
        // retry with anthropic header
        req_api_key = req.get_header_value("X-Api-Key");
    }

    // remove the "Bearer " prefix if needed
    std::string prefix = "Bearer ";
    if (req_api_key.substr(0, prefix.size()) == prefix) {
        req_api_key = req_api_key.substr(prefix.size());
    }

    // validate the API key
    if (std::find(workflow_->get_api_keys().begin(), workflow_->get_api_keys().end(), req_api_key) != workflow_->get_api_keys().end()) {
        return true; // API key is valid
    }

    // API key is invalid or not provided
    res.status = 401;
    res.set_content(
        safe_json_to_str(json {
            {"error", {
                {"message", "Invalid API Key"},
                {"type", "authentication_error"},
                {"code", 401}
            }}
        }),
        "application/json; charset=utf-8"
    );

    LOG_WARN("Unauthorized: Invalid API Key");
    return false;
}

HLIE_NAMESPACE_END
