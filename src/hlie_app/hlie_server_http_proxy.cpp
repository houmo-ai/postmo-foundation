#include "hlie_app/hlie_server_http_proxy.h"

#include <queue>
#include <string>

#include "hlie_base/hlie_logger.h"
#include "httplib.h"


namespace {
    std::string to_lower_copy(const std::string& value) {
        std::string lowered(value.size(), '\0');
        std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
        return lowered;
    }

    bool should_strip_proxy_header(const std::string& header_name) {
        // Headers that get duplicated when router forwards child responses
        if (header_name == "server" ||
            header_name == "transfer-encoding" ||
            header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
            header_name == "keep-alive") {
            return true;
            }

        // Router injects CORS, child also sends them: duplicate
        if (header_name.rfind("access-control-", 0) == 0) {
            return true;
        }

        return false;
    }
}


HLIE_NAMESPACE_BEGIN


// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};


ServerHttpProxy::ServerHttpProxy(const std::string& method, const std::string& host, int port, const std::string& path,
                                        const std::map<std::string, std::string>& headers, const std::string& body,
                                        const std::function<bool()>& should_stop) {
// shared between reader and writer threads
    auto cli  = std::make_shared<httplib::Client>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    // setup Client
    cli->set_connection_timeout(0, 200000); // 200 milliseconds
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            req.set_header(key, value);
        }
        req.body = body;
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    LOG_DEBUG("start proxy thread {} {}", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            LOG_ERROR("http client error: {}", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        }
        pipe->close_write(); // signal EOF to reader
        LOG_DEBUG("client request thread ended");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            LOG_DEBUG("received response headers");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            LOG_DEBUG("no response headers received (request cancelled?)");
        }
    }

}

HLIE_NAMESPACE_END
