#include "hlie_app/hlie_server_http_proxy.h"

#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "hlie_base/hlie_logger.h"
#include "httplib.h"
#include "json.hpp"


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

    std::string generate_multipart_boundary() {
        thread_local std::mt19937 gen(std::random_device{}());
        static constexpr char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
        std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);

        std::string boundary = "----hlie-proxy-";
        for (int i = 0; i < 16; ++i) {
            boundary += chars[dis(gen)];
        }
        return boundary;
    }

    std::string sanitize_multipart_header_value(const std::string& text) {
        std::string result;
        result.reserve(text.size());
        for (const char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    }

    std::string build_multipart_body(
        const nlohmann::json& form_fields,
        const std::map<std::string, uploaded_file>& files,
        const std::string& boundary) {
        std::ostringstream body;

        const auto append_field = [&](const std::string& key, const nlohmann::json& value) {
            if (!value.is_string()) {
                throw std::invalid_argument("multipart form field must be a string");
            }
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_multipart_header_value(key) << "\"\r\n\r\n";
            body << value.get<std::string>() << "\r\n";
        };

        for (const auto& [key, value] : form_fields.items()) {
            if (value.is_array()) {
                for (const auto& item : value) {
                    append_field(key, item);
                }
            } else {
                append_field(key, value);
            }
        }

        for (const auto& [key, file] : files) {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_multipart_header_value(key) << "\"";
            if (!file.filename.empty()) {
                body << "; filename=\"" << sanitize_multipart_header_value(file.filename) << "\"";
            }
            body << "\r\n";
            body << "Content-Type: "
                 << (file.content_type.empty() ? "application/octet-stream"
                                               : sanitize_multipart_header_value(file.content_type))
                 << "\r\n\r\n";
            body.write(reinterpret_cast<const char*>(file.data.data()), static_cast<std::streamsize>(file.data.size()));
            body << "\r\n";
        }

        body << "--" << boundary << "--\r\n";
        return body.str();
    }
}


HLIE_NAMESPACE_BEGIN


// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    static constexpr size_t max_queue_size = 16;

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    bool writer_closed = false;
    bool reader_closed = false;
    void close_write() {
        {
            std::lock_guard<std::mutex> lk(mutex);
            writer_closed = true;
        }
        cv.notify_all();
    }
    void close_read() {
        {
            std::lock_guard<std::mutex> lk(mutex);
            reader_closed = true;
        }
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                cv.notify_all();
                return true;
            }
            if (writer_closed) {
                return false; // clean EOF
            }
            if (should_stop()) {
                // The reader already owns mutex here. Update the predicate
                // under that lock, then notify the writer after unlocking so
                // cancellation cannot be lost between its predicate check and
                // wait.
                reader_closed = true;
                lk.unlock();
                cv.notify_all();
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] {
            return reader_closed || writer_closed ||
                   queue.size() < max_queue_size;
        });
        if (reader_closed || writer_closed) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

struct ServerHttpProxy::trailer_state_t {
    mutable std::mutex mutex;
    std::map<std::string, std::string> trailers;
};


ServerHttpProxy::~ServerHttpProxy() {
    cancel();
}

void ServerHttpProxy::cancel() {
    std::function<void()> cleanup_to_run;
    {
        std::lock_guard lock(cleanup_mutex_);
        if (cleaned_up_) {
            return;
        }
        cleaned_up_ = true;
        cleanup_to_run = cleanup;
    }
    if (cleanup_to_run) {
        cleanup_to_run();
    }
}

std::map<std::string, std::string> ServerHttpProxy::response_trailers() const {
    if (!trailer_state_) {
        return {};
    }
    std::lock_guard lock(trailer_state_->mutex);
    return trailer_state_->trailers;
}


ServerHttpProxy::ServerHttpProxy(const std::string& method, const std::string& host, int port, const std::string& path,
                                         const std::map<std::string, std::string>& headers, const std::string& body,
                                         const std::map<std::string, uploaded_file>& files,
                                         const std::function<bool()>& should_stop,
                                         const ProxyTimeoutOptions& timeout_options) {
// shared between reader and writer threads
    auto cli  = std::make_shared<httplib::Client>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();
    auto response_started = std::make_shared<std::atomic<bool>>(false);
    trailer_state_ = std::make_shared<trailer_state_t>();
    auto trailer_state = trailer_state_;

    // setup Client
    const auto timeouts = timeout_options.normalized();
    const auto connect_seconds = timeouts.connect_timeout_ms / 1000;
    const auto connect_microseconds = (timeouts.connect_timeout_ms % 1000) * 1000;
    cli->set_connection_timeout(connect_seconds, connect_microseconds);
    cli->set_read_timeout(timeouts.read_timeout_for_body(body), 0);
    cli->set_write_timeout(timeouts.write_timeout_sec, 0);
    cli->set_max_timeout(static_cast<time_t>(timeouts.max_request_timeout_sec) * 1000);
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe, cli]() {
        pipe->close_read();
        pipe->close_write();
        cli->stop();
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
    httplib::ResponseHandler response_handler = [pipe, response_started](const httplib::Response & response) {
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
        const bool accepted = pipe->write(std::move(msg)); // send headers first
        response_started->store(accepted);
        return accepted;
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    std::string effective_body = body;
    std::string override_content_type;
    const bool has_files = !files.empty();
    if (has_files) {
        const auto form_fields = nlohmann::json::parse(body, nullptr, false);
        if (form_fields.is_discarded() || !form_fields.is_object()) {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }

        const std::string boundary = generate_multipart_boundary();
        effective_body = build_multipart_body(form_fields, files, boundary);
        override_content_type = "multipart/form-data; boundary=" + boundary;
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding" || lowered == "transfer-encoding" || lowered == "content-length") {
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    continue;
                }
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                req.set_header(key, host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = std::move(effective_body);
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    LOG_DEBUG("start proxy thread {} {}", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req, trailer_state, response_started]() mutable {
        auto result = cli->send(std::move(req));
        if (result.error() == httplib::Error::Success) {
            std::lock_guard lock(trailer_state->mutex);
            for (const auto& [key, value] : result->trailers) {
                trailer_state->trailers[key] = value;
            }
        } else {
            auto err_str = httplib::to_string(result.error());
            LOG_ERROR("http client error: {}", err_str.c_str());
            // Once response headers have been forwarded, appending a textual
            // proxy error would corrupt a binary/audio stream. Only synthesize
            // an HTTP 500 body when the upstream failed before it responded.
            if (!response_started->load()) {
                pipe->write({{}, 500, "", "text/plain; charset=utf-8"});
                pipe->write({{}, 0, "proxy error: " + err_str, ""});
            }
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
