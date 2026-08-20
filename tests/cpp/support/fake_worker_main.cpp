#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

using json = nlohmann::json;

struct Options {
    std::string mode = "serve";
    std::string launch_token;
    std::uint64_t generation = 0;
    int port = 0;
    int crash_after_ms = -1;
    int exit_code = 1;
    int shutdown_grace_ms = 50;
};

std::unordered_map<std::string, std::string> parse_arguments(const int argc, char** argv) {
    std::unordered_map<std::string, std::string> result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        if (index + 1 < argc && std::string(argv[index + 1]).rfind("--", 0) != 0) {
            result[key] = argv[++index];
        } else {
            result[key] = "true";
        }
    }
    return result;
}

template <typename Integer>
bool parse_integer(const std::unordered_map<std::string, std::string>& arguments,
                   const std::string& key, Integer& value) {
    const auto it = arguments.find(key);
    if (it == arguments.end()) {
        return true;
    }
    try {
        const auto parsed = std::stoll(it->second);
        if (parsed < static_cast<long long>(std::numeric_limits<Integer>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<Integer>::max())) {
            return false;
        }
        value = static_cast<Integer>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_generation(const std::unordered_map<std::string, std::string>& arguments,
                      const std::string& key, std::uint64_t& value) {
    const auto it = arguments.find(key);
    if (it == arguments.end()) {
        return true;
    }
    try {
        std::size_t parsed_characters = 0;
        const auto parsed = std::stoull(it->second, &parsed_characters);
        if (parsed_characters != it->second.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool load_options(const int argc, char** argv, Options& options) {
    const auto arguments = parse_arguments(argc, argv);
    if (const auto it = arguments.find("--mode"); it != arguments.end()) {
        options.mode = it->second;
    }
    if (const auto it = arguments.find("--launch-token"); it != arguments.end()) {
        options.launch_token = it->second;
    }
    if (!parse_generation(arguments, "--generation", options.generation) ||
        !parse_integer(arguments, "--port", options.port) ||
        !parse_integer(arguments, "--crash-after-ms", options.crash_after_ms) ||
        !parse_integer(arguments, "--exit-code", options.exit_code) ||
        !parse_integer(arguments, "--shutdown-grace-ms", options.shutdown_grace_ms)) {
        return false;
    }
    return options.crash_after_ms >= -1 && options.shutdown_grace_ms >= 0;
}

[[noreturn]] void immediate_exit(const int exit_code) {
    std::_Exit(exit_code);
}

} // namespace

int main(const int argc, char** argv) {
    Options options;
    if (!load_options(argc, argv, options)) {
        std::cerr << "invalid fake worker arguments" << std::endl;
        return 64;
    }

    if (options.mode == "crash-before-ready") {
        const int delay_ms = options.crash_after_ms < 0 ? 250 : options.crash_after_ms;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        return options.exit_code;
    }
    if (options.mode != "serve" || options.port <= 0 || options.port > 65535) {
        std::cerr << "fake worker requires --mode serve and a valid --port" << std::endl;
        return 64;
    }

    httplib::Server server;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    bool shutdown_requested = false;
    bool server_finished = false;

    const auto ready_response = [&](const httplib::Request&, httplib::Response& response) {
        response.set_content(json{{"status", "live"},
                                  {"generation", options.generation}}.dump(),
                             "application/json");
    };
    server.Get("/live", ready_response);
    server.Get("/ready", ready_response);
    server.Post("/v1/fake/infer", [&](const httplib::Request& request, httplib::Response& response) {
        response.set_content(json{{"object", "fake.inference"},
                                  {"generation", options.generation},
                                  {"input", request.body}}.dump(),
                             "application/json");
    });
    server.Post("/internal/worker/shutdown",
                [&](const httplib::Request& request, httplib::Response& response) {
        try {
            const auto body = json::parse(request.body);
            const auto generation = body.value("generation", std::uint64_t{0});
            const auto token = body.value("launch_token", std::string{});
            if (generation != options.generation || token != options.launch_token) {
                response.status = 403;
                response.set_content(R"({"error":"invalid worker launch ticket"})",
                                     "application/json");
                return;
            }
        } catch (const std::exception&) {
            response.status = 400;
            response.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }

        response.set_content(R"({"status":"shutting_down"})", "application/json");
        {
            std::lock_guard lock(state_mutex);
            shutdown_requested = true;
        }
        state_cv.notify_all();
    });

    std::thread shutdown_thread([&] {
        std::unique_lock lock(state_mutex);
        state_cv.wait(lock, [&] { return shutdown_requested || server_finished; });
        if (!shutdown_requested) {
            return;
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(options.shutdown_grace_ms));
        server.stop();
    });

    std::thread crash_thread;
    if (options.crash_after_ms >= 0) {
        crash_thread = std::thread([&] {
            std::unique_lock lock(state_mutex);
            if (!state_cv.wait_for(lock, std::chrono::milliseconds(options.crash_after_ms),
                                   [&] { return shutdown_requested || server_finished; })) {
                immediate_exit(options.exit_code);
            }
        });
    }

    const bool listened = server.listen("127.0.0.1", options.port);
    {
        std::lock_guard lock(state_mutex);
        server_finished = true;
    }
    state_cv.notify_all();
    shutdown_thread.join();
    if (crash_thread.joinable()) {
        crash_thread.join();
    }
    return listened ? 0 : 65;
}
