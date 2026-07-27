#include "mirage_process.h"

#include <catch_amalgamated.hpp>
#include <httplib.h>

#include <string>

namespace {

class MirageFixture {
protected:
    MirageFixture() : process_(), client_(process_.host(), process_.port()) {
        client_.set_connection_timeout(2, 0);
        client_.set_read_timeout(5, 0);
    }

    httplib::Result post_chat(const std::string& body) {
        return client_.Post("/v1/chat/completions", body, "application/json");
    }

    MirageProcess process_;
    httplib::Client client_;
};

} // namespace

TEST_CASE_METHOD(MirageFixture, "Mirage exposes OpenAI-compatible discovery and chat", "[component][mirage]") {
    const auto health = client_.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
    CHECK(health->get_header_value("X-HLAW-Mirage") == "true");

    const auto models = client_.Get("/v1/models");
    REQUIRE(models);
    CHECK(models->status == 200);
    CHECK(models->body.find("mock-default") != std::string::npos);

    const auto chat = post_chat(R"({"model":"mock-default","messages":[]})");
    REQUIRE(chat);
    CHECK(chat->status == 200);
    CHECK(chat->body.find("Hello from HLAW Mirage.") != std::string::npos);
}

TEST_CASE_METHOD(MirageFixture, "Mirage streams valid SSE completion boundaries", "[component][mirage][stream]") {
    const auto response = post_chat(R"({"model":"mock-stream","messages":[],"stream":true})");
    REQUIRE(response);
    CHECK(response->status == 200);
    CHECK(response->get_header_value("Content-Type").find("text/event-stream") == 0);
    CHECK(response->body.find("data: [DONE]\n\n") != std::string::npos);
    CHECK(response->body.find(R"("finish_reason":"stop")") != std::string::npos);
}

TEST_CASE_METHOD(MirageFixture, "Mirage retry sequences are deterministic and resettable", "[component][mirage][retry]") {
    constexpr auto request = R"({"model":"mock-retry","messages":[]})";
    const auto first = post_chat(request);
    const auto second = post_chat(request);
    const auto third = post_chat(request);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(third);
    CHECK(first->status == 429);
    CHECK(second->status == 429);
    CHECK(third->status == 200);

    const auto reset = client_.Post("/_mock/reset", "{}", "application/json");
    REQUIRE(reset);
    CHECK(reset->status == 200);

    const auto after_reset = post_chat(request);
    REQUIRE(after_reset);
    CHECK(after_reset->status == 429);
}
