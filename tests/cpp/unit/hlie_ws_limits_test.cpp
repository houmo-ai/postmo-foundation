#include "hlie_app/hlie_ws_limits.h"

#include <catch_amalgamated.hpp>

using namespace hm_HLIE;

TEST_CASE("Realtime websocket message limit preserves the documented boundary", "[websocket][realtime]") {
    CHECK_FALSE(realtimeWsMessageExceedsLimit(REALTIME_WS_MESSAGE_SIZE_LIMIT));
    CHECK(realtimeWsMessageExceedsLimit(REALTIME_WS_MESSAGE_SIZE_LIMIT + 1U));
}

TEST_CASE("Realtime websocket transport limit leaves room for an application error", "[websocket][realtime]") {
    CHECK(REALTIME_WS_TRANSPORT_PAYLOAD_LIMIT > REALTIME_WS_MESSAGE_SIZE_LIMIT);

    // A 16 MiB raw audio chunk expands by roughly 4/3 when base64 encoded.
    // It must still reach the application callback so an error event can be sent.
    constexpr std::size_t json_envelope_bytes = 47U;
    constexpr std::size_t encoded_audio_bytes =
        4U * ((REALTIME_WS_MESSAGE_SIZE_LIMIT + 2U) / 3U);
    CHECK(json_envelope_bytes + encoded_audio_bytes <= REALTIME_WS_TRANSPORT_PAYLOAD_LIMIT);
}
