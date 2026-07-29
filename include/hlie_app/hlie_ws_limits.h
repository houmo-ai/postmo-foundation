#ifndef HLIE_WS_LIMITS_H
#define HLIE_WS_LIMITS_H

#include <cstddef>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Keep the public limit below the transport hard limit so the application can
// return a protocol error instead of letting uWebSockets drop the TCP socket.
inline constexpr std::size_t REALTIME_WS_MESSAGE_SIZE_LIMIT = 16U * 1024U * 1024U;
inline constexpr unsigned int REALTIME_WS_TRANSPORT_PAYLOAD_LIMIT = 32U * 1024U * 1024U;

[[nodiscard]] constexpr bool realtimeWsMessageExceedsLimit(const std::size_t payload_size) noexcept {
    return payload_size > REALTIME_WS_MESSAGE_SIZE_LIMIT;
}

static_assert(REALTIME_WS_TRANSPORT_PAYLOAD_LIMIT > REALTIME_WS_MESSAGE_SIZE_LIMIT);

HLIE_NAMESPACE_END

#endif // HLIE_WS_LIMITS_H
