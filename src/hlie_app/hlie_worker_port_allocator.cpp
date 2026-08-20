#include "hlie_app/runtime/hlie_worker_port_allocator.h"

#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

HLIE_NAMESPACE_BEGIN

std::optional<HLIEWorkerPortAssignment> HLIEWorkerPortAllocator::find_available(
    const int master_port, const int first_instance_id, const int max_attempts,
    const PortProbe& probe) {
    if (master_port <= 0 || master_port > 65535 || first_instance_id <= 0 ||
        max_attempts <= 0 || !probe) {
        return std::nullopt;
    }

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        const auto instance_id = static_cast<long long>(first_instance_id) + attempt;
        const auto port = static_cast<long long>(master_port) + instance_id;
        if (instance_id > 65535 || port > 65535) {
            break;
        }
        if (probe(static_cast<int>(port))) {
            return HLIEWorkerPortAssignment{
                static_cast<int>(instance_id), static_cast<int>(port), attempt};
        }
    }
    return std::nullopt;
}

bool HLIEWorkerPortAllocator::is_tcp_port_available(const int port) {
    if (port <= 0 || port > 65535) {
        return false;
    }

#ifdef _WIN32
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        return false;
    }
    const SOCKET socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_handle == INVALID_SOCKET) {
        WSACleanup();
        return false;
    }
    BOOL exclusive = TRUE;
    (void)setsockopt(socket_handle, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                     reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
#else
    const int socket_handle = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_handle < 0) {
        return false;
    }
#endif

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    const bool available = bind(socket_handle,
                                reinterpret_cast<const sockaddr*>(&address),
                                sizeof(address)) == 0;

#ifdef _WIN32
    closesocket(socket_handle);
    WSACleanup();
#else
    close(socket_handle);
#endif
    return available;
}

HLIE_NAMESPACE_END
