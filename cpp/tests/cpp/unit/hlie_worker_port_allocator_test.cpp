#include <catch_amalgamated.hpp>

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

#include "hlie_app/runtime/hlie_worker_port_allocator.h"

using namespace hm_HLIE;

namespace {

class OccupiedTcpPort {
public:
    OccupiedTcpPort() {
#ifdef _WIN32
        REQUIRE(WSAStartup(MAKEWORD(2, 2), &winsock_data_) == 0);
        socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        REQUIRE(socket_ != INVALID_SOCKET);
#else
        socket_ = socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(socket_ >= 0);
#endif
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = 0;
        REQUIRE(bind(socket_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
        REQUIRE(listen(socket_, 1) == 0);

        sockaddr_in bound{};
#ifdef _WIN32
        int length = sizeof(bound);
#else
        socklen_t length = sizeof(bound);
#endif
        REQUIRE(getsockname(socket_, reinterpret_cast<sockaddr*>(&bound), &length) == 0);
        port_ = ntohs(bound.sin_port);
    }

    ~OccupiedTcpPort() {
#ifdef _WIN32
        if (socket_ != INVALID_SOCKET) {
            closesocket(socket_);
        }
        WSACleanup();
#else
        if (socket_ >= 0) {
            close(socket_);
        }
#endif
    }

    [[nodiscard]] int port() const { return port_; }

private:
#ifdef _WIN32
    WSADATA winsock_data_{};
    SOCKET socket_ = INVALID_SOCKET;
#else
    int socket_ = -1;
#endif
    int port_ = 0;
};

} // namespace

TEST_CASE("Worker port allocator skips occupied candidates",
          "[worker-port][worker-supervisor]") {
    const auto assignment = HLIEWorkerPortAllocator::find_available(
        7901, 1, 8, [](const int port) { return port == 7904; });
    REQUIRE(assignment);
    CHECK(assignment->instance_id == 3);
    CHECK(assignment->port == 7904);
    CHECK(assignment->skipped_candidates == 2);
}

TEST_CASE("Worker port allocator respects TCP port range and attempt limit",
          "[worker-port][worker-supervisor]") {
    CHECK(HLIEWorkerPortAllocator::find_available(
        65534, 1, 1, [](const int) { return true; }));
    CHECK_FALSE(HLIEWorkerPortAllocator::find_available(
        65534, 2, 8, [](const int) { return true; }));
    CHECK_FALSE(HLIEWorkerPortAllocator::find_available(
        7901, 1, 0, [](const int) { return true; }));
}

TEST_CASE("TCP port probe detects a real occupied listener",
          "[worker-port][worker-supervisor][socket]") {
    OccupiedTcpPort occupied;
    REQUIRE(occupied.port() > 0);
    CHECK_FALSE(HLIEWorkerPortAllocator::is_tcp_port_available(occupied.port()));
    CHECK_FALSE(HLIEWorkerPortAllocator::is_tcp_port_available(0));
    CHECK_FALSE(HLIEWorkerPortAllocator::is_tcp_port_available(65536));
}
