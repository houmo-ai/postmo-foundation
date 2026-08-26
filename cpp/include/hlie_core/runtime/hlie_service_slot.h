#ifndef HLIE_CORE_SERVICE_SLOT_H
#define HLIE_CORE_SERVICE_SLOT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Thread-safe ownership slot for one local service kind.
//
// A load operation must first acquire a ticket. Expensive service creation and
// initialization happen outside the slot lock, then publish() atomically makes
// the service visible if the ticket is still current. detach() invalidates any
// in-flight ticket so a late loader cannot republish a service after unload.
template <typename ServiceT>
class HLIEServiceSlot {
public:
    struct LoadTicket {
        std::uint64_t epoch = 0;
    };

    [[nodiscard]] std::optional<LoadTicket> begin_load() {
        std::lock_guard lock(mutex_);
        if (service_ || loading_) {
            return std::nullopt;
        }

        loading_ = true;
        return LoadTicket{++epoch_};
    }

    [[nodiscard]] bool publish(const LoadTicket& ticket, std::shared_ptr<ServiceT> service) {
        if (!service) {
            return false;
        }

        std::lock_guard lock(mutex_);
        if (!loading_ || service_ || ticket.epoch != epoch_) {
            return false;
        }

        service_ = std::move(service);
        loading_ = false;
        return true;
    }

    [[nodiscard]] bool fail_load(const LoadTicket& ticket) {
        std::lock_guard lock(mutex_);
        if (!loading_ || ticket.epoch != epoch_) {
            return false;
        }

        loading_ = false;
        return true;
    }

    [[nodiscard]] std::shared_ptr<ServiceT> snapshot() const {
        std::lock_guard lock(mutex_);
        return service_;
    }

    [[nodiscard]] std::shared_ptr<ServiceT> detach() {
        std::lock_guard lock(mutex_);
        ++epoch_;
        loading_ = false;
        return std::exchange(service_, {});
    }

    [[nodiscard]] bool is_loading() const {
        std::lock_guard lock(mutex_);
        return loading_;
    }

    [[nodiscard]] std::uint64_t epoch() const {
        std::lock_guard lock(mutex_);
        return epoch_;
    }

private:
    mutable std::mutex mutex_;
    std::shared_ptr<ServiceT> service_;
    bool loading_ = false;
    std::uint64_t epoch_ = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_SERVICE_SLOT_H
