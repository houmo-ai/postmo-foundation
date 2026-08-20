#ifndef HLIE_CORE_ADMISSION_CONTROLLER_H
#define HLIE_CORE_ADMISSION_CONTROLLER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEAdmissionSnapshot {
    std::size_t max_in_flight = 0;
    std::size_t in_flight = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
};

class HLIEAdmissionController {
public:
    class Permit {
    public:
        Permit() = default;
        ~Permit() { reset(); }
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        Permit(Permit&& other) noexcept : owner_(other.owner_) { other.owner_ = nullptr; }
        Permit& operator=(Permit&& other) noexcept {
            if (this != &other) {
                reset();
                owner_ = other.owner_;
                other.owner_ = nullptr;
            }
            return *this;
        }
        [[nodiscard]] bool active() const { return owner_ != nullptr; }
        void reset() {
            if (owner_) {
                owner_->release();
                owner_ = nullptr;
            }
        }

    private:
        friend class HLIEAdmissionController;
        explicit Permit(HLIEAdmissionController* owner) : owner_(owner) {}
        HLIEAdmissionController* owner_ = nullptr;
    };

    explicit HLIEAdmissionController(const std::size_t max_in_flight = 1)
        : max_in_flight_(std::max<std::size_t>(1, max_in_flight)) {}

    void configure(const std::size_t max_in_flight) {
        std::lock_guard lock(mutex_);
        max_in_flight_ = std::max<std::size_t>(1, max_in_flight);
    }

    [[nodiscard]] std::optional<Permit> try_acquire() {
        std::lock_guard lock(mutex_);
        if (in_flight_ >= max_in_flight_) {
            ++rejected_;
            return std::nullopt;
        }
        ++in_flight_;
        ++accepted_;
        return Permit(this);
    }

    [[nodiscard]] HLIEAdmissionSnapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {max_in_flight_, in_flight_, accepted_, rejected_};
    }

private:
    void release() {
        std::lock_guard lock(mutex_);
        if (in_flight_ > 0) {
            --in_flight_;
        }
    }

    mutable std::mutex mutex_;
    std::size_t max_in_flight_ = 1;
    std::size_t in_flight_ = 0;
    std::uint64_t accepted_ = 0;
    std::uint64_t rejected_ = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_ADMISSION_CONTROLLER_H
