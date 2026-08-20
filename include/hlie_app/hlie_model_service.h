#ifndef HLIE_MODEL_SERVICE_H
#define HLIE_MODEL_SERVICE_H

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hlie_core/resource/hlie_device_reservation_ledger.h"
#include "hlie_core/domain/hlie_cancellation_observation.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;

// Compatibility aliases keep backend request code stable while the metrics
// contract remains independent from this concrete app control type.
using HLIEQueuedRequestStopReason = HLIECancellationStopReason;
using HLIEQueuedRequestStopObservation = HLIECancellationObservation;

struct HLIEQueuedRequestControl {
    std::chrono::steady_clock::time_point deadline{};
    std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
    std::function<bool()> should_stop;

    [[nodiscard]] bool is_cancelled() const {
        if (cancelled && cancelled->load()) {
            return true;
        }
        if (should_stop && should_stop()) {
            cancel();
            return true;
        }
        return false;
    }

    [[nodiscard]] bool is_expired() const {
        return deadline.time_since_epoch().count() > 0 &&
            std::chrono::steady_clock::now() >= deadline;
    }

    void cancel() const {
        mark_stopped(HLIEQueuedRequestStopReason::CANCELLED,
                     std::chrono::steady_clock::now());
    }

    void expire() const {
        const auto requested_at = deadline.time_since_epoch().count() > 0
            ? deadline : std::chrono::steady_clock::now();
        mark_stopped(HLIEQueuedRequestStopReason::DEADLINE, requested_at);
    }

    [[nodiscard]] HLIEQueuedRequestStopObservation stop_observation() const {
        std::lock_guard lock(stop_state_->mutex);
        return {stop_state_->reason, stop_state_->requested_at};
    }

    // Refreshes callback/deadline state before an app adapter reports backend
    // completion to the neutral cancellation metrics contract.
    [[nodiscard]] HLIECancellationObservation observe_for_completion() const {
        (void)is_cancelled();
        if (is_expired()) {
            expire();
        }
        return stop_observation();
    }

private:
    struct StopState {
        std::mutex mutex;
        HLIEQueuedRequestStopReason reason = HLIEQueuedRequestStopReason::None;
        std::chrono::steady_clock::time_point requested_at{};
    };

    std::shared_ptr<StopState> stop_state_ = std::make_shared<StopState>();

    void mark_stopped(const HLIEQueuedRequestStopReason reason,
                      const std::chrono::steady_clock::time_point requested_at) const {
        {
            std::lock_guard lock(stop_state_->mutex);
            if (stop_state_->reason == HLIEQueuedRequestStopReason::None) {
                stop_state_->reason = reason;
                stop_state_->requested_at = requested_at;
            }
        }
        if (cancelled) {
            cancelled->store(true);
        }
    }
};

struct HLIEModelType {
    static constexpr auto LLM = "llm";
    static constexpr auto SD = "sd";
    static constexpr auto WHISPER = "whisper";
    static constexpr auto COSYVOICE = "cosyvoice";
    static constexpr auto TTS = "tts";
    static constexpr auto QWEN3_TTS = "qwen3_tts";
    static constexpr auto DOCUMENT = "document";

    // model real instance type
    static constexpr auto Qwen25 = "qwen2.5";
    static constexpr auto Qwen3 = "qwen3";
    static constexpr auto DeepSeek = "deepseek";
    static constexpr auto SD3 = "sd3";
};

// Backward-compatible aliases. Keep HLIEModelType as the single source of string literals.
inline const std::string LLM = HLIEModelType::LLM;
inline const std::string SD = HLIEModelType::SD;
inline const std::string WHISPER = HLIEModelType::WHISPER;

class HLIEModelInfo;

class IHLIEModelService {
public:
    IHLIEModelService() = default;
    virtual ~IHLIEModelService() = default;

    virtual int init(const std::string& model_dir, const HLIEModelInfo& model_info, const std::shared_ptr<HLIEDeviceService>& device_service) = 0;

    virtual void run() = 0;

    virtual void process() = 0;

    virtual void stop() = 0;

    virtual int get_state() = 0;

    [[nodiscard]] std::vector<int> get_devices() const { return devices_; }

protected:
    std::string model_path_;
    std::atomic<int> state_;
    std::vector<int> devices_;
    int acquired_mem_mb_ = 0;
    int queue_wait_timeout_ms_ = 30000;
    HLIEDeviceReservationId reservation_id_ = 0;
    // An externally supplied grant is owned by ResourceScheduler. The service
    // may validate/use it but must not commit or release it directly.
    bool reservation_owned_externally_ = false;
    bool is_multi_instance_ {false};
    std::shared_ptr<HLIEDeviceService> device_service_;
};

HLIE_NAMESPACE_END


#endif // HLIE_MODEL_SERVICE_H
