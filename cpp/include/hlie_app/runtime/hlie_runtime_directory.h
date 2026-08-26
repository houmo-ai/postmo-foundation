#ifndef HLIE_APP_RUNTIME_DIRECTORY_H
#define HLIE_APP_RUNTIME_DIRECTORY_H

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_core/runtime/hlie_model_runtime.h"

HLIE_NAMESPACE_BEGIN

// Thread-safe handle directory. Registry remains the lifecycle source of
// truth; this class stores only generation-fenced runtime ownership handles.
class HLIERuntimeDirectory {
public:
    [[nodiscard]] HLIERuntimeIdentity allocate_embedded_identity(
        const HLIEModelLaunchTicket& ticket);

    [[nodiscard]] bool publish(std::shared_ptr<IHLIEModelRuntime> runtime);
    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> find(
        const HLIEModelLaunchTicket& ticket) const;
    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> find(
        const std::string& model_key) const;
    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> detach(
        const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> detach(
        const std::string& model_key);
    [[nodiscard]] std::vector<std::shared_ptr<IHLIEModelRuntime>> detach_all();
    [[nodiscard]] std::vector<HLIERuntimeHealth> health_snapshots() const;
    [[nodiscard]] std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<IHLIEModelRuntime>> runtimes_;
    std::atomic<HLIERuntimeId> next_embedded_runtime_id_{1};
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_RUNTIME_DIRECTORY_H
