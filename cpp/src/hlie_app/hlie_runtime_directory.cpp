#include "hlie_app/runtime/hlie_runtime_directory.h"

#include <utility>

HLIE_NAMESPACE_BEGIN

HLIERuntimeIdentity HLIERuntimeDirectory::allocate_embedded_identity(
    const HLIEModelLaunchTicket& ticket) {
    if (!ticket.valid()) {
        return {};
    }
    return {
        next_embedded_runtime_id_.fetch_add(1),
        HLIERuntimeKind::Embedded,
        ticket,
        0,
        0,
    };
}

bool HLIERuntimeDirectory::publish(std::shared_ptr<IHLIEModelRuntime> runtime) {
    if (!runtime) {
        return false;
    }
    const auto identity = runtime->identity();
    if (!identity.valid()) {
        return false;
    }

    std::lock_guard lock(mutex_);
    const auto current = runtimes_.find(identity.launch_ticket.model_key);
    if (current != runtimes_.end()) {
        const auto current_identity = current->second->identity();
        if (current_identity.launch_ticket.generation >=
            identity.launch_ticket.generation) {
            return current_identity.matches(identity);
        }
    }
    runtimes_.insert_or_assign(identity.launch_ticket.model_key, std::move(runtime));
    return true;
}

std::shared_ptr<IHLIEModelRuntime> HLIERuntimeDirectory::find(
    const HLIEModelLaunchTicket& ticket) const {
    if (!ticket.valid()) {
        return nullptr;
    }
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(ticket.model_key);
    if (it == runtimes_.end() ||
        !it->second->identity().launch_ticket.matches(ticket)) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<IHLIEModelRuntime> HLIERuntimeDirectory::find(
    const std::string& model_key) const {
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(model_key);
    return it == runtimes_.end() ? nullptr : it->second;
}

std::shared_ptr<IHLIEModelRuntime> HLIERuntimeDirectory::detach(
    const HLIEModelLaunchTicket& ticket) {
    if (!ticket.valid()) {
        return nullptr;
    }
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(ticket.model_key);
    if (it == runtimes_.end() ||
        !it->second->identity().launch_ticket.matches(ticket)) {
        return nullptr;
    }
    auto runtime = std::move(it->second);
    runtimes_.erase(it);
    return runtime;
}

std::shared_ptr<IHLIEModelRuntime> HLIERuntimeDirectory::detach(
    const std::string& model_key) {
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(model_key);
    if (it == runtimes_.end()) {
        return nullptr;
    }
    auto runtime = std::move(it->second);
    runtimes_.erase(it);
    return runtime;
}

std::vector<std::shared_ptr<IHLIEModelRuntime>> HLIERuntimeDirectory::detach_all() {
    std::vector<std::shared_ptr<IHLIEModelRuntime>> runtimes;
    {
        std::lock_guard lock(mutex_);
        runtimes.reserve(runtimes_.size());
        for (auto& [_, runtime] : runtimes_) {
            runtimes.push_back(std::move(runtime));
        }
        runtimes_.clear();
    }
    return runtimes;
}

std::vector<HLIERuntimeHealth> HLIERuntimeDirectory::health_snapshots() const {
    std::vector<std::shared_ptr<IHLIEModelRuntime>> runtimes;
    {
        std::lock_guard lock(mutex_);
        runtimes.reserve(runtimes_.size());
        for (const auto& [_, runtime] : runtimes_) {
            runtimes.push_back(runtime);
        }
    }
    std::vector<HLIERuntimeHealth> result;
    result.reserve(runtimes.size());
    for (const auto& runtime : runtimes) {
        result.push_back(runtime->health());
    }
    return result;
}

std::size_t HLIERuntimeDirectory::size() const {
    std::lock_guard lock(mutex_);
    return runtimes_.size();
}

HLIE_NAMESPACE_END
