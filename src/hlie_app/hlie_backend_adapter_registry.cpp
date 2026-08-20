#include "hlie_app/runtime/hlie_backend_adapter_registry.h"

#include <algorithm>
#include <cctype>
#include <utility>

HLIE_NAMESPACE_BEGIN

std::string HLIEBackendAdapterRegistry::normalize(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
    value = first < last ? std::string(first, last) : std::string{};
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool HLIEBackendAdapterRegistry::register_adapter(
    std::shared_ptr<const IHLIEBackendAdapter> adapter) {
    if (!adapter) {
        return false;
    }
    const auto id = normalize(adapter->backend_id());
    if (id.empty() || find(id)) {
        return false;
    }
    adapters_.push_back({id, std::move(adapter)});
    return true;
}

std::shared_ptr<const IHLIEBackendAdapter> HLIEBackendAdapterRegistry::find(
    const std::string& backend_id) const {
    const auto id = normalize(backend_id);
    if (id.empty()) {
        return nullptr;
    }
    const auto it = std::find_if(
        adapters_.begin(), adapters_.end(),
        [&](const auto& entry) { return entry.backend_id == id; });
    return it == adapters_.end() ? nullptr : it->adapter;
}

std::shared_ptr<IHLIEModelService>
HLIEBackendAdapterRegistry::create_embedded_service(const std::string& backend_id) const {
    const auto adapter = find(backend_id);
    return adapter ? adapter->create_embedded_service() : nullptr;
}

HLIE_NAMESPACE_END
