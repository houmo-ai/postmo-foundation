#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class IHLIEModelService;

// App-side bridge from a stable core backend id to concrete HOUMO service
// construction. SDK/service types remain on this side of the dependency line.
class IHLIEBackendAdapter {
public:
    virtual ~IHLIEBackendAdapter() = default;

    [[nodiscard]] virtual std::string backend_id() const = 0;
    [[nodiscard]] virtual std::shared_ptr<IHLIEModelService>
    create_embedded_service() const = 0;
};

// Built during app composition and read-only during request processing.
class HLIE_API HLIEBackendAdapterRegistry {
public:
    [[nodiscard]] bool register_adapter(
        std::shared_ptr<const IHLIEBackendAdapter> adapter);
    [[nodiscard]] std::shared_ptr<const IHLIEBackendAdapter> find(
        const std::string& backend_id) const;
    [[nodiscard]] std::shared_ptr<IHLIEModelService> create_embedded_service(
        const std::string& backend_id) const;
    [[nodiscard]] std::size_t size() const noexcept { return adapters_.size(); }

private:
    struct Entry {
        std::string backend_id;
        std::shared_ptr<const IHLIEBackendAdapter> adapter;
    };

    static std::string normalize(std::string value);

    std::vector<Entry> adapters_;
};

HLIE_NAMESPACE_END
