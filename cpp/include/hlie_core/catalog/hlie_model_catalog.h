#ifndef HLIE_CORE_MODEL_CATALOG_H
#define HLIE_CORE_MODEL_CATALOG_H

#include <optional>
#include <string>
#include <vector>

#include "hlie_core/domain/hlie_model_spec.h"

HLIE_NAMESPACE_BEGIN

// Read-only core contract. Filesystem discovery and app compatibility payloads
// remain on the app side of this boundary.
class IHLIEModelCatalog {
public:
    virtual ~IHLIEModelCatalog() = default;

    [[nodiscard]] virtual std::optional<HLIEModelSpec> resolve(
        const std::string& selector) const = 0;
    [[nodiscard]] virtual std::vector<HLIEModelSpec> snapshots() const = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_MODEL_CATALOG_H
