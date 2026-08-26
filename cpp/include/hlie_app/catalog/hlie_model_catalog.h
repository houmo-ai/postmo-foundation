#ifndef HLIE_APP_MODEL_CATALOG_H
#define HLIE_APP_MODEL_CATALOG_H

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <server/server-models.h>

#include "hlie_app/hlie_model_info.h"
#include "hlie_core/catalog/hlie_model_catalog.h"

HLIE_NAMESPACE_BEGIN

struct HLIEModelCatalogEntry {
    HLIEModelSpec spec;
    server_model_meta meta;
    HLIEModelInfo model_info;
};

// App-owned static model catalog. It deliberately contains no lifecycle,
// generation, request-count, port, or process state.
class HLIEModelCatalog final : public IHLIEModelCatalog {
public:
    using EntryUpdater = std::function<void(HLIEModelCatalogEntry&)>;

    void upsert(HLIEModelCatalogEntry entry);
    [[nodiscard]] bool insert_if_absent(HLIEModelCatalogEntry entry);
    [[nodiscard]] bool contains(const std::string& model_key) const;
    [[nodiscard]] std::optional<HLIEModelCatalogEntry> get(
        const std::string& model_key) const;
    [[nodiscard]] bool update(const std::string& model_key,
                              const EntryUpdater& updater);

    [[nodiscard]] std::optional<HLIEModelSpec> resolve(
        const std::string& selector) const override;
    [[nodiscard]] std::vector<HLIEModelSpec> snapshots() const override;

private:
    static std::string normalize(std::string value);
    static void normalize_entry(HLIEModelCatalogEntry& entry);

    std::map<std::string, HLIEModelCatalogEntry> entries_;
    mutable std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODEL_CATALOG_H
