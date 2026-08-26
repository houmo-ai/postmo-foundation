#include "hlie_app/catalog/hlie_model_catalog.h"

#include <algorithm>
#include <cctype>
#include <utility>

HLIE_NAMESPACE_BEGIN

void HLIEModelCatalog::upsert(HLIEModelCatalogEntry entry) {
    normalize_entry(entry);
    std::lock_guard lock(mutex_);
    entries_[entry.spec.identity.model_key] = std::move(entry);
}

bool HLIEModelCatalog::insert_if_absent(HLIEModelCatalogEntry entry) {
    normalize_entry(entry);
    std::lock_guard lock(mutex_);
    const auto key = entry.spec.identity.model_key;
    if (entries_.find(key) != entries_.end()) {
        return false;
    }
    entries_.emplace(key, std::move(entry));
    return true;
}

bool HLIEModelCatalog::contains(const std::string& model_key) const {
    std::lock_guard lock(mutex_);
    return entries_.find(model_key) != entries_.end();
}

std::optional<HLIEModelCatalogEntry> HLIEModelCatalog::get(
    const std::string& model_key) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(model_key);
    return it == entries_.end()
        ? std::nullopt
        : std::optional<HLIEModelCatalogEntry>{it->second};
}

bool HLIEModelCatalog::update(const std::string& model_key,
                              const EntryUpdater& updater) {
    if (!updater) {
        return false;
    }
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(model_key);
    if (it == entries_.end()) {
        return false;
    }
    updater(it->second);
    normalize_entry(it->second);
    return true;
}

std::optional<HLIEModelSpec> HLIEModelCatalog::resolve(
    const std::string& selector) const {
    const auto normalized = normalize(selector);
    if (normalized.empty()) {
        return std::nullopt;
    }
    std::lock_guard lock(mutex_);
    for (const auto& [key, entry] : entries_) {
        const auto& identity = entry.spec.identity;
        if (normalize(key) == normalized ||
            normalize(identity.model_name) == normalized ||
            normalize(identity.display_name) == normalized) {
            return entry.spec;
        }
    }
    return std::nullopt;
}

std::vector<HLIEModelSpec> HLIEModelCatalog::snapshots() const {
    std::lock_guard lock(mutex_);
    std::vector<HLIEModelSpec> result;
    result.reserve(entries_.size());
    for (const auto& [_, entry] : entries_) {
        result.push_back(entry.spec);
    }
    return result;
}

std::string HLIEModelCatalog::normalize(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](const unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) {
        return {};
    }
    std::string result(first, last);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](const unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return result;
}

void HLIEModelCatalog::normalize_entry(HLIEModelCatalogEntry& entry) {
    if (entry.spec.identity.model_key.empty()) {
        entry.spec.identity.model_key = entry.model_info.model_key_;
    }
    entry.model_info.model_key_ = entry.spec.identity.model_key;
    const auto display_name = entry.meta.name.empty()
        ? entry.spec.identity.model_key
        : entry.meta.name;
    entry.spec = entry.model_info.to_model_spec();
    entry.spec.identity.model_key = entry.model_info.model_key_;
    entry.spec.identity.display_name = display_name;
    if (entry.meta.name.empty()) {
        entry.meta.name = entry.spec.identity.model_key;
    }
    // Runtime values are projected by ModelRegistry at read time.
    entry.meta.status = SERVER_MODEL_STATUS_UNLOADED;
    entry.meta.port = 0;
    entry.meta.exit_code = 0;
}

HLIE_NAMESPACE_END
