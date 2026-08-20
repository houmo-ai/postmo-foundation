#include "hlie_core/runtime/hlie_model_runtime_registry.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <limits>
#include <utility>

HLIE_NAMESPACE_BEGIN

HLIEModelRuntimeRegistry::HLIEModelRuntimeRegistry(
    LaunchTokenFactory launch_token_factory)
    : launch_token_factory_(std::move(launch_token_factory)) {}

std::string HLIEModelRuntimeRegistry::normalize_model_name(const std::string& name) {
    const auto first = std::find_if_not(
        name.begin(), name.end(), [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(
        name.rbegin(), name.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) {
        return {};
    }

    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized;
}

void HLIEModelRuntimeRegistry::normalize_registration(Registration& registration) {
    auto& snapshot = registration.snapshot;
    if (snapshot.identity.display_name.empty()) {
        snapshot.identity.display_name = snapshot.identity.model_key;
    }
    if (snapshot.failure_code != 0) {
        snapshot.state = HLIEModelLifecycleState::Failed;
    }
    if (snapshot.state == HLIEModelLifecycleState::Unloaded ||
        snapshot.state == HLIEModelLifecycleState::Failed) {
        registration.launch_token.clear();
    }
}

HLIEModelRuntimeRegistry::Record* HLIEModelRuntimeRegistry::find_record_locked(
    const std::string& model_name) {
    return const_cast<Record*>(std::as_const(*this).find_record_locked(model_name));
}

const HLIEModelRuntimeRegistry::Record* HLIEModelRuntimeRegistry::find_record_locked(
    const std::string& model_name) const {
    if (const auto it = records_.find(model_name); it != records_.end()) {
        return &it->second;
    }

    const std::string normalized_name = normalize_model_name(model_name);
    if (normalized_name.empty()) {
        return nullptr;
    }

    const Record* matched = nullptr;
    int matched_rank = 2;
    int matched_state_rank = 2;
    bool ambiguous = false;
    for (const auto& [key, record] : records_) {
        const auto& identity = record.snapshot.identity;
        const bool key_matched = normalize_model_name(key) == normalized_name;
        const bool model_name_matched =
            normalize_model_name(identity.model_name) == normalized_name;
        const bool display_name_matched =
            normalize_model_name(identity.display_name) == normalized_name;
        if (!key_matched && !model_name_matched && !display_name_matched) {
            continue;
        }

        const int rank = key_matched ? 0 : 1;
        const int state_rank = hlie_model_state_is_ready(record.snapshot.state) ? 0 : 1;
        if (!matched || rank < matched_rank ||
            (rank == matched_rank && state_rank < matched_state_rank)) {
            matched = &record;
            matched_rank = rank;
            matched_state_rank = state_rank;
            ambiguous = false;
        } else if (rank == matched_rank && state_rank == matched_state_rank) {
            ambiguous = true;
        }
    }
    return ambiguous ? nullptr : matched;
}

void HLIEModelRuntimeRegistry::upsert(Registration registration) {
    normalize_registration(registration);
    if (registration.snapshot.identity.model_key.empty()) {
        return;
    }

    std::lock_guard lk(mutex_);
    if (hlie_model_state_is_ready(registration.snapshot.state) &&
        (registration.snapshot.generation == 0 || registration.launch_token.empty())) {
        registration.snapshot.generation = registration.snapshot.generation ==
                std::numeric_limits<HLIEModelGeneration>::max()
            ? 1
            : registration.snapshot.generation + 1;
        registration.launch_token = launch_token_factory_ ? launch_token_factory_() : std::string{};
        if (registration.launch_token.empty()) {
            registration.snapshot.state = HLIEModelLifecycleState::Failed;
            registration.snapshot.failure_code = -1;
            registration.snapshot.port = 0;
        }
    }

    const std::string key = registration.snapshot.identity.model_key;
    records_[key] = Record{std::move(registration.snapshot),
                           std::move(registration.launch_token)};
}

bool HLIEModelRuntimeRegistry::register_if_absent(Registration registration) {
    normalize_registration(registration);
    const std::string key = registration.snapshot.identity.model_key;
    if (key.empty()) {
        return false;
    }

    std::lock_guard lk(mutex_);
    if (records_.find(key) != records_.end()) {
        return false;
    }
    if (hlie_model_state_is_ready(registration.snapshot.state) &&
        (registration.snapshot.generation == 0 || registration.launch_token.empty())) {
        registration.snapshot.generation = registration.snapshot.generation ==
                std::numeric_limits<HLIEModelGeneration>::max()
            ? 1
            : registration.snapshot.generation + 1;
        registration.launch_token = launch_token_factory_ ? launch_token_factory_() : std::string{};
        if (registration.launch_token.empty()) {
            registration.snapshot.state = HLIEModelLifecycleState::Failed;
            registration.snapshot.failure_code = -1;
            registration.snapshot.port = 0;
        }
    }
    return records_.try_emplace(
        key, Record{std::move(registration.snapshot), std::move(registration.launch_token)}).second;
}

bool HLIEModelRuntimeRegistry::contains(const std::string& model_key) const {
    std::lock_guard lk(mutex_);
    return records_.find(model_key) != records_.end();
}

std::optional<std::string> HLIEModelRuntimeRegistry::resolve_model_key(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const Record* record = find_record_locked(model_name);
    return record ? std::optional<std::string>{record->snapshot.identity.model_key}
                  : std::nullopt;
}

std::optional<HLIEModelRuntimeSnapshot> HLIEModelRuntimeRegistry::get_snapshot(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const Record* record = find_record_locked(model_name);
    return record ? std::optional<HLIEModelRuntimeSnapshot>{record->snapshot} : std::nullopt;
}

std::vector<HLIEModelRuntimeSnapshot> HLIEModelRuntimeRegistry::get_all_snapshots() const {
    std::lock_guard lk(mutex_);
    std::vector<HLIEModelRuntimeSnapshot> snapshots;
    snapshots.reserve(records_.size());
    for (const auto& [_, record] : records_) {
        snapshots.push_back(record.snapshot);
    }
    return snapshots;
}

bool HLIEModelRuntimeRegistry::update_identity(const std::string& model_name,
                                               HLIEModelIdentity identity) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    identity.model_key = record->snapshot.identity.model_key;
    if (identity.display_name.empty()) {
        identity.display_name = identity.model_key;
    }
    record->snapshot.identity = std::move(identity);
    return true;
}

void HLIEModelRuntimeRegistry::record_access(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return;
    }
    record->snapshot.last_access_time = std::time(nullptr);
    ++record->snapshot.hit_count;
}

std::optional<HLIEModelLaunchTicket> HLIEModelRuntimeRegistry::begin_request(
    const std::string& model_name) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record || !hlie_model_state_is_ready(record->snapshot.state) ||
        !record->snapshot.evictable || record->snapshot.generation == 0 ||
        record->launch_token.empty()) {
        return std::nullopt;
    }

    record->snapshot.last_access_time = std::time(nullptr);
    ++record->snapshot.hit_count;
    ++record->snapshot.active_requests;
    return HLIEModelLaunchTicket{record->snapshot.identity.model_key,
                                 record->snapshot.generation,
                                 record->launch_token};
}

void HLIEModelRuntimeRegistry::end_request(const HLIEModelLaunchTicket& ticket) {
    if (!ticket.valid()) {
        return;
    }
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return;
    }
    if (record->snapshot.active_requests > 0) {
        --record->snapshot.active_requests;
    }
}

bool HLIEModelRuntimeRegistry::try_begin_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record || !hlie_model_state_is_ready(record->snapshot.state) ||
        record->snapshot.active_requests > 0 || !record->snapshot.evictable) {
        return false;
    }
    record->snapshot.evictable = false;
    return true;
}

void HLIEModelRuntimeRegistry::abort_eviction(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    if (Record* record = find_record_locked(model_name)) {
        record->snapshot.evictable = true;
    }
}

std::optional<HLIEModelLaunchTicket> HLIEModelRuntimeRegistry::begin_launch_locked(
    Record& record) {
    const std::string token = launch_token_factory_ ? launch_token_factory_() : std::string{};
    if (token.empty()) {
        return std::nullopt;
    }
    record.snapshot.generation = record.snapshot.generation ==
            std::numeric_limits<HLIEModelGeneration>::max()
        ? 1
        : record.snapshot.generation + 1;
    record.launch_token = token;
    record.snapshot.state = HLIEModelLifecycleState::Starting;
    record.snapshot.failure_code = 0;
    record.snapshot.evictable = false;
    return HLIEModelLaunchTicket{record.snapshot.identity.model_key,
                                 record.snapshot.generation,
                                 record.launch_token};
}

std::optional<HLIEModelLaunchTicket> HLIEModelRuntimeRegistry::try_begin_loading(
    const std::string& model_name) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record || hlie_model_state_protects_reservation(record->snapshot.state)) {
        return std::nullopt;
    }
    return begin_launch_locked(*record);
}

bool HLIEModelRuntimeRegistry::mark_loading(const std::string& model_name) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    return record && begin_launch_locked(*record).has_value();
}

std::optional<HLIEModelLaunchTicket> HLIEModelRuntimeRegistry::get_launch_ticket(
    const std::string& model_name) const {
    std::lock_guard lk(mutex_);
    const Record* record = find_record_locked(model_name);
    if (!record || record->snapshot.generation == 0 || record->launch_token.empty()) {
        return std::nullopt;
    }
    return HLIEModelLaunchTicket{record->snapshot.identity.model_key,
                                 record->snapshot.generation,
                                 record->launch_token};
}

bool HLIEModelRuntimeRegistry::matches_current_locked(
    const Record& record, const HLIEModelLaunchTicket& ticket) const {
    return ticket.valid() && record.snapshot.identity.model_key == ticket.model_key &&
           record.snapshot.generation == ticket.generation &&
           record.launch_token == ticket.token;
}

bool HLIEModelRuntimeRegistry::is_current_launch(const HLIEModelLaunchTicket& ticket) const {
    if (!ticket.valid()) {
        return false;
    }
    std::lock_guard lk(mutex_);
    const Record* record = find_record_locked(ticket.model_key);
    return record && matches_current_locked(*record, ticket);
}

void HLIEModelRuntimeRegistry::mark_ready_locked(Record& record, const int port) {
    record.snapshot.state = HLIEModelLifecycleState::Ready;
    record.snapshot.port = port;
    record.snapshot.failure_code = 0;
    record.snapshot.last_load_time = std::time(nullptr);
    record.snapshot.evictable = true;
}

bool HLIEModelRuntimeRegistry::mark_ready(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    if (record->snapshot.generation == 0 || record->launch_token.empty()) {
        if (!begin_launch_locked(*record)) {
            return false;
        }
    }
    mark_ready_locked(*record, port);
    return true;
}

void HLIEModelRuntimeRegistry::mark_unloaded_locked(Record& record, const int port) {
    record.snapshot.state = HLIEModelLifecycleState::Unloaded;
    record.snapshot.port = port;
    record.snapshot.failure_code = 0;
    record.snapshot.last_load_time = 0;
    record.snapshot.active_requests = 0;
    record.snapshot.evictable = true;
    record.launch_token.clear();
}

bool HLIEModelRuntimeRegistry::mark_unloaded(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    mark_unloaded_locked(*record, port);
    return true;
}

void HLIEModelRuntimeRegistry::mark_failed_locked(Record& record, const int failure_code) {
    record.snapshot.state = HLIEModelLifecycleState::Failed;
    record.snapshot.port = 0;
    record.snapshot.failure_code = failure_code == 0 ? -1 : failure_code;
    record.snapshot.last_load_time = 0;
    record.snapshot.active_requests = 0;
    record.snapshot.evictable = true;
    record.launch_token.clear();
}

void HLIEModelRuntimeRegistry::mark_orphaned_locked(
    Record& record, const int failure_code) {
    record.snapshot.state = HLIEModelLifecycleState::Orphaned;
    record.snapshot.failure_code = failure_code == 0 ? -1 : failure_code;
    // Preserve port, launch token and request accounting until ownership is
    // reconciled. A new launch must not replace an unconfirmed runtime.
    record.snapshot.evictable = false;
}

bool HLIEModelRuntimeRegistry::mark_failed(const std::string& model_name,
                                           const int failure_code) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    mark_failed_locked(*record, failure_code);
    return true;
}

bool HLIEModelRuntimeRegistry::update_state(const std::string& model_name,
                                           const HLIEModelLifecycleState state,
                                           const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    record->snapshot.state = state;
    record->snapshot.port = port;
    return true;
}

bool HLIEModelRuntimeRegistry::update_port(const std::string& model_name, const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(model_name);
    if (!record) {
        return false;
    }
    record->snapshot.port = port;
    return true;
}

bool HLIEModelRuntimeRegistry::mark_ready_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return false;
    }
    mark_ready_locked(*record, port);
    return true;
}

bool HLIEModelRuntimeRegistry::mark_unloaded_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return false;
    }
    mark_unloaded_locked(*record, port);
    return true;
}

bool HLIEModelRuntimeRegistry::mark_failed_if_current(
    const HLIEModelLaunchTicket& ticket, const int failure_code) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return false;
    }
    mark_failed_locked(*record, failure_code);
    return true;
}

bool HLIEModelRuntimeRegistry::mark_orphaned_if_current(
    const HLIEModelLaunchTicket& ticket, const int failure_code) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return false;
    }
    mark_orphaned_locked(*record, failure_code);
    return true;
}

bool HLIEModelRuntimeRegistry::update_state_if_current(
    const HLIEModelLaunchTicket& ticket,
    const HLIEModelLifecycleState state,
    const int port) {
    std::lock_guard lk(mutex_);
    Record* record = find_record_locked(ticket.model_key);
    if (!record || !matches_current_locked(*record, ticket)) {
        return false;
    }
    record->snapshot.state = state;
    record->snapshot.port = port;
    return true;
}

HLIE_NAMESPACE_END
