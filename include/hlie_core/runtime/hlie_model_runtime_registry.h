#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"
#include "hlie_core/domain/hlie_model_runtime_snapshot.h"

HLIE_NAMESPACE_BEGIN

// Mutable lifecycle facts owned by the control plane. Catalog metadata,
// protocol DTOs and concrete model-service objects deliberately stay outside.
class HLIEModelRuntimeRegistry {
public:
    struct Registration {
        HLIEModelRuntimeSnapshot snapshot;
        // The token is accepted only at the mutation boundary and is never
        // published through the general-purpose snapshot API.
        std::string launch_token;
    };

    using LaunchTokenFactory = std::function<std::string()>;

    explicit HLIEModelRuntimeRegistry(LaunchTokenFactory launch_token_factory);

    void upsert(Registration registration);
    [[nodiscard]] bool register_if_absent(Registration registration);
    [[nodiscard]] bool contains(const std::string& model_key) const;

    [[nodiscard]] std::optional<std::string> resolve_model_key(
        const std::string& model_name) const;
    [[nodiscard]] std::optional<HLIEModelRuntimeSnapshot> get_snapshot(
        const std::string& model_name) const;
    [[nodiscard]] std::vector<HLIEModelRuntimeSnapshot> get_all_snapshots() const;

    [[nodiscard]] bool update_identity(const std::string& model_name,
                                       HLIEModelIdentity identity);
    void record_access(const std::string& model_name);
    [[nodiscard]] std::optional<HLIEModelLaunchTicket> begin_request(
        const std::string& model_name);
    void end_request(const HLIEModelLaunchTicket& ticket);

    [[nodiscard]] bool try_begin_eviction(const std::string& model_name);
    void abort_eviction(const std::string& model_name);

    [[nodiscard]] std::optional<HLIEModelLaunchTicket> try_begin_loading(
        const std::string& model_name);
    [[nodiscard]] bool mark_loading(const std::string& model_name);
    [[nodiscard]] std::optional<HLIEModelLaunchTicket> get_launch_ticket(
        const std::string& model_name) const;
    [[nodiscard]] bool is_current_launch(const HLIEModelLaunchTicket& ticket) const;

    [[nodiscard]] bool mark_ready(const std::string& model_name, int port);
    [[nodiscard]] bool mark_unloaded(const std::string& model_name, int port = 0);
    [[nodiscard]] bool mark_failed(const std::string& model_name, int failure_code);
    [[nodiscard]] bool update_state(const std::string& model_name,
                                    HLIEModelLifecycleState state,
                                    int port);
    [[nodiscard]] bool update_port(const std::string& model_name, int port);

    [[nodiscard]] bool mark_ready_if_current(const HLIEModelLaunchTicket& ticket,
                                             int port);
    [[nodiscard]] bool mark_unloaded_if_current(const HLIEModelLaunchTicket& ticket,
                                                int port = 0);
    [[nodiscard]] bool mark_failed_if_current(const HLIEModelLaunchTicket& ticket,
                                              int failure_code);
    [[nodiscard]] bool mark_orphaned_if_current(
        const HLIEModelLaunchTicket& ticket, int failure_code);
    [[nodiscard]] bool update_state_if_current(const HLIEModelLaunchTicket& ticket,
                                               HLIEModelLifecycleState state,
                                               int port);

private:
    struct Record {
        HLIEModelRuntimeSnapshot snapshot;
        std::string launch_token;
    };

    static std::string normalize_model_name(const std::string& name);
    static void normalize_registration(Registration& registration);
    Record* find_record_locked(const std::string& model_name);
    const Record* find_record_locked(const std::string& model_name) const;
    [[nodiscard]] bool matches_current_locked(const Record& record,
                                              const HLIEModelLaunchTicket& ticket) const;
    [[nodiscard]] std::optional<HLIEModelLaunchTicket> begin_launch_locked(Record& record);
    static void mark_ready_locked(Record& record, int port);
    static void mark_unloaded_locked(Record& record, int port);
    static void mark_failed_locked(Record& record, int failure_code);
    static void mark_orphaned_locked(Record& record, int failure_code);

    LaunchTokenFactory launch_token_factory_;
    std::map<std::string, Record> records_;
    mutable std::mutex mutex_;
};

HLIE_NAMESPACE_END
