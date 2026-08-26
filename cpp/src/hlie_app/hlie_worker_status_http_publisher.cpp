#include "hlie_app/http/hlie_worker_status_http_publisher.h"

#include <nlohmann/json.hpp>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

const char* status_to_string(const HLIEWorkerModelStatus status) {
    switch (status) {
    case HLIEWorkerModelStatus::Unloaded:
        return "unloaded";
    case HLIEWorkerModelStatus::Loading:
        return "loading";
    case HLIEWorkerModelStatus::Loaded:
        return "loaded";
    case HLIEWorkerModelStatus::Sleeping:
        return "sleeping";
    }
    return "unknown";
}

} // namespace

bool HLIEHttpWorkerStatusPublisher::publish(
    const HLIEWorkerStatusPublication& publication) {
    httplib::Client client("127.0.0.1", publication.router_port);
    client.set_connection_timeout(0, 200000);

    httplib::Request request;
    request.method = "POST";
    request.path = "/models/status";
    request.set_header("Content-Type", "application/json");

    nlohmann::json body;
    body["model"] = publication.model_key;
    body["port"] = publication.port;
    body["value"] = status_to_string(publication.status);
    if (publication.launch_ticket.valid()) {
        body["generation"] = publication.launch_ticket.generation;
        body["launch_token"] = publication.launch_ticket.token;
    }
    request.body = body.dump();

    LOG_INFO("notifying router server (port={}) that model {} at [{}] is ready",
             publication.router_port, publication.model_key,
             publication.port);
    const auto result = client.send(request);
    if (result.error() != httplib::Error::Success) {
        const auto error = httplib::to_string(result.error());
        LOG_ERROR("failed to notify router server: {}", error);
        return false;
    }
    if (result->status < 200 || result->status >= 300) {
        LOG_ERROR(
            "Router status update rejected: router_port={}, model={}, "
            "generation={}, http_status={}, response={}",
            publication.router_port, nlohmann::json(publication.model_key).dump(),
            publication.launch_ticket.generation, result->status,
            nlohmann::json(result->body).dump());
        return false;
    }
    return true;
}

HLIE_NAMESPACE_END
