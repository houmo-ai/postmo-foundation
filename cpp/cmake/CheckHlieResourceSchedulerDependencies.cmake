if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(SCHEDULER_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/resource/hlie_resource_scheduler.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_resource_scheduler.cpp")
set(SCHEDULER_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_registry"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_model_service"
        "server/"
        "httplib"
        "tcim"
        "ggml")

foreach (SCHEDULER_FILE IN LISTS SCHEDULER_FILES)
    if (NOT EXISTS "${SCHEDULER_FILE}")
        message(FATAL_ERROR "ResourceScheduler source is missing: ${SCHEDULER_FILE}")
    endif ()
    file(READ "${SCHEDULER_FILE}" CONTENT)
    string(TOLOWER "${CONTENT}" CONTENT_LOWER)
    foreach (BANNED IN LISTS SCHEDULER_BANNED_DEPENDENCIES)
        string(FIND "${CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ResourceScheduler dependency boundary violation: ${SCHEDULER_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

set(WORKFLOW_FILE "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp")
file(READ "${WORKFLOW_FILE}" WORKFLOW_CONTENT)
set(WORKFLOW_BANNED_RESERVATION_OWNERSHIP
        "model_reservations_"
        "reservation_mutex_"
        "device_service_->reserve_mem"
        "device_service_->bind_reservation"
        "device_service_->commit_reservation"
        "device_service_->rollback_reservation"
        "device_service_->release_reservation"
        "device_service_->rollback_expired_pending_reservations"
        "device_service_->get_reservations"
        "device_service_->get_ledger_snapshots")
foreach (BANNED IN LISTS WORKFLOW_BANNED_RESERVATION_OWNERSHIP)
    string(FIND "${WORKFLOW_CONTENT}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns a reservation-ledger concern: ${BANNED}")
    endif ()
endforeach ()

message(STATUS "ResourceScheduler dependency and Workflow ownership checks passed")
