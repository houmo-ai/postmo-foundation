if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_RECONCILER_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_control_plane_reconciler.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_control_plane_reconciler.cpp")
set(HLIE_RECONCILER_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_registry"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_model_service"
        "singleton<"
        "server/"
        "httplib"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_RECONCILER_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ControlPlaneReconciler source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_RECONCILER_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ControlPlaneReconciler dependency boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

set(HLIE_WORKFLOW_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/hlie_workflow.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp")
foreach (HLIE_FILE IN LISTS HLIE_WORKFLOW_FILES)
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    foreach (HLIE_BANNED IN ITEMS
            "residency_ttl_thread_"
            "residency_ttl_stop_"
            "residency_ttl_loop"
            "reconcile_expired_pending_reservations")
        string(FIND "${HLIE_CONTENT}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Workflow still owns a Reconciler concern: ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

message(STATUS "ControlPlaneReconciler dependency and Workflow ownership checks passed")
