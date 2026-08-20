if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_OBSERVABILITY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/observability/hlie_observability_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/observability/hlie_observability_application_service.cpp")
set(HLIE_OBSERVABILITY_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_resource_scheduler"
        "hlie_runtime_directory"
        "hlie_worker_supervisor"
        "hlie_inference_dispatcher"
        "hlie_control_plane_reconciler"
        "hlie_model_discovery_service"
        "hlie_admission_controller"
        "hlie_cancellation_metrics"
        "hlie_model_service_manager"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_OBSERVABILITY_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "Observability application source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_OBSERVABILITY_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Observability application boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "HLIEWorkflow::handle_health"
        "HLIEWorkflow::handle_live"
        "HLIEWorkflow::handle_ready"
        "HLIEWorkflow::handle_metrics"
        "hlie_models_loaded")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns observability DTO rendering: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEObservabilityApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own ObservabilityApplicationService construction")
endif ()

message(STATUS "ObservabilityApplicationService dependency and Workflow ownership checks passed")
