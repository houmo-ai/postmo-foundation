if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(CONTROLLER_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_model_controller.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_controller.cpp")

foreach (CONTROLLER_FILE IN LISTS CONTROLLER_FILES)
    file(READ "${CONTROLLER_FILE}" CONTENT)
    string(TOLOWER "${CONTENT}" CONTENT_LOWER)
    foreach (BANNED IN ITEMS
            "hlie_workflow"
            "hlie_model_service_manager"
            "hlie_process_manager"
            "hlie_device_service"
            "httplib"
            "tcim"
            "ggml")
        string(FIND "${CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelController dependency boundary violation: ${CONTROLLER_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_model_controller.h"
        CONTROLLER_HEADER)
string(FIND "${CONTROLLER_HEADER}" "HLIEModelRegistry registry_" REGISTRY_OWNER_INDEX)
if (REGISTRY_OWNER_INDEX EQUAL -1)
    message(FATAL_ERROR "ModelController must privately own the app Registry facade")
endif ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" WORKFLOW_SOURCE)
foreach (BANNED IN ITEMS
        "model_registry_"
        "HLIEWorkflow::on_runtime_event"
        "HLIEWorkflow::rollback_model_start"
        "HLIEWorkflow::stop_and_detach_runtime"
        "HLIEWorkflow::reconcile_runtime_publish_failure"
        "HLIEWorkflow::evict_model"
        "wait_for_model_load"
        "slave_cv_")
    string(FIND "${WORKFLOW_SOURCE}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow bypasses ModelController lifecycle ownership: ${BANNED}")
    endif ()
endforeach ()

file(READ
     "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_readiness_service.cpp"
     READINESS_SOURCE)
foreach (REQUIRED IN ITEMS
        "controller_.ensure_ready"
        "controller_.wait_until_ready")
    string(FIND "${READINESS_SOURCE}" "${REQUIRED}" MATCH_INDEX)
    if (MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "ModelReadinessService does not delegate load coalescing to ModelController: ${REQUIRED}")
    endif ()
endforeach ()

string(FIND "${WORKFLOW_SOURCE}"
        "static_cast<IHLIERuntimeEventSink&>(*model_controller_)"
        CONTROLLER_SINK_INDEX)
if (CONTROLLER_SINK_INDEX EQUAL -1)
    message(FATAL_ERROR "WorkerSupervisor events must enter ModelController")
endif ()

message(STATUS "ModelController dependency boundary check passed")
