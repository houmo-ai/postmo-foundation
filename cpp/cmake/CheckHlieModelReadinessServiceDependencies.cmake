if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_READINESS_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_model_readiness_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_readiness_service.cpp")
set(HLIE_READINESS_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_model_service_manager"
        "hlie_model_service_factory"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "singleton<"
        "server_http"
        "httplib"
        "nlohmann"
        "ordered_json"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_READINESS_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ModelReadinessService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_READINESS_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelReadinessService boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "auto_load_tasks_"
        "model_controller_->ensure_ready("
        "model_controller_->wait_until_ready("
        "readiness_task_launch_failed"
        "workflow_draining_before_auto_load")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns auto-load readiness coordination: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ
     "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES "make_unique<HLIEModelReadinessService>")
    message(FATAL_ERROR
            "Application composition root does not own ModelReadinessService construction")
endif ()

message(STATUS "ModelReadinessService dependency and Workflow ownership checks passed")
