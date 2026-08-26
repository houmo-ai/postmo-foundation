if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_WORKER_CONTROL_APPLICATION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_worker_control_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_worker_control_application_service.cpp")
set(HLIE_WORKER_CONTROL_APPLICATION_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_service_manager"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_WORKER_CONTROL_APPLICATION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "Worker control application source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_WORKER_CONTROL_APPLICATION_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Worker control application boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "HLIEWorkflow::handle_get_status"
        "HLIEWorkflow::handle_post_status"
        "HLIEWorkflow::handle_worker_shutdown"
        "Received model status update")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns Worker control HTTP orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEWorkerControlApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own WorkerControlApplicationService construction")
endif ()

message(STATUS "WorkerControlApplicationService dependency and Workflow ownership checks passed")
