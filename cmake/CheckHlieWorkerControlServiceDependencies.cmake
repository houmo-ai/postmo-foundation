if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_WORKER_CONTROL_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_worker_control_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_worker_control_service.cpp")
set(HLIE_WORKER_CONTROL_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_process_manager"
        "hlie_model_service_manager"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "server_http"
        "httplib"
        "nlohmann"
        "ordered_json"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_WORKER_CONTROL_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "WorkerControlService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_WORKER_CONTROL_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "WorkerControlService boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES "make_unique<HLIEWorkerControlService>")
    message(FATAL_ERROR
            "Application composition root does not own WorkerControlService construction")
endif ()

message(STATUS "WorkerControlService dependency and Workflow ownership checks passed")
