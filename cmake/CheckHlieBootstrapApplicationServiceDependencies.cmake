if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_BOOTSTRAP_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/startup/hlie_bootstrap_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_bootstrap_application_service.cpp")
set(HLIE_BOOTSTRAP_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_service_manager"
        "hlie_process_manager"
        "hlie_worker_supervisor"
        "hlie_device_service"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "server_http"
        "httplib"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_BOOTSTRAP_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "Bootstrap source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_BOOTSTRAP_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Bootstrap boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "generation_it"
        "one_time_launch_manifest"
        "worker_consumed"
        "Successfully parsed model launch manifest"
        "model_infos.emplace_back")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns bootstrap orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEBootstrapApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own BootstrapApplicationService construction")
endif ()

message(STATUS "BootstrapApplicationService dependency and Workflow ownership checks passed")
