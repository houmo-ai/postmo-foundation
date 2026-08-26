if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_EMBEDDED_STARTUP_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/startup/hlie_embedded_startup_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_embedded_startup_service.cpp")
set(HLIE_EMBEDDED_STARTUP_BANNED_DEPENDENCIES
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
        "server_http"
        "httplib"
        "nlohmann"
        "ordered_json"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_EMBEDDED_STARTUP_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "EmbeddedStartupService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_EMBEDDED_STARTUP_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "EmbeddedStartupService boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

set(HLIE_DEFAULT_PROVIDER_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/startup/hlie_default_embedded_startup_target_provider.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_default_embedded_startup_target_provider.cpp")
foreach (HLIE_FILE IN LISTS HLIE_DEFAULT_PROVIDER_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "Default embedded startup provider source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN ITEMS
            "hlie_workflow"
            "hlie_process_manager"
            "server_http"
            "httplib"
            "nlohmann"
            "singleton<"
            "tcim"
            "ggml")
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Default embedded startup provider boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "publish_initialized_runtime"
        "model_service_manager_.llm()->run()"
        "model_service_manager_.sd()->run()"
        "model_service_manager_.whisper()->run()"
        "model_service_manager_.cosyvoice()->run()"
        "model_service_manager_.qwen3_tts()->run()")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns embedded startup sequencing: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES "make_unique<HLIEEmbeddedStartupService>")
    message(FATAL_ERROR
            "Application composition root does not own EmbeddedStartupService construction")
endif ()
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "stop_and_reset_by_type\\(")
    message(FATAL_ERROR
            "Application composition root does not provide unpublished embedded Service cleanup")
endif ()

message(STATUS "EmbeddedStartupService dependency and Workflow ownership checks passed")
