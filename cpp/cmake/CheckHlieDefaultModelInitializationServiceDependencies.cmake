if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_DEFAULT_INITIALIZATION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/startup/hlie_default_model_initialization_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_default_model_initialization_service.cpp")
set(HLIE_DEFAULT_INITIALIZATION_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_service_manager"
        "hlie_model_service_factory"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "hlie_process_manager"
        "hlie_device_service"
        "server_http"
        "httplib"
        "nlohmann"
        "ordered_json"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_DEFAULT_INITIALIZATION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "Default model initialization source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_DEFAULT_INITIALIZATION_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Default model initialization boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "found_default_model"
        "found_sd3"
        "default_mmproj file does not exist"
        "Initializing default model service"
        "No default models found for slave instance")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns default model initialization: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEDefaultModelInitializationService>")
    message(FATAL_ERROR
            "Application composition root does not own DefaultModelInitializationService construction")
endif ()

message(STATUS "Default model initialization dependency and Workflow ownership checks passed")
