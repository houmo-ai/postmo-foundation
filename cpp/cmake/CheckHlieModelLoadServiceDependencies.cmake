if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_MODEL_LOAD_SERVICE_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/control/hlie_model_load_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_load_service.cpp")
set(HLIE_MODEL_LOAD_SERVICE_BANNED_DEPENDENCIES
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

foreach (HLIE_FILE IN LISTS HLIE_MODEL_LOAD_SERVICE_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ModelLoadService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_MODEL_LOAD_SERVICE_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelLoadService dependency boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "resource_scheduler_->reserve("
        "model_controller_->load_model("
        "rollback_pending_reservation"
        "prepare_model_info_with_devices"
        "serialized_start_body"
        "server_http_req start_req"
        "parse_and_build_sd_model"
        "make_unique<HLIEModelLoadService>")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns model-load orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ
     "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES "make_unique<HLIEModelLoadService>")
    message(FATAL_ERROR
            "Application composition root does not own ModelLoadService construction")
endif ()

message(STATUS "ModelLoadService dependency and Workflow ownership checks passed")
