if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_LLM_APPLICATION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/llm/hlie_llm_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/llm/hlie_llm_application_service.cpp")
set(HLIE_LLM_APPLICATION_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_registry"
        "hlie_model_load_service"
        "hlie_model_readiness_service"
        "hlie_model_discovery_service"
        "hlie_runtime_directory"
        "hlie_inference_dispatcher"
        "hlie_model_service_manager"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "singleton<"
        "json_value("
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_LLM_APPLICATION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "LLM application service source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_LLM_APPLICATION_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "LLM application service boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "HLIEWorkflow::route_llm_request"
        "handlers.llm = [this](const server_http_req& req,\n                          const HLIEInferenceOperation operation,\n                          const std::string& method) {\n        auto permit")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow regained LLM compatibility routing ownership: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIELlmApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own LlmApplicationService construction")
endif ()
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "handle_admitted\\(request, operation\\)")
    message(FATAL_ERROR
            "ASR composition does not reuse the admitted LLM route")
endif ()

message(STATUS "LlmApplicationService dependency and Workflow ownership checks passed")
