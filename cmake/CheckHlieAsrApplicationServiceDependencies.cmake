if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_ASR_APPLICATION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/asr/hlie_asr_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/asr/hlie_asr_application_service.cpp")
set(HLIE_ASR_APPLICATION_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_service_manager"
        "hlie_llm_model_service"
        "hlie_sd_model_service"
        "hlie_whisper_model_service"
        "hlie_cosyvoice_model_service"
        "hlie_qwen3_tts_model_service"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_ws_service"
        "singleton<"
        "httplib"
        "tcim"
        "ggml"
        "server_res_generator"
        "safe_json_to_str"
        "format_error_response")

foreach (HLIE_FILE IN LISTS HLIE_ASR_APPLICATION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ASR application service source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_ASR_APPLICATION_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ASR application service boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "ensure_asr_websocket"
        "handle_asr_stream"
        "handle_transcriptions_pipeline"
        "is_whisper_transcription_model"
        "HLIEWebSocketService"
        "HLIEQwenAsrStreamDecoder"
        "requests_word_timestamps"
        "is_force_aligner_name"
        "is_qwen3_asr_name")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns ASR application orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEAsrApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own ASR application service construction")
endif ()
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "HLIEDefaultAsrRealtimeTransport")
    message(FATAL_ERROR
            "Application composition root does not configure ASR realtime transport")
endif ()

message(STATUS "ASR application service dependency and Workflow ownership checks passed")
