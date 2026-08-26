if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_APPLICATION_LIFECYCLE_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/lifecycle/hlie_application_lifecycle_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_lifecycle_service.cpp")
set(HLIE_APPLICATION_LIFECYCLE_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_service_manager"
        "hliemodelservicemanager"
        "hlie_process_manager"
        "hlieprocessmanager"
        "hlie_device_service"
        "hliedeviceservice"
        "hlie_llm_model_service"
        "hliellmmodelservice"
        "hlie_sd_model_service"
        "hliesdmodelservice"
        "hlie_whisper_model_service"
        "hliewhispermodelservice"
        "hlie_cosyvoice_model_service"
        "hliecosyvoicemodelservice"
        "hlie_qwen3_tts_model_service"
        "hlieqwen3ttsmodelservice"
        "hlie_worker_supervisor"
        "hlieworkersupervisor"
        "hlie_control_plane_reconciler"
        "hliecontrolplanereconciler"
        "hlie_model_discovery_service"
        "hliemodeldiscoveryservice"
        "hlie_model_readiness_service"
        "hliemodelreadinessservice"
        "hlie_worker_control_application_service"
        "hlieworkercontrolapplicationservice"
        "hlie_worker_control_service"
        "hlieworkercontrolservice"
        "server_http"
        "httplib"
        "json_value"
        "nlohmann"
        "rapidjson"
        "json.hpp"
        "singleton<"
        "tcim"
        "ggml")

foreach (HLIE_FILE IN LISTS HLIE_APPLICATION_LIFECYCLE_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "Application lifecycle service source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_APPLICATION_LIFECYCLE_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Application lifecycle boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "embedded_startup_service_->start_group"
        "control_plane_reconciler_->start"
        "control_plane_reconciler_->stop"
        "model_discovery_service_->start"
        "model_discovery_service_->stop"
        "model_controller_->begin_draining"
        "model_readiness_service_->stop_and_join"
        "asr_application_service_->stop_realtime"
        "model_controller_->stop_all"
        "worker_supervisor_->shutdown_all"
        "stop_and_reset_sd"
        "stop_and_reset_whisper"
        "stop_and_reset_cosyvoice"
        "stop_and_reset_qwen3_tts"
        "stop_and_reset_llm"
        "worker_control_service_->stop_and_join")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns application lifecycle orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

foreach (HLIE_REQUIRED IN ITEMS
        "const auto started = application_lifecycle_service_->start"
        "if (!started.started())"
        "application_lifecycle_service_->stop()")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_REQUIRED}" HLIE_MATCH_INDEX)
    if (HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow does not propagate lifecycle startup failure: ${HLIE_REQUIRED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_app_controller.cpp"
     HLIE_APP_CONTROLLER)
if (NOT HLIE_APP_CONTROLLER MATCHES "if \\(!workflow_->run\\(\\)\\)")
    message(FATAL_ERROR
            "AppController can publish WebService after lifecycle startup failure")
endif ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEApplicationLifecycleService>")
    message(FATAL_ERROR
            "Application composition root does not own ApplicationLifecycleService construction")
endif ()

message(STATUS
        "ApplicationLifecycleService dependency and Workflow ownership checks passed")
