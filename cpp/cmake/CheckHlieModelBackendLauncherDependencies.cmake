if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(LAUNCHER_PORT
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_model_backend_launcher.h")
file(READ "${LAUNCHER_PORT}" LAUNCHER_PORT_CONTENT)
string(TOLOWER "${LAUNCHER_PORT_CONTENT}" LAUNCHER_PORT_CONTENT_LOWER)
foreach (BANNED IN ITEMS
        "hlie_workflow"
        "hlie_model_service_manager"
        "hlie_worker_supervisor"
        "hlie_process_manager"
        "hlie_device_service"
        "server/"
        "httplib"
        "tcim"
        "ggml")
    string(FIND "${LAUNCHER_PORT_CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "BackendLauncher port leaks implementation dependency: ${BANNED}")
    endif ()
endforeach ()

set(DEFAULT_LAUNCHER_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_default_model_backend_launcher.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_default_model_backend_launcher.cpp")
foreach (LAUNCHER_FILE IN LISTS DEFAULT_LAUNCHER_FILES)
    file(READ "${LAUNCHER_FILE}" LAUNCHER_CONTENT)
    string(TOLOWER "${LAUNCHER_CONTENT}" LAUNCHER_CONTENT_LOWER)
    foreach (BANNED IN ITEMS
            "hlie_workflow"
            "hlie_model_registry"
            "server/"
            "httplib"
            "tcim"
            "ggml")
        string(FIND "${LAUNCHER_CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Default BackendLauncher crosses composition boundary: ${LAUNCHER_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_controller.cpp"
        CONTROLLER_CONTENT)
foreach (REQUIRED IN ITEMS
        "backend_launcher_->launch"
        "adopt_embedded_runtime"
        "adopt_process_runtime"
        "cleanup_unadopted")
    string(FIND "${CONTROLLER_CONTENT}" "${REQUIRED}" MATCH_INDEX)
    if (MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "ModelController does not own the complete load transaction: ${REQUIRED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_load_service.cpp"
        MODEL_LOAD_SERVICE_CONTENT)
string(FIND "${MODEL_LOAD_SERVICE_CONTENT}" "controller_.load_model"
        MATCH_INDEX)
if (MATCH_INDEX EQUAL -1)
    message(FATAL_ERROR
            "ModelLoadService does not delegate startup through Controller/BackendLauncher")
endif ()
file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
        COMPOSITION_CONTENT)
string(FIND "${COMPOSITION_CONTENT}" "HLIEDefaultModelBackendLauncher"
        MATCH_INDEX)
if (MATCH_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Composition root is missing HLIEDefaultModelBackendLauncher")
endif ()
foreach (BANNED IN ITEMS
        "HLIEWorkflow::start_whisper"
        "HLIEWorkflow::start_cosyvoice"
        "HLIEWorkflow::start_qwen3_tts"
        "model_controller_->begin_load"
        "model_controller_->adopt_embedded_runtime"
        "model_controller_->adopt_process_runtime"
        "model_service_manager_.start_"
        "worker_supervisor_->reserve_port"
        "worker_supervisor_->spawn")
    string(FIND "${WORKFLOW_CONTENT}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow bypasses Controller-owned load transaction: ${BANNED}")
    endif ()
endforeach ()

message(STATUS "BackendLauncher and Controller load-transaction boundary check passed")
