if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(CORE_RUNTIME_HEADER
        "${HLIE_SOURCE_DIR}/include/hlie_core/runtime/hlie_model_runtime.h")
file(READ "${CORE_RUNTIME_HEADER}" CORE_RUNTIME_CONTENT)
foreach (BANNED IN ITEMS "hlie_app" "server/" "httplib" "nlohmann" "tcim" "ggml" "void *" "std::any")
    string(FIND "${CORE_RUNTIME_CONTENT}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR "Core runtime contract leaks forbidden dependency/type: ${BANNED}")
    endif ()
endforeach ()

set(EMBEDDED_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_embedded_model_runtime.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_embedded_model_runtime.cpp")
set(PROCESS_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_process_model_runtime.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_process_model_runtime.cpp")
set(DIRECTORY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_runtime_directory.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_runtime_directory.cpp")
set(RUNTIME_FACTORY_PORT
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_model_runtime_factory.h")
set(DEFAULT_RUNTIME_FACTORY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_default_model_runtime_factory.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_default_model_runtime_factory.cpp")

foreach (RUNTIME_FILE IN LISTS EMBEDDED_FILES PROCESS_FILES DIRECTORY_FILES)
    file(READ "${RUNTIME_FILE}" CONTENT)
    string(TOLOWER "${CONTENT}" CONTENT_LOWER)
    foreach (BANNED IN ITEMS "hlie_workflow" "hlie_model_registry" "hlie_device_service" "server/" "httplib" "tcim" "ggml")
        string(FIND "${CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelRuntime dependency boundary violation: ${RUNTIME_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${RUNTIME_FACTORY_PORT}" FACTORY_PORT_CONTENT)
string(TOLOWER "${FACTORY_PORT_CONTENT}" FACTORY_PORT_CONTENT_LOWER)
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
    string(FIND "${FACTORY_PORT_CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "RuntimeFactory port leaks implementation dependency: ${BANNED}")
    endif ()
endforeach ()

foreach (FACTORY_FILE IN LISTS DEFAULT_RUNTIME_FACTORY_FILES)
    file(READ "${FACTORY_FILE}" FACTORY_CONTENT)
    string(TOLOWER "${FACTORY_CONTENT}" FACTORY_CONTENT_LOWER)
    foreach (BANNED IN ITEMS
            "hlie_workflow"
            "hlie_model_registry"
            "hlie_device_service"
            "server/"
            "httplib"
            "tcim"
            "ggml")
        string(FIND "${FACTORY_CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Default RuntimeFactory crosses composition boundary: ${FACTORY_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

foreach (DIRECTORY_FILE IN LISTS DIRECTORY_FILES)
    file(READ "${DIRECTORY_FILE}" DIRECTORY_CONTENT)
    string(TOLOWER "${DIRECTORY_CONTENT}" DIRECTORY_CONTENT_LOWER)
    foreach (BANNED IN ITEMS "hlie_worker_supervisor" "hlie_model_service_manager" "hlie_process_manager")
        string(FIND "${DIRECTORY_CONTENT_LOWER}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "RuntimeDirectory must remain a handle-only container: ${DIRECTORY_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_embedded_model_runtime.h" EMBEDDED_HEADER)
foreach (PUBLIC_BANNED IN ITEMS "hlie_model_service_manager" "hlie_model_service" "hlie_llm_model_service")
    string(FIND "${EMBEDDED_HEADER}" "${PUBLIC_BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR "Embedded runtime public header leaks concrete service type: ${PUBLIC_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_process_model_runtime.h" PROCESS_HEADER)
foreach (PUBLIC_BANNED IN ITEMS "hlie_process_manager" "httplib" "nlohmann")
    string(FIND "${PROCESS_HEADER}" "${PUBLIC_BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR "Process runtime public header leaks implementation type: ${PUBLIC_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" WORKFLOW_CONTENT)
file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
        COMPOSITION_CONTENT)
file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_embedded_startup_service.cpp"
        EMBEDDED_STARTUP_CONTENT)
string(FIND "${WORKFLOW_CONTENT}" "transfer_to_local_runtime" LEGACY_TRANSFER_INDEX)
if (NOT LEGACY_TRANSFER_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Workflow must not transfer local reservation ownership to concrete services")
endif ()
string(FIND "${WORKFLOW_CONTENT}" "model_controller_->publish_embedded_runtime"
        WORKFLOW_PUBLISH_INDEX)
string(FIND "${EMBEDDED_STARTUP_CONTENT}"
        "controller_.publish_embedded_runtime" STARTUP_PUBLISH_INDEX)
if (WORKFLOW_PUBLISH_INDEX EQUAL -1 AND STARTUP_PUBLISH_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Application is missing ModelRuntime embedded publication integration")
endif ()
string(FIND "${COMPOSITION_CONTENT}"
        "controller->stop_all(\n                            HLIERuntimeStopReason::Shutdown)"
        LIFECYCLE_STOP_ALL_INDEX)
if (LIFECYCLE_STOP_ALL_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Application lifecycle is missing shutdown-wide ModelRuntime stop integration")
endif ()
string(FIND "${WORKFLOW_CONTENT}" "model_controller_->stop_all"
        WORKFLOW_STOP_ALL_INDEX)
if (NOT WORKFLOW_STOP_ALL_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Workflow must delegate shutdown-wide ModelRuntime stop through ApplicationLifecycleService")
endif ()
string(FIND "${COMPOSITION_CONTENT}" "controller->evict_runtime"
        COMPOSITION_EVICTION_INDEX)
if (COMPOSITION_EVICTION_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Composition root is missing ModelRuntime eviction integration")
endif ()
string(FIND "${WORKFLOW_CONTENT}" "model_controller_->evict_runtime"
        WORKFLOW_EVICTION_INDEX)
if (NOT WORKFLOW_EVICTION_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Workflow must delegate ModelRuntime eviction through application services")
endif ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_load_service.cpp"
        MODEL_LOAD_SERVICE_CONTENT)
string(FIND "${MODEL_LOAD_SERVICE_CONTENT}" "controller_.load_model"
        MODEL_LOAD_INDEX)
if (MODEL_LOAD_INDEX EQUAL -1)
    message(FATAL_ERROR
            "ModelLoadService is missing ModelRuntime load integration")
endif ()

string(FIND "${COMPOSITION_CONTENT}" "HLIEDefaultModelRuntimeFactory"
        MATCH_INDEX)
if (MATCH_INDEX EQUAL -1)
    message(FATAL_ERROR
            "Composition root is missing HLIEDefaultModelRuntimeFactory")
endif ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_controller.cpp"
        CONTROLLER_CONTENT)
foreach (REQUIRED IN ITEMS
        "runtime_directory_.detach"
        "HLIERuntimeStopReason::Eviction"
        "scheduler_.release"
        "mark_unloaded_if_current")
    string(FIND "${CONTROLLER_CONTENT}" "${REQUIRED}" MATCH_INDEX)
    if (MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "ModelController is missing ModelRuntime ownership integration: ${REQUIRED}")
    endif ()
endforeach ()

foreach (SERVICE_FILE IN ITEMS
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_llm_model_service.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_sd_model_service.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_whisper_model_service.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_cosyvoice_model_service.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_qwen3_tts_model_service.cpp")
    file(READ "${SERVICE_FILE}" SERVICE_CONTENT)
    string(FIND "${SERVICE_CONTENT}" "reservation_owned_externally_" MATCH_INDEX)
    if (MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Concrete service does not preserve ResourceScheduler ownership: ${SERVICE_FILE}")
    endif ()
endforeach ()

message(STATUS "ModelRuntime dependency boundary check passed")
