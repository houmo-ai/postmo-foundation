if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_DISCOVERY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/discovery/hlie_model_discovery_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_discovery_service.cpp")
set(HLIE_DISCOVERY_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_registry"
        "hlie_config_service"
        "hlie_process_manager"
        "hlie_device_service"
        "singleton<"
        "server/"
        "httplib"
        "tcim"
        "ggml"
        "hlie_gguf")

foreach (HLIE_FILE IN LISTS HLIE_DISCOVERY_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ModelDiscoveryService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_DISCOVERY_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelDiscoveryService dependency boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

set(HLIE_WORKFLOW_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/hlie_workflow.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp")
foreach (HLIE_FILE IN LISTS HLIE_WORKFLOW_FILES)
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    foreach (HLIE_BANNED IN ITEMS
            "model_discovery_thread_"
            "model_discovery_stop_"
            "model_discovery_requested_"
            "model_discovery_loop"
            "build_model_directory_snapshot")
        string(FIND "${HLIE_CONTENT}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Workflow still owns a DiscoveryService concern: ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

message(STATUS "ModelDiscoveryService dependency and Workflow ownership checks passed")
