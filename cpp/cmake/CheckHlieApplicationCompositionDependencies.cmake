if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_COMPOSITION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/composition/hlie_application_composition_root.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp")
foreach (HLIE_FILE IN LISTS HLIE_COMPOSITION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "Application composition source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN ITEMS
            "hlie_workflow"
            "singleton<"
            "hlie_config_service"
            "hlie_process_manager")
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "Application composition boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW_CONTENT)
foreach (HLIE_BANNED IN ITEMS
        "make_unique<HLIEResourceScheduler>"
        "make_unique<HLIEDefaultModelRuntimeFactory>"
        "make_unique<HLIEDefaultModelBackendLauncher>"
        "make_unique<HLIEModelController>"
        "make_unique<HLIEInferenceDispatcher>"
        "make_unique<HLIEControlPlaneReconciler>"
        "make_unique<HLIEModelDiscoveryService>")
    string(FIND "${HLIE_WORKFLOW_CONTENT}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still constructs a control-plane component: ${HLIE_BANNED}")
    endif ()
endforeach ()

message(STATUS "Application composition root boundary checks passed")
