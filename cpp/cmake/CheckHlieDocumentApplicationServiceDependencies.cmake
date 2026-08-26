if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_DOCUMENT_APPLICATION_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/document/hlie_document_application_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/document/hlie_document_application_service.cpp")
set(HLIE_DOCUMENT_APPLICATION_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_service_manager"
        "hlie_document_pipeline_service"
        "hlie_llm_model_service"
        "hlie_process_manager"
        "hlie_device_service"
        "singleton<"
        "httplib"
        "tcim"
        "ggml"
        "pdfium")

foreach (HLIE_FILE IN LISTS HLIE_DOCUMENT_APPLICATION_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR
                "DocumentApplicationService source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_DOCUMENT_APPLICATION_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "DocumentApplicationService boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
foreach (HLIE_BANNED IN ITEMS
        "DocumentInputAdapter::inspect"
        "HLIEInferenceOperation::DocumentPreprocess"
        "HLIEInferenceOperation::DocumentOcr"
        "stop_and_reset_document_pipeline")
    string(FIND "${HLIE_WORKFLOW}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "Workflow still owns document application orchestration: ${HLIE_BANNED}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_application_composition_root.cpp"
     HLIE_COMPOSITION_ROOT)
if (NOT HLIE_COMPOSITION_ROOT MATCHES
        "make_unique<HLIEDocumentApplicationService>")
    message(FATAL_ERROR
            "Application composition root does not own DocumentApplicationService construction")
endif ()

message(STATUS "DocumentApplicationService dependency and Workflow ownership checks passed")
