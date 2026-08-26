if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_INFERENCE_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/inference/hlie_inference_endpoint.h"
        "${HLIE_SOURCE_DIR}/include/hlie_app/inference/hlie_inference_dispatcher.h"
        "${HLIE_SOURCE_DIR}/include/hlie_app/inference/hlie_inference_http_error_mapper.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_inference_dispatcher.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_inference_http_error_mapper.cpp")

foreach (HLIE_FILE IN LISTS HLIE_INFERENCE_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "Missing inference data-plane file: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    if (HLIE_CONTENT MATCHES "hlie_workflow\\.h|hlie_model_service_manager\\.h|hlie_(llm|sd|whisper|cosyvoice|qwen3_tts)_model_service\\.h|hlie_device_service\\.h|tcim|ggml")
        message(FATAL_ERROR
                "Inference dispatcher contract depends on a forbidden concrete owner/backend: ${HLIE_FILE}")
    endif ()
endforeach ()

file(READ
     "${HLIE_SOURCE_DIR}/include/hlie_app/inference/hlie_inference_endpoint.h"
     HLIE_ENDPOINT_HEADER)
if (HLIE_ENDPOINT_HEADER MATCHES "nlohmann|ordered_json|HLIEModelInfo|HLIEModelService")
    message(FATAL_ERROR
            "Inference endpoint public contract leaks catalog, JSON, or concrete Service types")
endif ()

set(HLIE_HTTP_BOUNDARY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/http/hlie_api_router.h"
        "${HLIE_SOURCE_DIR}/include/hlie_app/http/hlie_http_route_provider.h"
        "${HLIE_SOURCE_DIR}/include/hlie_app/hlie_web_service.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_api_router.cpp"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_web_service.cpp")
foreach (HLIE_FILE IN LISTS HLIE_HTTP_BOUNDARY_FILES)
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    if (HLIE_CONTENT MATCHES "hlie_workflow\\.h|HLIEWorkflow|hlie_model_service_manager\\.h|hlie_(llm|sd|whisper|cosyvoice|qwen3_tts)_model_service\\.h")
        message(FATAL_ERROR
                "WebService/HTTP provider boundary depends on Workflow or a concrete Service: ${HLIE_FILE}")
    endif ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_api_router.cpp" HLIE_API_ROUTER)
if (HLIE_API_ROUTER MATCHES "HLIEModelController|HLIEModelServiceManager|HLIEDeviceService|ServerHttpProxy")
    message(FATAL_ERROR
            "ApiRouter depends on a control owner, concrete ServiceManager, device, or proxy")
endif ()

file(READ "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_workflow.cpp" HLIE_WORKFLOW)
if (HLIE_WORKFLOW MATCHES "model_service_manager_[^\n]*->handle_|ServerHttpProxy|class ModelRequestLease|attach_model_request_lease")
    message(FATAL_ERROR
            "Workflow regained direct inference Service/proxy dispatch or request-lease ownership")
endif ()

message(STATUS "HLIE inference dispatcher dependency gate passed")
