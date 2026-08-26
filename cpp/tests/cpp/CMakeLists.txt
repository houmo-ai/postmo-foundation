add_test(NAME hlie.architecture.core-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieCoreDependencies.cmake)
set_tests_properties(hlie.architecture.core-dependencies PROPERTIES
        LABELS "architecture;core"
        TIMEOUT 30)

add_test(NAME hlie.architecture.worker-supervisor-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieWorkerSupervisorDependencies.cmake)
set_tests_properties(hlie.architecture.worker-supervisor-dependencies PROPERTIES
        LABELS "architecture;worker-supervisor"
        TIMEOUT 30)

add_test(NAME hlie.architecture.resource-scheduler-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieResourceSchedulerDependencies.cmake)
set_tests_properties(hlie.architecture.resource-scheduler-dependencies PROPERTIES
        LABELS "architecture;resource-scheduler"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-runtime-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelRuntimeDependencies.cmake)
set_tests_properties(hlie.architecture.model-runtime-dependencies PROPERTIES
        LABELS "architecture;model-runtime"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-controller-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelControllerDependencies.cmake)
set_tests_properties(hlie.architecture.model-controller-dependencies PROPERTIES
        LABELS "architecture;model-controller"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-backend-launcher-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelBackendLauncherDependencies.cmake)
set_tests_properties(hlie.architecture.model-backend-launcher-dependencies PROPERTIES
        LABELS "architecture;model-controller;backend-launcher"
        TIMEOUT 30)

add_test(NAME hlie.architecture.inference-dispatcher-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieInferenceDispatcherDependencies.cmake)
set_tests_properties(hlie.architecture.inference-dispatcher-dependencies PROPERTIES
        LABELS "architecture;inference-dispatcher"
        TIMEOUT 30)

add_test(NAME hlie.architecture.control-plane-reconciler-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieControlPlaneReconcilerDependencies.cmake)
set_tests_properties(hlie.architecture.control-plane-reconciler-dependencies PROPERTIES
        LABELS "architecture;control-plane-reconciler"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-discovery-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelDiscoveryDependencies.cmake)
set_tests_properties(hlie.architecture.model-discovery-dependencies PROPERTIES
        LABELS "architecture;model-discovery"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-catalog-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelCatalogDependencies.cmake)
set_tests_properties(hlie.architecture.model-catalog-dependencies PROPERTIES
        LABELS "architecture;model-catalog"
        TIMEOUT 30)

add_test(NAME hlie.architecture.application-composition-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieApplicationCompositionDependencies.cmake)
set_tests_properties(hlie.architecture.application-composition-dependencies PROPERTIES
        LABELS "architecture;composition-root"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-load-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelLoadServiceDependencies.cmake)
set_tests_properties(hlie.architecture.model-load-service-dependencies PROPERTIES
        LABELS "architecture;model-load-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-readiness-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelReadinessServiceDependencies.cmake)
set_tests_properties(hlie.architecture.model-readiness-service-dependencies PROPERTIES
        LABELS "architecture;model-readiness-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.worker-control-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieWorkerControlServiceDependencies.cmake)
set_tests_properties(hlie.architecture.worker-control-service-dependencies PROPERTIES
        LABELS "architecture;worker-control-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.embedded-startup-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieEmbeddedStartupServiceDependencies.cmake)
set_tests_properties(hlie.architecture.embedded-startup-service-dependencies PROPERTIES
        LABELS "architecture;embedded-startup-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.document-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieDocumentApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.document-application-service-dependencies PROPERTIES
        LABELS "architecture;document-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.asr-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieAsrApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.asr-application-service-dependencies PROPERTIES
        LABELS "architecture;asr-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.speech-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieSpeechApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.speech-application-service-dependencies PROPERTIES
        LABELS "architecture;speech-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.default-model-initialization-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieDefaultModelInitializationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.default-model-initialization-dependencies PROPERTIES
        LABELS "architecture;default-model-initialization"
        TIMEOUT 30)

add_test(NAME hlie.architecture.bootstrap-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieBootstrapApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.bootstrap-application-service-dependencies PROPERTIES
        LABELS "architecture;bootstrap-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.application-lifecycle-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieApplicationLifecycleServiceDependencies.cmake)
set_tests_properties(hlie.architecture.application-lifecycle-service-dependencies PROPERTIES
        LABELS "architecture;application-lifecycle-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-list-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelListApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.model-list-application-service-dependencies PROPERTIES
        LABELS "architecture;model-list-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.model-unload-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieModelUnloadApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.model-unload-application-service-dependencies PROPERTIES
        LABELS "architecture;model-unload-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.stop-service-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieStopServiceApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.stop-service-application-service-dependencies PROPERTIES
        LABELS "architecture;stop-service-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.observability-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieObservabilityApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.observability-application-service-dependencies PROPERTIES
        LABELS "architecture;observability-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.llm-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieLlmApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.llm-application-service-dependencies PROPERTIES
        LABELS "architecture;llm-application-service"
        TIMEOUT 30)

add_test(NAME hlie.architecture.worker-control-application-service-dependencies
        COMMAND ${CMAKE_COMMAND}
            -DHLIE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -P ${PROJECT_SOURCE_DIR}/cmake/CheckHlieWorkerControlApplicationServiceDependencies.cmake)
set_tests_properties(hlie.architecture.worker-control-application-service-dependencies PROPERTIES
        LABELS "architecture;worker-control-application-service"
        TIMEOUT 30)

if (HLIE_TESTS_ONLY)
    # A proprietary-SDK-free control-plane slice for TSAN and fast Linux CI.
    # Keep this target limited to sources that do not initialize or link TCIM,
    # HAL, llama, Whisper, Stable Diffusion, or uWebSockets runtimes.
    add_library(hlie-control-plane-test-support STATIC
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_backend_adapter_registry.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_api_router.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_info_core.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_llm_param_set.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_controller.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_readiness_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_control_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_control_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_embedded_startup_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_default_model_initialization_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_bootstrap_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_application_lifecycle_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_model_list_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_model_unload_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_stop_service_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/observability/hlie_observability_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/llm/hlie_llm_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_input_adapter.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/asr/hlie_asr_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_qwen_asr_stream_decoder.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/speech/hlie_speech_application_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_inference_dispatcher.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_control_plane_reconciler.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_discovery_service.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_catalog.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_registry.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_resource_scheduler.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_runtime_directory.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_cancellation_metrics.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_base/hlie_util.cpp
            ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_background_task_group.cpp
            support/ggml_assert_test_shim.cpp)
    target_include_directories(hlie-control-plane-test-support PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include
            ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include/nlohmann
            ${PROJECT_SOURCE_DIR}/third_party/spdlog/include)
    target_compile_definitions(hlie-control-plane-test-support PUBLIC
            HLIE_STATIC
            SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
    target_compile_features(hlie-control-plane-test-support PUBLIC cxx_std_17)
    target_link_libraries(hlie-control-plane-test-support PUBLIC
            hlie-core
            cpp-httplib
            Threads::Threads
            ${CMAKE_DL_LIBS})
    if (WIN32)
        target_link_libraries(hlie-control-plane-test-support PUBLIC Rpcrt4)
    endif ()

    add_executable(hlie-unit-tests
            unit/hlie_admission_controller_test.cpp
            unit/hlie_api_router_test.cpp
            unit/hlie_backend_adapter_registry_test.cpp
            unit/hlie_backend_registry_test.cpp
            unit/hlie_blocking_queue_test.cpp
            unit/hlie_control_contract_test.cpp
            unit/hlie_control_plane_stress_test.cpp
            unit/hlie_control_plane_reconciler_test.cpp
            unit/hlie_model_discovery_service_test.cpp
            unit/hlie_model_catalog_test.cpp
            unit/hlie_device_reservation_ledger_test.cpp
            unit/hlie_inference_dispatcher_test.cpp
            unit/hlie_model_registry_test.cpp
            unit/hlie_model_controller_test.cpp
            unit/hlie_worker_control_service_test.cpp
            unit/hlie_worker_control_application_service_test.cpp
            unit/hlie_embedded_startup_service_test.cpp
            unit/hlie_default_model_initialization_service_test.cpp
            unit/hlie_bootstrap_application_service_test.cpp
            unit/hlie_application_lifecycle_service_test.cpp
            unit/hlie_model_list_application_service_test.cpp
            unit/hlie_model_unload_application_service_test.cpp
            unit/hlie_stop_service_application_service_test.cpp
            unit/hlie_observability_application_service_test.cpp
            unit/hlie_llm_application_service_test.cpp
            unit/hlie_document_application_service_test.cpp
            unit/hlie_asr_application_service_test.cpp
            unit/hlie_speech_application_service_test.cpp
            unit/hlie_model_runtime_registry_test.cpp
            unit/hlie_model_runtime_contract_test.cpp
            unit/hlie_model_residency_manager_test.cpp
            unit/hlie_queued_request_control_test.cpp
            unit/hlie_resource_scheduler_test.cpp
            unit/hlie_runtime_directory_test.cpp
            unit/hlie_service_slot_test.cpp)
    target_link_libraries(hlie-unit-tests PRIVATE
            Catch2::Catch2WithMain
            hlie-control-plane-test-support)
    target_compile_features(hlie-unit-tests PRIVATE cxx_std_17)
    if (HLIE_SANITIZE_THREAD)
        target_compile_definitions(hlie-unit-tests PRIVATE
                HLIE_THREAD_SANITIZER=1)
    endif ()

    add_test(NAME hlie.unit
            COMMAND hlie-unit-tests --reporter console)
    set_tests_properties(hlie.unit PROPERTIES
            LABELS "unit;control-plane"
            TIMEOUT 120)

    add_test(NAME hlie.stress.control-plane
            COMMAND hlie-unit-tests "[p0][stress]" --reporter console)
    set_tests_properties(hlie.stress.control-plane PROPERTIES
            LABELS "stress;p0;control-plane"
            TIMEOUT 120
            RUN_SERIAL TRUE)

    set_target_properties(
            hlie-control-plane-test-support
            hlie-unit-tests
            PROPERTIES FOLDER "tests/control-plane")
    return()
endif ()

add_library(hlie-testable-core STATIC
        ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_background_task_group.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_api_router.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_backend_adapter_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_config_bundle_validator.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_info_core.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_info.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_memory_estimator.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_llm_param_set.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_llm_preset_resolver.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_model_load_waiter.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_controller.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_load_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_readiness_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_control_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_control_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_embedded_startup_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_default_model_initialization_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_bootstrap_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_application_lifecycle_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_model_list_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_model_unload_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/models/hlie_stop_service_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/observability/hlie_observability_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/llm/hlie_llm_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_input_adapter.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/asr/hlie_asr_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/speech/hlie_speech_application_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_inference_dispatcher.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_control_plane_reconciler.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_discovery_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_catalog.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_cancellation_metrics.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_process_manager.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_process_model_runtime.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_resource_scheduler.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_runtime_directory.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_supervisor.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_worker_port_allocator.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_proxy_timeout.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_qwen_asr_stream_decoder.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/audio/hlie_realtime_vad.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_ws_service.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_core.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_region_cropper.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/document/hlie_document_types.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_hal_path_resolver.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/util/hlie_http_log_sanitizer.cpp
        support/ggml_assert_test_shim.cpp)
target_include_directories(hlie-testable-core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/uwebsockets/include
        ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include
        ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include/nlohmann
        ${PROJECT_SOURCE_DIR}/third_party/spdlog/include)
target_compile_definitions(hlie-testable-core PUBLIC HLIE_STATIC)
target_compile_features(hlie-testable-core PUBLIC cxx_std_17)
target_link_libraries(hlie-testable-core PUBLIC
        hlie-core
        hlie-base
        ${LLAMA_CPP_LIBRARY_GGML}
        ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU}
        ${UWEBSOCKET_LIBRARY}
        ${UV_LIBRARY}
        cpp-httplib
        Threads::Threads
        ${CMAKE_DL_LIBS})

add_executable(hlie-unit-tests
        unit/hlie_background_task_group_test.cpp
        unit/hlie_admission_controller_test.cpp
        unit/hlie_api_router_test.cpp
        unit/hlie_backend_adapter_registry_test.cpp
        unit/hlie_backend_registry_test.cpp
        unit/hlie_blocking_queue_test.cpp
        unit/hlie_control_contract_test.cpp
        unit/hlie_config_service_test.cpp
        unit/hlie_config_bundle_validator_test.cpp
        unit/hlie_control_plane_stress_test.cpp
        unit/hlie_control_plane_reconciler_test.cpp
        unit/hlie_model_discovery_service_test.cpp
        unit/hlie_model_catalog_test.cpp
        unit/hlie_document_core_test.cpp
        unit/hlie_document_application_service_test.cpp
        unit/hlie_asr_application_service_test.cpp
        unit/hlie_speech_application_service_test.cpp
        unit/hlie_device_reservation_ledger_test.cpp
        unit/hlie_hal_path_resolver_test.cpp
        unit/hlie_logger_test.cpp
        unit/hlie_http_log_sanitizer_test.cpp
        unit/hlie_inference_dispatcher_test.cpp
        unit/hlie_llm_preset_resolver_test.cpp
        unit/hlie_model_load_waiter_test.cpp
        unit/hlie_model_family_registry_test.cpp
        unit/hlie_model_info_test.cpp
        unit/hlie_model_controller_test.cpp
        unit/hlie_model_load_service_test.cpp
        unit/hlie_worker_control_service_test.cpp
        unit/hlie_worker_control_application_service_test.cpp
        unit/hlie_embedded_startup_service_test.cpp
        unit/hlie_default_model_initialization_service_test.cpp
        unit/hlie_bootstrap_application_service_test.cpp
        unit/hlie_application_lifecycle_service_test.cpp
        unit/hlie_model_list_application_service_test.cpp
        unit/hlie_model_unload_application_service_test.cpp
        unit/hlie_stop_service_application_service_test.cpp
        unit/hlie_observability_application_service_test.cpp
        unit/hlie_llm_application_service_test.cpp
        unit/hlie_model_registry_test.cpp
        unit/hlie_model_runtime_contract_test.cpp
        unit/hlie_model_runtime_registry_test.cpp
        unit/hlie_model_residency_manager_test.cpp
        unit/hlie_proxy_timeout_test.cpp
        unit/hlie_process_manager_test.cpp
        unit/hlie_worker_port_allocator_test.cpp
        unit/hlie_queued_request_control_test.cpp
        unit/hlie_resource_scheduler_test.cpp
        unit/hlie_runtime_directory_test.cpp
        unit/hlie_service_slot_test.cpp
        unit/hlie_ws_limits_test.cpp
        unit/hlie_whisper_language_test.cpp
        unit/audio/hlie_realtime_vad_test.cpp
        unit/audio/hlie_realtime_session_test.cpp)
target_sources(hlie-unit-tests PRIVATE
        unit/hlie_qwen_asr_stream_decoder_test.cpp)
target_link_libraries(hlie-unit-tests PRIVATE
        Catch2::Catch2WithMain
        hlie-testable-core)
target_compile_features(hlie-unit-tests PRIVATE cxx_std_17)
if (HLIE_SANITIZE_THREAD)
    target_compile_definitions(hlie-unit-tests PRIVATE
            HLIE_THREAD_SANITIZER=1)
endif ()

if (UNIX AND NOT APPLE AND NOT HLIE_TESTS_ONLY)
    set(HLIE_TEST_LINK_TIME_LIBRARY_DIRS
            "${PROJECT_SOURCE_DIR}/third_party/houmo/lib"
            "${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/lib"
            "${PROJECT_SOURCE_DIR}/third_party/kaldi-native-fbank/lib"
            "${PROJECT_SOURCE_DIR}/third_party/uwebsockets/lib")
    list(JOIN HLIE_TEST_LINK_TIME_LIBRARY_DIRS ":" HLIE_TEST_LINK_TIME_LIBRARY_PATH)
    target_link_options(hlie-unit-tests PRIVATE
            "-Wl,-rpath-link,${HLIE_TEST_LINK_TIME_LIBRARY_PATH}")
    set(HLIE_TEST_RUNTIME_LIBRARY_PATH
            "${CMAKE_BINARY_DIR}/bin:${HLIE_TEST_LINK_TIME_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}")
endif ()

add_test(NAME hlie.unit COMMAND hlie-unit-tests --reporter console)
set_tests_properties(hlie.unit PROPERTIES
        LABELS "unit"
        TIMEOUT 30)
if (HLIE_TEST_RUNTIME_LIBRARY_PATH)
    set_property(TEST hlie.unit APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${HLIE_TEST_RUNTIME_LIBRARY_PATH}")
endif ()

add_test(NAME hlie.stress.control-plane
        COMMAND hlie-unit-tests "[p0][stress]" --reporter console)
set_tests_properties(hlie.stress.control-plane PROPERTIES
        LABELS "stress;p0;control-plane"
        TIMEOUT 30
        RUN_SERIAL TRUE)
if (HLIE_TEST_RUNTIME_LIBRARY_PATH)
    set_property(TEST hlie.stress.control-plane APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${HLIE_TEST_RUNTIME_LIBRARY_PATH}")
endif ()

add_executable(hlie-fake-worker
        support/fake_worker_main.cpp)
target_include_directories(hlie-fake-worker PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include)
target_link_libraries(hlie-fake-worker PRIVATE
        cpp-httplib)
target_compile_features(hlie-fake-worker PRIVATE cxx_std_17)

add_executable(hlie-worker-supervisor-component-tests
        component/hlie_worker_supervisor_test.cpp)
target_link_libraries(hlie-worker-supervisor-component-tests PRIVATE
        Catch2::Catch2WithMain
        hlie-testable-core
        cpp-httplib)
target_compile_definitions(hlie-worker-supervisor-component-tests PRIVATE
        HLIE_FAKE_WORKER_EXECUTABLE_DEFAULT="$<TARGET_FILE:hlie-fake-worker>")
target_compile_features(hlie-worker-supervisor-component-tests PRIVATE cxx_std_17)
add_dependencies(hlie-worker-supervisor-component-tests hlie-fake-worker)

if (HLIE_TEST_LINK_TIME_LIBRARY_PATH)
    target_link_options(hlie-worker-supervisor-component-tests PRIVATE
            "-Wl,-rpath-link,${HLIE_TEST_LINK_TIME_LIBRARY_PATH}")
endif ()

if (WIN32)
    target_link_libraries(hlie-fake-worker PRIVATE ws2_32)
    target_link_libraries(hlie-worker-supervisor-component-tests PRIVATE ws2_32)
endif ()

add_test(NAME hlie.component.worker-supervisor
        COMMAND hlie-worker-supervisor-component-tests --reporter console)
set_tests_properties(hlie.component.worker-supervisor PROPERTIES
        LABELS "component;worker-supervisor;fake-worker"
        TIMEOUT 30
        ENVIRONMENT "HLIE_FAKE_WORKER_EXECUTABLE=$<TARGET_FILE:hlie-fake-worker>")
if (HLIE_TEST_RUNTIME_LIBRARY_PATH)
    set_property(TEST hlie.component.worker-supervisor APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${HLIE_TEST_RUNTIME_LIBRARY_PATH}")
endif ()

set_target_properties(
        hlie-testable-core
        hlie-unit-tests
        hlie-fake-worker
        hlie-worker-supervisor-component-tests
        PROPERTIES FOLDER "tests")

if (HLIE_BUILD_TCIM_HARDWARE_TESTS)
    find_library(HLIE_TEST_TCIM_RUNTIME
            NAMES tcim_runtime_lite
            PATHS ${PROJECT_SOURCE_DIR}/third_party/houmo/lib
            NO_DEFAULT_PATH REQUIRED)
    find_library(HLIE_TEST_TCIM_DEV_CTRL
            NAMES tcim_dev_ctrl
            PATHS ${PROJECT_SOURCE_DIR}/third_party/houmo/lib
            NO_DEFAULT_PATH REQUIRED)
    find_library(HLIE_TEST_GGML_BASE
            NAMES ggml-base
            PATHS ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/lib
            NO_DEFAULT_PATH REQUIRED)

    add_executable(tcim-module-lifecycle-test
            unit/tcim_module_lifecycle_test.cpp)
    target_include_directories(tcim-module-lifecycle-test PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/houmo/include
            ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include)
    target_link_libraries(tcim-module-lifecycle-test PRIVATE
            ${HLIE_TEST_TCIM_RUNTIME}
            ${HLIE_TEST_TCIM_DEV_CTRL}
            ${HLIE_TEST_GGML_BASE})
    target_compile_features(tcim-module-lifecycle-test PRIVATE cxx_std_17)
    set_target_properties(tcim-module-lifecycle-test PROPERTIES FOLDER "tests/hardware")
endif ()

set(HLIE_MIRAGE_EXECUTABLE "" CACHE FILEPATH
        "Path to the HLAW Mirage executable used by HTTP component tests")
if (HLIE_MIRAGE_EXECUTABLE AND NOT EXISTS "${HLIE_MIRAGE_EXECUTABLE}")
    message(FATAL_ERROR
            "HLIE_MIRAGE_EXECUTABLE does not exist: ${HLIE_MIRAGE_EXECUTABLE}")
endif ()
if (NOT HLIE_MIRAGE_EXECUTABLE)
    find_program(HLIE_MIRAGE_EXECUTABLE_AUTO
            NAMES hlaw-mirage hlaw-mirage.exe
            HINTS ${PROJECT_SOURCE_DIR}/../hlaw-mirage/bin
            NO_DEFAULT_PATH)
    if (HLIE_MIRAGE_EXECUTABLE_AUTO)
        set(HLIE_MIRAGE_EXECUTABLE "${HLIE_MIRAGE_EXECUTABLE_AUTO}" CACHE FILEPATH
                "Path to the HLAW Mirage executable used by HTTP component tests" FORCE)
    endif ()
endif ()

if (HLIE_MIRAGE_EXECUTABLE)
    file(TO_CMAKE_PATH "${HLIE_MIRAGE_EXECUTABLE}" HLIE_MIRAGE_EXECUTABLE_CMAKE_PATH)
    add_executable(hlie-mirage-component-tests
            component/hlaw_mirage_test.cpp
            support/mirage_process.cpp)
    target_include_directories(hlie-mirage-component-tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/support)
    target_link_libraries(hlie-mirage-component-tests PRIVATE
            Catch2::Catch2WithMain
            cpp-httplib)
    target_compile_definitions(hlie-mirage-component-tests PRIVATE
            HLIE_MIRAGE_EXECUTABLE_DEFAULT="${HLIE_MIRAGE_EXECUTABLE_CMAKE_PATH}")
    target_compile_features(hlie-mirage-component-tests PRIVATE cxx_std_17)
    if (WIN32)
        target_link_libraries(hlie-mirage-component-tests PRIVATE ws2_32)
    endif ()

    add_test(NAME hlie.component.mirage
            COMMAND hlie-mirage-component-tests --reporter console)
    set_tests_properties(hlie.component.mirage PROPERTIES
            LABELS "component;mirage"
            TIMEOUT 30
            ENVIRONMENT "HLIE_MIRAGE_EXECUTABLE=${HLIE_MIRAGE_EXECUTABLE}")
    set_target_properties(hlie-mirage-component-tests PROPERTIES FOLDER "tests")
else ()
    message(WARNING
            "HLAW Mirage not found; Mirage HTTP component tests will be skipped. "
            "Set HLIE_MIRAGE_EXECUTABLE to enable them.")
endif ()

# Real-hardware P0 integration is intentionally an explicit build target rather
# than part of the default CTest set. It owns an isolated server process, loads
# real models, injects Worker crashes, and can take several minutes.
if (NOT HLIE_TESTS_ONLY)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    set(HLIE_P0_HARDWARE_PRIMARY_MODEL "" CACHE FILEPATH
            "GGUF used for the main-process model in the P0 hardware smoke")
    set(HLIE_P0_HARDWARE_PRIMARY_MMPROJ "" CACHE FILEPATH
            "Primary multimodal projector used by the P0 hardware smoke")
    set(HLIE_P0_HARDWARE_WORKER_MODEL "" CACHE FILEPATH
            "GGUF used for the dynamic Worker; defaults to the primary GGUF")
    set(HLIE_P0_HARDWARE_WORKER_MMPROJ "" CACHE FILEPATH
            "Optional second multimodal projector used by the dynamic Worker")
    set(HLIE_P0_HARDWARE_PRIMARY_DEVICE "0" CACHE STRING
            "Device used by the main-process P0 model")
    set(HLIE_P0_HARDWARE_WORKER_DEVICE "1" CACHE STRING
            "Device used by the dynamic P0 Worker")
    set(HLIE_P0_HARDWARE_PORT "17901" CACHE STRING
            "Isolated master port used by the P0 hardware smoke")
    set(HLIE_P0_HARDWARE_SOAK_CYCLES "500" CACHE STRING
            "Dynamic Worker load/infer/unload cycles used by the P0 hardware soak")
    set(HLIE_P0_HARDWARE_SOAK_INFERENCE_EVERY "10" CACHE STRING
            "Run one proxy inference every N P0 hardware soak cycles")

    if (WIN32)
        set(HLIE_P0_SERVER_TARGET houmo-llm-hi-cpp-win)
    elseif (UNIX)
        set(HLIE_P0_SERVER_TARGET houmo-llm-hi-cpp-linux)
    endif ()
    set(HLIE_P0_EFFECTIVE_WORKER_MODEL "${HLIE_P0_HARDWARE_WORKER_MODEL}")
    if (NOT HLIE_P0_EFFECTIVE_WORKER_MODEL)
        set(HLIE_P0_EFFECTIVE_WORKER_MODEL "${HLIE_P0_HARDWARE_PRIMARY_MODEL}")
    endif ()

    set(HLIE_P0_WORKER_MMPROJ_ARGS)
    if (HLIE_P0_HARDWARE_WORKER_MMPROJ)
        list(APPEND HLIE_P0_WORKER_MMPROJ_ARGS
                --worker-mmproj ${HLIE_P0_HARDWARE_WORKER_MMPROJ})
    endif ()

    set(HLIE_P0_PRIMARY_MMPROJ_ARGS)
    if (HLIE_P0_HARDWARE_PRIMARY_MMPROJ)
        list(APPEND HLIE_P0_PRIMARY_MMPROJ_ARGS
                --primary-mmproj ${HLIE_P0_HARDWARE_PRIMARY_MMPROJ})
    endif ()

    if (Python3_Interpreter_FOUND AND HLIE_P0_SERVER_TARGET AND
        HLIE_P0_HARDWARE_PRIMARY_MODEL)
        add_custom_target(hlie-p0-hardware-smoke
                COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/python/p0/control_plane_hardware_smoke.py
                    --server-executable $<TARGET_FILE:${HLIE_P0_SERVER_TARGET}>
                    --config-dir ${CMAKE_BINARY_DIR}/bin/config
                    --primary-model ${HLIE_P0_HARDWARE_PRIMARY_MODEL}
                    ${HLIE_P0_PRIMARY_MMPROJ_ARGS}
                    --worker-model ${HLIE_P0_EFFECTIVE_WORKER_MODEL}
                    ${HLIE_P0_WORKER_MMPROJ_ARGS}
                    --primary-device ${HLIE_P0_HARDWARE_PRIMARY_DEVICE}
                    --worker-device ${HLIE_P0_HARDWARE_WORKER_DEVICE}
                    --port ${HLIE_P0_HARDWARE_PORT}
                DEPENDS ${HLIE_P0_SERVER_TARGET}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                USES_TERMINAL
                VERBATIM)
        add_custom_target(hlie-p0-hardware-soak
                COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tests/python/p0/control_plane_hardware_smoke.py
                    --server-executable $<TARGET_FILE:${HLIE_P0_SERVER_TARGET}>
                    --config-dir ${CMAKE_BINARY_DIR}/bin/config
                    --primary-model ${HLIE_P0_HARDWARE_PRIMARY_MODEL}
                    ${HLIE_P0_PRIMARY_MMPROJ_ARGS}
                    --worker-model ${HLIE_P0_EFFECTIVE_WORKER_MODEL}
                    ${HLIE_P0_WORKER_MMPROJ_ARGS}
                    --primary-device ${HLIE_P0_HARDWARE_PRIMARY_DEVICE}
                    --worker-device ${HLIE_P0_HARDWARE_WORKER_DEVICE}
                    --port ${HLIE_P0_HARDWARE_PORT}
                    --skip-pre-ready-crash
                    --soak-cycles ${HLIE_P0_HARDWARE_SOAK_CYCLES}
                    --soak-inference-every ${HLIE_P0_HARDWARE_SOAK_INFERENCE_EVERY}
                DEPENDS ${HLIE_P0_SERVER_TARGET}
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                USES_TERMINAL
                VERBATIM)
    else ()
        add_custom_target(hlie-p0-hardware-smoke
                COMMAND ${CMAKE_COMMAND} -E echo
                    "P0 hardware smoke is not configured. Set HLIE_P0_HARDWARE_PRIMARY_MODEL, then reconfigure."
                COMMAND ${CMAKE_COMMAND} -E false
                USES_TERMINAL
                VERBATIM)
        add_custom_target(hlie-p0-hardware-soak
                COMMAND ${CMAKE_COMMAND} -E echo
                    "P0 hardware soak is not configured. Set HLIE_P0_HARDWARE_PRIMARY_MODEL, then reconfigure."
                COMMAND ${CMAKE_COMMAND} -E false
                USES_TERMINAL
                VERBATIM)
    endif ()
endif ()
