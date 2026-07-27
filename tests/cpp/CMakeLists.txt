add_library(hlie-testable-core STATIC
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_family_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_info_core.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_registry.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_model_residency_manager.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_proxy_timeout.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_document_core.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_document_region_cropper.cpp
        ${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_document_types.cpp
        support/ggml_assert_test_shim.cpp)
target_include_directories(hlie-testable-core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include
        ${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include/nlohmann
        ${PROJECT_SOURCE_DIR}/third_party/spdlog/include)
target_compile_definitions(hlie-testable-core PUBLIC HLIE_STATIC)
target_compile_features(hlie-testable-core PUBLIC cxx_std_17)

add_executable(hlie-unit-tests
        unit/hlie_document_core_test.cpp
        unit/hlie_model_family_registry_test.cpp
        unit/hlie_model_info_test.cpp
        unit/hlie_model_registry_test.cpp
        unit/hlie_model_residency_manager_test.cpp
        unit/hlie_proxy_timeout_test.cpp)
target_link_libraries(hlie-unit-tests PRIVATE
        Catch2::Catch2WithMain
        hlie-testable-core)
target_compile_features(hlie-unit-tests PRIVATE cxx_std_17)

add_test(NAME hlie.unit COMMAND hlie-unit-tests --reporter console)
set_tests_properties(hlie.unit PROPERTIES
        LABELS "unit"
        TIMEOUT 30)

set_target_properties(
        hlie-testable-core
        hlie-unit-tests
        PROPERTIES FOLDER "tests")

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
