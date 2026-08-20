cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("hi.cpp" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Linux packages place executables in bin/ and all shared libraries in lib/.
# Use the install RPATH for build artifacts too because the packaging scripts
# copy directly from the build tree instead of running cmake --install.
if (UNIX AND NOT APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
endif ()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/config)
configure_file(
        ${CMAKE_SOURCE_DIR}/config/config_schema.json
        ${CMAKE_BINARY_DIR}/bin/config/config_schema.json
        COPYONLY
)
configure_file(
        ${CMAKE_SOURCE_DIR}/config/model_presets.json
        ${CMAKE_BINARY_DIR}/bin/config/model_presets.json
        COPYONLY
)
configure_file(
        ${CMAKE_SOURCE_DIR}/config/model_overrides.json
        ${CMAKE_BINARY_DIR}/bin/config/model_overrides.json
        COPYONLY
)
configure_file(
        ${CMAKE_SOURCE_DIR}/config/memory_estimates.json
        ${CMAKE_BINARY_DIR}/bin/config/memory_estimates.json
        COPYONLY
)

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/wd4251>") # 同一个项目，没事
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)


# Required for relocatable CMake package
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

option(HLIE_SANITIZE_ADDRESS "Build hicpp targets with AddressSanitizer" OFF)
option(HLIE_SANITIZE_UNDEFINED "Build hicpp targets with UndefinedBehaviorSanitizer" OFF)
option(HLIE_SANITIZE_THREAD "Build hicpp targets with ThreadSanitizer" OFF)

# Keep the inherited llama.cpp switches working while exposing hicpp-owned
# names to CI and developers.
if (LLAMA_SANITIZE_ADDRESS)
    set(HLIE_SANITIZE_ADDRESS ON)
endif ()
if (LLAMA_SANITIZE_UNDEFINED)
    set(HLIE_SANITIZE_UNDEFINED ON)
endif ()
if (LLAMA_SANITIZE_THREAD)
    set(HLIE_SANITIZE_THREAD ON)
endif ()

if (HLIE_SANITIZE_THREAD AND (HLIE_SANITIZE_ADDRESS OR HLIE_SANITIZE_UNDEFINED))
    message(FATAL_ERROR "ThreadSanitizer must use a separate build from ASAN/UBSAN")
endif ()

if (HLIE_SANITIZE_ADDRESS OR HLIE_SANITIZE_UNDEFINED OR HLIE_SANITIZE_THREAD)
    if (MSVC)
        message(FATAL_ERROR "HLIE sanitizer builds currently require GCC or Clang on Linux")
    endif ()

    set(HLIE_ENABLED_SANITIZERS "")
    if (HLIE_SANITIZE_ADDRESS)
        list(APPEND HLIE_ENABLED_SANITIZERS address)
    endif ()
    if (HLIE_SANITIZE_UNDEFINED)
        list(APPEND HLIE_ENABLED_SANITIZERS undefined)
    endif ()
    if (HLIE_SANITIZE_THREAD)
        list(APPEND HLIE_ENABLED_SANITIZERS thread)
    endif ()
    list(JOIN HLIE_ENABLED_SANITIZERS "," HLIE_SANITIZER_LIST)
    set(HLIE_SANITIZER_FLAG "-fsanitize=${HLIE_SANITIZER_LIST}")
    message(STATUS "Using hicpp sanitizers: ${HLIE_SANITIZER_LIST}")

    add_compile_options(${HLIE_SANITIZER_FLAG} -fno-omit-frame-pointer)
    add_link_options(${HLIE_SANITIZER_FLAG})
    if (HLIE_SANITIZE_UNDEFINED)
        add_compile_options(-fno-sanitize-recover=undefined)
        add_link_options(-fno-sanitize-recover=undefined)
    endif ()
endif ()

#
# build the library
#

option(BUILD_TESTING "Build automated tests" OFF)
option(HLIE_BUILD_TESTS "Compatibility alias for BUILD_TESTING" OFF)
option(HLIE_TESTS_ONLY "Configure test targets without proprietary inference dependencies" OFF)
option(HLIE_BUILD_TCIM_HARDWARE_TESTS "Build direct TCIM hardware lifecycle tests" OFF)

if (HLIE_BUILD_TESTS OR HLIE_TESTS_ONLY)
    set(BUILD_TESTING ON CACHE BOOL "Build automated tests" FORCE)
endif ()

include(CTest)
find_package(Threads REQUIRED)

# cpp-httplib is lightweight and is also used by HTTP component tests. Define it
# before production targets so target resolution is deterministic.
add_subdirectory(third_party/cpp-httplib)

# Keep gzip support self-contained and reproducible on Windows and Linux. zlib
# is vendored at a fixed release and linked statically into hlie-base, so no
# zlib DLL/shared-library is required on deployment hosts (Ubuntu 20.04+).
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "Build zlib shared library" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "Build zlib static library" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "Build zlib tests" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "Install zlib" FORCE)
add_subdirectory(third_party/zlib EXCLUDE_FROM_ALL)
set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT HLIE_TESTS_ONLY)
    add_subdirectory(src)
else ()
    # Build the same production hlie-core implementation for the SDK-free
    # sanitizer/control-plane slice instead of hand-copying its sources in tests.
    add_subdirectory(src/hlie_core)
endif ()

if (BUILD_TESTING)
    add_subdirectory(third_party/Catch2 EXCLUDE_FROM_ALL)
    add_subdirectory(tests/cpp)
endif ()


