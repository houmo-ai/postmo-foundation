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

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/config)
configure_file(
        ${CMAKE_SOURCE_DIR}/config/config_schema.json
        ${CMAKE_BINARY_DIR}/bin/config/config_schema.json
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

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#
# build the library
#

option(BUILD_TESTING "Build automated tests" OFF)
option(HLIE_BUILD_TESTS "Compatibility alias for BUILD_TESTING" OFF)
option(HLIE_TESTS_ONLY "Configure test targets without proprietary inference dependencies" OFF)

if (HLIE_BUILD_TESTS OR HLIE_TESTS_ONLY)
    set(BUILD_TESTING ON CACHE BOOL "Build automated tests" FORCE)
endif ()

include(CTest)

# cpp-httplib is lightweight and is also used by HTTP component tests. Define it
# before production targets so target resolution is deterministic.
add_subdirectory(third_party/cpp-httplib)

if (NOT HLIE_TESTS_ONLY)
    add_subdirectory(src)
endif ()

if (BUILD_TESTING)
    add_subdirectory(third_party/Catch2 EXCLUDE_FROM_ALL)
    add_subdirectory(tests/cpp)
endif ()


