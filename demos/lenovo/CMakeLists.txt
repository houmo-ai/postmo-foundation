set(TARGET houmo-llm-demo)

set(HOUMO_LLM_SDK_PATH $ENV{HOUMO_LLM_SDK_PATH})
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
    add_compile_options("/utf-8")
endif ()


find_library(HOUMO_LLM_LIBRARY NAMES houmo-llm tcim_runtime_lite ai_core hdplrt riscv PATHS ${HOUMO_LLM_SDK_PATH}/lib NO_DEFAULT_PATH)

include_directories(${HOUMO_LLM_SDK_PATH}/include)

message("HOUMO_LLM_LIBRARY=${HOUMO_LLM_LIBRARY}")

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -rdynamic")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggerganov/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

add_library(utf8_console
    utils/utf8_console.cpp

    utils/wcwidth.c
    utils/utfwidth.cpp
    utils/utf8_cin_windows.cpp
)

target_include_directories(utf8_console PUBLIC "utils/include")

add_executable(${TARGET} demo-lenovo.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE utf8_console ${HOUMO_LLM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)