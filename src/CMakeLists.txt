set(TARGET houmo-llm)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(TCIM_RUNTIME_PATH ${CMAKE_SOURCE_DIR}/third_party/houmo)
set(LLAMA_CPP_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo")
set(UWEBSOCKET_PATH "${PROJECT_SOURCE_DIR}/third_party/uwebsockets")
set(INIH_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/inih")

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite ai_core hdplrt riscv PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_LLAMA NAMES llama PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_GGML NAMES ggml PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_GGML_BASE NAMES ggml-base PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_GGML_CPU NAMES ggml-cpu PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_COMMON NAMES common PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LLAMA_CPP_LIBRARY_SERVER NAMES server-context PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)


option(HOUMO_XH2 "Enable HOUMO_XH2 feature" OFF)  # 默认关闭
if (HOUMO_XH2)
    add_compile_definitions(HOUMO_XH2)
    message(STATUS "HOUMO_XH2 enabled")
    set(STABLE_DIFFUSION_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/sd.cpp")
    find_library(STABLE_DIFFUSION NAMES stable-diffusion PATHS ${STABLE_DIFFUSION_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    message("STABLE_DIFFUSION_LIBRARY=${STABLE_DIFFUSION}")
    set(HMSYS_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/houmo")
    if (WIN32)
        find_library(HMSYS_LIBRARY NAMES hal_xh2a.dll PATHS ${HMSYS_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    elseif (UNIX)
        find_library(HMSYS_LIBRARY NAMES hal_xh2a PATHS ${HMSYS_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    endif ()
    message("HMSYS_LIBRARY=${HMSYS_LIBRARY}")

    set(WHISPER_CPP_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/whisper.cpp")
    find_library(WHISPER_CPP NAMES whisper PATHS ${WHISPER_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    message("WHISPER_CPP_LIBRARY=${WHISPER_CPP}")

    find_library(UWEBSOCKET_LIBRARY NAMES uSockets.a uSockets PATHS ${UWEBSOCKET_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(UV_LIBRARY NAMES uv PATHS ${UWEBSOCKET_PATH}/lib NO_DEFAULT_PATH REQUIRED)
    message("UWEBSOCKET_LIBRARY=${UWEBSOCKET_LIBRARY}")
    include_directories(${UWEBSOCKET_PATH}/include)
endif ()


find_library(LLAMA_CPP_LIBRARY_MTMD NAMES mtmd PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(INIH_LIBRARY NAMES inih PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(INIH_LIBRARY_CPP NAMES INIReader PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TCIM_RUNTIME_PATH}/include)
include_directories(${TCIM_RUNTIME_PATH}/examples/common/hpp)

include_directories(${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

include_directories(${PROJECT_SOURCE_DIR}/third_party/sd.cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/whisper.cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/houmo/include)

include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

message("LLAMA_CPP_LIBRARY_PATH=${LLAMA_CPP_LIBRARY_PATH}")
message("TCIM_LIBRARY=${TCIM_LIBRARY}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message("CMAKE_COMMAND=${CMAKE_COMMAND}")
message("LLAMA_CPP_LIBRARY_COMMON=${LLAMA_CPP_LIBRARY_COMMON}")
message("LLAMA_CPP_LIBRARY_SERVER=${LLAMA_CPP_LIBRARY_SERVER}")
message("LLAMA_CPP_LIBRARY_MTMD=${LLAMA_CPP_LIBRARY_MTMD}")

if (WINSERVICE_MODE)
    add_compile_definitions(WINSERVICE_MODE)
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
else ()
    add_definitions(-w)
    add_definitions(-std=c++2a)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -g -rdynamic")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_library(utf8_console
        utf8/utf8_console.cpp
        utf8/wcwidth.c
        utf8/utfwidth.cpp
        utf8/utf8_cin_windows.cpp
)

target_include_directories(utf8_console PUBLIC "utf8/include")


add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

add_library(${TARGET} houmo-llm-sdk.cpp
)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR} ../include)


target_link_libraries(${TARGET} PRIVATE ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_MTMD} cpp-httplib ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU} ${INIH_LIBRARY_CPP} ${INIH_LIBRARY} "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${LLAMA_CPP_LIBRARY_SERVER})

target_compile_features(${TARGET} PRIVATE cxx_std_17)


if (WIN32)
    message(STATUS "Building hi-cpp-win for Windows")
    add_executable(${TARGET}-hi-cpp-win main-win.cpp)
    install(TARGETS ${TARGET}-hi-cpp-win RUNTIME)
    target_include_directories(${TARGET}-hi-cpp-win PRIVATE ${CMAKE_SOURCE_DIR})
    if (HOUMO_XH2)
        target_link_libraries(${TARGET}-hi-cpp-win PUBLIC ${STABLE_DIFFUSION})
    endif ()
    target_link_libraries(${TARGET}-hi-cpp-win PUBLIC hlie-app ${CMAKE_THREAD_LIBS_INIT})
    TARGET_LINK_LIBRARIES(${TARGET}-hi-cpp-win PRIVATE ws2_32)
    target_compile_features(${TARGET}-hi-cpp-win PRIVATE cxx_std_17)
elseif (UNIX)
    message(STATUS "Building hi-cpp-linux for Linux/Unix")
    add_executable(${TARGET}-hi-cpp-linux main-linux.cpp)
    install(TARGETS ${TARGET}-hi-cpp-linux RUNTIME)
    target_include_directories(${TARGET}-hi-cpp-linux PRIVATE ${CMAKE_SOURCE_DIR})
    if (HOUMO_XH2)
        target_link_libraries(${TARGET}-hi-cpp-linux PUBLIC ${STABLE_DIFFUSION})
    endif ()
    target_link_libraries(${TARGET}-hi-cpp-linux PUBLIC hlie-app ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET}-hi-cpp-linux PRIVATE cxx_std_17)
else ()
    message(FATAL_ERROR "Unsupported operating system")
endif ()


add_executable(${TARGET}-demo-lenovo demo-lenovo.cpp)
install(TARGETS ${TARGET}-demo-lenovo RUNTIME)
target_include_directories(${TARGET}-demo-lenovo PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-lenovo PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})


if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo PRIVATE ws2_32)
else ()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo PRIVATE -lpthread -lm)
endif ()

target_compile_features(${TARGET}-demo-lenovo PRIVATE cxx_std_17)

#添加demo-hi,基于通用API编写的测试demo
add_executable(${TARGET}-demo-hi demo-hi.cpp)
install(TARGETS ${TARGET}-demo-hi RUNTIME)
target_include_directories(${TARGET}-demo-hi PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-hi PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})


if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-hi PRIVATE ws2_32)
else ()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-hi PRIVATE -lpthread -lm)
endif ()

target_compile_features(${TARGET}-demo-hi PRIVATE cxx_std_17)


#添加demo-lenevo-14B代码
add_executable(${TARGET}-demo-lenovo-14B demo-lenovo-14B.cpp)
install(TARGETS ${TARGET}-demo-lenovo-14B RUNTIME)
target_include_directories(${TARGET}-demo-lenovo-14B PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-lenovo-14B PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})


if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo-14B PRIVATE ws2_32)
else ()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo-14B PRIVATE -lpthread -lm)
endif ()

target_compile_features(${TARGET}-demo-lenovo-14B PRIVATE cxx_std_17)

add_subdirectory(hlie_base)
add_subdirectory(hlie_app)
