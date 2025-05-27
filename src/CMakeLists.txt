set(TARGET houmo-llm)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
set(LLAMA_CPP_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/llama.cpp")
set(INIH_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/inih")

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite ai_core hdplrt riscv PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_LLAMA NAMES llama PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML NAMES ggml PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML_BASE NAMES ggml-base PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML_CPU NAMES ggml-cpu PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_COMMON NAMES common PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(INIH_LIBRARY NAMES inih PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(INIH_LIBRARY_CPP NAMES INIReader PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)

include_directories(${TCIM_RUNTIME_PATH}/include)
include_directories(${TCIM_RUNTIME_PATH}/examples/common/hpp)
include_directories(${PROJECT_SOURCE_DIR}/third_party/llama.cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)

message("LLAMA_CPP_PATH=${LLAMA_CPP_PATH}")
message("TCIM_LIBRARY=${TCIM_LIBRARY}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message("CMAKE_COMMAND=${CMAKE_COMMAND}")
message("LLAMA_CPP_LIBRARY=${LLAMA_CPP_LIBRARY}")
message("LLAMA_CPP_LIBRARY_COMMON=${LLAMA_CPP_LIBRARY_COMMON}")


if(WINSERVICE_MODE)
    add_compile_definitions(WINSERVICE_MODE)
endif()

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -rdynamic")
endif()

find_package(Torch REQUIRED)

option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)

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

set(TARGET_SRCS

    utils.hpp
    httplib.h
)

set(HOUMO_LLM_SRCS
    houmo-llm.h
    houmo-llm.cpp
)

set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)


add_library(${TARGET} ${HOUMO_LLM_SRCS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET} PRIVATE ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU} ${INIH_LIBRARY} ${INIH_LIBRARY_CPP} "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(${TARGET}-hi-cpp-win main-win.cpp)
install(TARGETS ${TARGET}-hi-cpp-win RUNTIME)
target_include_directories(${TARGET}-hi-cpp-win PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-hi-cpp-win PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU} ${INIH_LIBRARY} ${INIH_LIBRARY_CPP} "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET}-hi-cpp-win PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET}-hi-cpp-win PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-hi-cpp-win PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-hi-cpp-win PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-hi-cpp-win PRIVATE cxx_std_17)

# add_executable(${TARGET}-server ${TARGET_SRCS})
# install(TARGETS ${TARGET}-server RUNTIME)
# target_include_directories(${TARGET}-server PRIVATE ${CMAKE_SOURCE_DIR})
# target_link_libraries(${TARGET}-server PRIVATE ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

# if (LLAMA_SERVER_SSL)
#     find_package(OpenSSL REQUIRED)
#     target_link_libraries(${TARGET}-server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
#     target_compile_definitions(${TARGET}-server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
# endif()

# if (WIN32)
#     TARGET_LINK_LIBRARIES(${TARGET}-server PRIVATE ws2_32)
# else()
#     TARGET_LINK_LIBRARIES(${TARGET}-server PRIVATE -lpthread -lm)
# endif()

# target_compile_features(${TARGET}-server PRIVATE cxx_std_17)

add_executable(${TARGET}-demo-lenovo demo-lenovo.cpp)
install(TARGETS ${TARGET}-demo-lenovo RUNTIME)
target_include_directories(${TARGET}-demo-lenovo PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-lenovo PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET}-demo-lenovo PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET}-demo-lenovo PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-demo-lenovo PRIVATE cxx_std_17)

#添加demo-lenevo-14B代码
add_executable(${TARGET}-demo-lenovo-14B demo-lenovo-14B.cpp)
install(TARGETS ${TARGET}-demo-lenovo-14B RUNTIME)
target_include_directories(${TARGET}-demo-lenovo-14B PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-lenovo-14B PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET}-demo-lenovo-14B PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET}-demo-lenovo-14B PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo-14B PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-lenovo-14B PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-demo-lenovo-14B PRIVATE cxx_std_17)

# 添加deepseek模型推理可执行程序
# add_subdirectory(tokenizers-cpp)
add_executable(${TARGET}-modelinference_test model_inference_test.cpp)
install(TARGETS ${TARGET}-modelinference_test RUNTIME)
target_include_directories(${TARGET}-modelinference_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-modelinference_test PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU} ${INIH_LIBRARY} ${INIH_LIBRARY_CPP} "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} )

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-modelinference_test PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-modelinference_test PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-modelinference_test PRIVATE cxx_std_17)