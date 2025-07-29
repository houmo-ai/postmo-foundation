set(TARGET houmo-llm)

#cmake -DENABLE_HMLLAMACPP=OFF .. 加载公版llama.cpp
option(ENABLE_HMLLAMACPP "Enable Houmo LLama.cpp" ON)  # 默认编译

if(ENABLE_HMLLAMACPP)
    add_definitions(-DENABLE_HOUMO_LLAMACPP=1)
else()
    add_definitions(-DENABLE_HOUMO_LLAMACPP=0)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
if(ENABLE_HMLLAMACPP)
set(LLAMA_CPP_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo")
else()
set(LLAMA_CPP_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/llama.cpp")
endif()
set(INIH_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/inih")

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite ai_core hdplrt riscv PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_LLAMA NAMES llama PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML NAMES ggml PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML_BASE NAMES ggml-base PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_GGML_CPU NAMES ggml-cpu PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(LLAMA_CPP_LIBRARY_COMMON NAMES common PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
if(ENABLE_HMLLAMACPP)
find_library(LLAMA_CPP_LIBRARY_MTMD NAMES mtmd PATHS ${LLAMA_CPP_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
endif()
find_library(INIH_LIBRARY NAMES inih PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(INIH_LIBRARY_CPP NAMES INIReader PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TCIM_RUNTIME_PATH}/include)
include_directories(${TCIM_RUNTIME_PATH}/examples/common/hpp)
if(ENABLE_HMLLAMACPP)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/llama.cpp_houmo/include)
else()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/llama.cpp/include)
endif()
include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

message("LLAMA_CPP_PATH=${LLAMA_CPP_PATH}")
message("TCIM_LIBRARY=${TCIM_LIBRARY}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message("CMAKE_COMMAND=${CMAKE_COMMAND}")
message("LLAMA_CPP_LIBRARY=${LLAMA_CPP_LIBRARY}")
message("LLAMA_CPP_LIBRARY_COMMON=${LLAMA_CPP_LIBRARY_COMMON}")

option(HOUMO_M50 "Enable HOUMO_M50 feature" OFF)  # 默认关闭

if(HOUMO_M50)
    add_compile_definitions(HOUMO_M50)  # 定义宏 -HOUMO_M50
    message(STATUS "HOUMO_M50 enabled")
endif()


if(WINSERVICE_MODE)
    add_compile_definitions(WINSERVICE_MODE)
endif()

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -rdynamic")
endif()

if(UNIX)
message(STATUS "##Linking libstdc++.so under path(if exists):/opt/hicpp/lib/ for Linux/Unix")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L/opt/hicpp/lib/ -lstdc++ ")
endif()

if(ENABLE_HMLLAMACPP)
message(STATUS "##Linking ${LLAMA_CPP_LIBRARY_MTMD}   for mtmd")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${LLAMA_CPP_LIBRARY_PATH}/lib -lmtmd ")
endif()

find_package(Torch REQUIRED)

option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggerganov/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()


add_library(utf8_console
    utf8/utf8_console.cpp
    utf8/wcwidth.c
    utf8/utfwidth.cpp
    utf8/utf8_cin_windows.cpp
)

target_include_directories(utf8_console PUBLIC "utf8/include")


set(HOUMO_LLM_SRCS
    hi.h
    houmo-llm.h
    houmo-llm.cpp
)

add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

add_library(${TARGET} ${HOUMO_LLM_SRCS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR} ../include)

if(ENABLE_HMLLAMACPP)
target_link_libraries(${TARGET} PRIVATE hlie-base ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_MTMD} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU}  ${INIH_LIBRARY_CPP}  ${INIH_LIBRARY}  "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
else()
target_link_libraries(${TARGET} PRIVATE hlie-base ${LLAMA_CPP_LIBRARY_COMMON}  ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU}  ${INIH_LIBRARY_CPP}  ${INIH_LIBRARY}  "${TORCH_LIBRARIES}"  ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)


if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET}-hi-cpp-win PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET}-hi-cpp-win PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    message(STATUS "Building hi-cpp-win for Windows")
    add_executable(${TARGET}-hi-cpp-win main-win.cpp)
    install(TARGETS ${TARGET}-hi-cpp-win RUNTIME)
    target_include_directories(${TARGET}-hi-cpp-win PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${TARGET}-hi-cpp-win PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU}  ${INIH_LIBRARY_CPP}  ${INIH_LIBRARY}  "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
    TARGET_LINK_LIBRARIES(${TARGET}-hi-cpp-win PRIVATE ws2_32)
    target_compile_features(${TARGET}-hi-cpp-win PRIVATE cxx_std_17)
elseif(UNIX)
    message(STATUS "Building hi-cpp-linux for Linux/Unix")
    add_executable(${TARGET}-hi-cpp-linux main-linux.cpp)
    install(TARGETS ${TARGET}-hi-cpp-linux RUNTIME)
    target_include_directories(${TARGET}-hi-cpp-linux PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${TARGET}-hi-cpp-linux PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE}
        ${LLAMA_CPP_LIBRARY_GGML_CPU}  ${INIH_LIBRARY_CPP}  ${INIH_LIBRARY}  "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

    TARGET_LINK_LIBRARIES(${TARGET}-hi-cpp-linux PRIVATE -lpthread -lm)
    target_compile_features(${TARGET}-hi-cpp-linux PRIVATE cxx_std_17)
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()



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

#添加demo-hi,基于通用API编写的测试demo
add_executable(${TARGET}-demo-hi demo-hi.cpp)
install(TARGETS ${TARGET}-demo-hi RUNTIME)
target_include_directories(${TARGET}-demo-hi PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET}-demo-hi PRIVATE houmo-llm ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${LLAMA_CPP_LIBRARY_GGML_CPU} utf8_console "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET}-demo-hi PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET}-demo-hi PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-demo-hi PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-demo-hi PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-demo-hi PRIVATE cxx_std_17)


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
        ${LLAMA_CPP_LIBRARY_GGML_CPU}  ${INIH_LIBRARY_CPP}  ${INIH_LIBRARY}  "${TORCH_LIBRARIES}" ${TCIM_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} )

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET}-modelinference_test PRIVATE ws2_32)
else()
    TARGET_LINK_LIBRARIES(${TARGET}-modelinference_test PRIVATE -lpthread -lm)
endif()

target_compile_features(${TARGET}-modelinference_test PRIVATE cxx_std_17)

add_subdirectory(hlie_base)
