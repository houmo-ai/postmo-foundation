set(TARGET hlie-app)

include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

add_definitions(-DHLIE_SHARED_LIB_BUILDING)
add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_app/*.cpp")
file(GLOB MODEL_SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_app/models/*.cpp")
list(APPEND SOURCES ${MODEL_SOURCES})

add_library(${TARGET} ${SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET} PUBLIC ${STABLE_DIFFUSION} ${WHISPER_CPP} ${UWEBSOCKET_LIBRARY} ${UV_LIBRARY} ${SAMPLERATE_LIBRARY})
target_link_libraries(${TARGET} PRIVATE ${LLAMA_CPP_LIBRARY_SERVER} PUBLIC hlie-base cpp-httplib ${HMSYS_LIBRARY} ${TCIM_LIBRARY}  ${INIH_LIBRARY_CPP} ${INIH_LIBRARY}
        ${LLAMA_CPP_LIBRARY_COMMON} ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_MTMD} ${LLAMA_CPP_LIBRARY_GGML_CPU} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE})
target_compile_features(${TARGET} PRIVATE cxx_std_17)