set(TARGET hlie-app)

include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

add_definitions(-DHLIE_SHARED_LIB_BUILDING)
add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_app/*.cpp")

if(NOT HOUMO_XH2)
    list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_app/hlie_sd_model_service.cpp")
endif ()

add_library(${TARGET} ${SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET} PRIVATE ${STABLE_DIFFUSION} ${TCIM_LIBRARY} ${LLAMA_CPP_LIBRARY_GGML_CPU})

target_link_libraries(${TARGET} PRIVATE hlie-base ${STABLE_DIFFUSION} ${INIH_LIBRARY_CPP} ${INIH_LIBRARY} ${LLAMA_CPP_LIBRARY_COMMON}
        ${LLAMA_CPP_LIBRARY_LLAMA} ${LLAMA_CPP_LIBRARY_MTMD} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE})
target_compile_features(${TARGET} PRIVATE cxx_std_17)