set(TARGET hlie-base)


include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

add_definitions(-DHLIE_SHARED_LIB_BUILDING)
add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_base/*.cpp")

add_library(${TARGET} ${SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})


target_link_libraries(${TARGET} PRIVATE  ${INIH_LIBRARY_CPP} ${INIH_LIBRARY} ${LLAMA_CPP_LIBRARY_GGML} ${LLAMA_CPP_LIBRARY_GGML_BASE} ${TCIM_LIBRARY} ${KALDI_LIBRARY})
target_compile_features(${TARGET} PRIVATE cxx_std_17)