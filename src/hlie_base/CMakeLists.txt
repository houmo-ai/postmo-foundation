set(TARGET hlie-base)

set(INIH_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/third_party/inih")

find_library(INIH_LIBRARY_CPP NAMES INIReader PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)
find_library(INIH_LIBRARY NAMES inih PATHS ${INIH_LIBRARY_PATH}/lib NO_DEFAULT_PATH)

include_directories(${PROJECT_SOURCE_DIR}/third_party/inih/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog/include)

add_definitions(-DHLIE_SHARED_LIB_BUILDING)
add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/hlie_base/*.cpp")

add_library(${TARGET} ${SOURCES})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${TARGET} PRIVATE  ${INIH_LIBRARY_CPP} ${INIH_LIBRARY})
target_compile_features(${TARGET} PRIVATE cxx_std_17)