add_library(hlie-catch2-with-main STATIC catch_amalgamated.cpp)
add_library(Catch2::Catch2WithMain ALIAS hlie-catch2-with-main)

target_include_directories(hlie-catch2-with-main SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(hlie-catch2-with-main PUBLIC cxx_std_14)

set_target_properties(hlie-catch2-with-main PROPERTIES
        FOLDER "third_party"
        POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    target_compile_options(hlie-catch2-with-main PRIVATE /wd4061 /wd4242 /wd4244 /wd4702)
else ()
    target_compile_options(hlie-catch2-with-main PRIVATE -w)
endif ()
