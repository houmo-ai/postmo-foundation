if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

file(GLOB_RECURSE HLIE_CORE_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_core/*.h"
        "${HLIE_SOURCE_DIR}/include/hlie_core/*.hpp"
        "${HLIE_SOURCE_DIR}/src/hlie_core/*.cpp")

set(HLIE_CORE_FORBIDDEN_INCLUDE_PATTERN
        "hlie_app|server/|tcim|ggml|httplib|uwebsockets|hlie_model_info")
set(HLIE_CORE_VIOLATIONS "")

foreach (HLIE_CORE_FILE IN LISTS HLIE_CORE_FILES)
    file(STRINGS "${HLIE_CORE_FILE}" HLIE_CORE_INCLUDE_LINES
            REGEX "^[ \t]*#[ \t]*include")
    foreach (HLIE_CORE_INCLUDE_LINE IN LISTS HLIE_CORE_INCLUDE_LINES)
        string(TOLOWER "${HLIE_CORE_INCLUDE_LINE}" HLIE_CORE_INCLUDE_LINE_LOWER)
        if (HLIE_CORE_INCLUDE_LINE_LOWER MATCHES
                "${HLIE_CORE_FORBIDDEN_INCLUDE_PATTERN}")
            file(RELATIVE_PATH HLIE_CORE_RELATIVE_FILE
                    "${HLIE_SOURCE_DIR}" "${HLIE_CORE_FILE}")
            string(APPEND HLIE_CORE_VIOLATIONS
                    "\n  ${HLIE_CORE_RELATIVE_FILE}: ${HLIE_CORE_INCLUDE_LINE}")
        endif ()
    endforeach ()
endforeach ()

if (HLIE_CORE_VIOLATIONS)
    message(FATAL_ERROR
            "hlie-core contains forbidden reverse/concrete dependencies:${HLIE_CORE_VIOLATIONS}")
endif ()

list(LENGTH HLIE_CORE_FILES HLIE_CORE_FILE_COUNT)
message(STATUS "hlie-core dependency audit passed (${HLIE_CORE_FILE_COUNT} files)")
