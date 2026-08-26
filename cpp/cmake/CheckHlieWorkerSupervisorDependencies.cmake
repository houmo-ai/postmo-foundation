if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(SUPERVISOR_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_worker_supervisor.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_worker_supervisor.cpp")

set(BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_registry"
        "hlie_device_service"
        "hlie_device_reservation"
        "hlie_model_service"
        "server/server-models")

foreach (SUPERVISOR_FILE IN LISTS SUPERVISOR_FILES)
    if (NOT EXISTS "${SUPERVISOR_FILE}")
        message(FATAL_ERROR "WorkerSupervisor source is missing: ${SUPERVISOR_FILE}")
    endif ()
    file(READ "${SUPERVISOR_FILE}" CONTENT)
    foreach (BANNED IN LISTS BANNED_DEPENDENCIES)
        string(FIND "${CONTENT}" "${BANNED}" MATCH_INDEX)
        if (NOT MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "WorkerSupervisor dependency boundary violation: ${SUPERVISOR_FILE} contains ${BANNED}")
        endif ()
    endforeach ()
endforeach ()

file(READ "${HLIE_SOURCE_DIR}/include/hlie_app/runtime/hlie_worker_supervisor.h" PUBLIC_HEADER)
foreach (PUBLIC_BANNED IN ITEMS "httplib" "nlohmann" "hlie_process_manager")
    string(FIND "${PUBLIC_HEADER}" "${PUBLIC_BANNED}" MATCH_INDEX)
    if (NOT MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "WorkerSupervisor public header leaks implementation dependency: ${PUBLIC_BANNED}")
    endif ()
endforeach ()

message(STATUS "WorkerSupervisor dependency boundary check passed")
