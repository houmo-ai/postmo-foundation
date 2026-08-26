if (NOT DEFINED HLIE_SOURCE_DIR)
    message(FATAL_ERROR "HLIE_SOURCE_DIR is required")
endif ()

set(HLIE_APP_CATALOG_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/catalog/hlie_model_catalog.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_catalog.cpp")
set(HLIE_APP_CATALOG_BANNED_DEPENDENCIES
        "hlie_workflow"
        "hlie_model_controller"
        "hlie_model_registry"
        "hlie_model_runtime_registry"
        "hlie_process_manager"
        "hlie_device_service"
        "hlie_config_service"
        "singleton<"
        "filesystem"
        "httplib"
        "tcim"
        "ggml"
        "hlie_gguf")

foreach (HLIE_FILE IN LISTS HLIE_APP_CATALOG_FILES)
    if (NOT EXISTS "${HLIE_FILE}")
        message(FATAL_ERROR "ModelCatalog source is missing: ${HLIE_FILE}")
    endif ()
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(TOLOWER "${HLIE_CONTENT}" HLIE_CONTENT_LOWER)
    foreach (HLIE_BANNED IN LISTS HLIE_APP_CATALOG_BANNED_DEPENDENCIES)
        string(FIND "${HLIE_CONTENT_LOWER}" "${HLIE_BANNED}" HLIE_MATCH_INDEX)
        if (NOT HLIE_MATCH_INDEX EQUAL -1)
            message(FATAL_ERROR
                    "ModelCatalog dependency boundary violation: ${HLIE_FILE} contains ${HLIE_BANNED}")
        endif ()
    endforeach ()
endforeach ()

set(HLIE_REGISTRY_FILES
        "${HLIE_SOURCE_DIR}/include/hlie_app/hlie_model_registry.h"
        "${HLIE_SOURCE_DIR}/src/hlie_app/hlie_model_registry.cpp")
foreach (HLIE_FILE IN LISTS HLIE_REGISTRY_FILES)
    file(READ "${HLIE_FILE}" HLIE_CONTENT)
    string(FIND "${HLIE_CONTENT}" "catalog_entries_" HLIE_MATCH_INDEX)
    if (NOT HLIE_MATCH_INDEX EQUAL -1)
        message(FATAL_ERROR
                "ModelRegistry still owns static catalog storage: ${HLIE_FILE}")
    endif ()
endforeach ()

message(STATUS "ModelCatalog dependency and Registry ownership checks passed")
