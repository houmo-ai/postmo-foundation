set(TARGET hlie-core)

# Keep the production core source list explicit. P1 must not accidentally pull
# HTTP, process, model-service, or SDK implementations into this target.
set(SOURCES
        hlie_backend_registry.cpp
        hlie_device_reservation_ledger.cpp
        hlie_model_family_registry.cpp
        hlie_model_runtime_registry.cpp
        hlie_model_residency_manager.cpp)

add_library(${TARGET} STATIC ${SOURCES})
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${TARGET} PUBLIC
        ${PROJECT_SOURCE_DIR}/include)
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TARGET} PRIVATE Threads::Threads)

# Full production builds preserve the intended app -> core -> base dependency.
# HLIE_TESTS_ONLY deliberately omits the proprietary base target; this first
# core slice only consumes the header-only namespace/export macros from base.
if (TARGET hlie-base)
    target_link_libraries(${TARGET} PRIVATE hlie-base)
else ()
    target_compile_definitions(${TARGET} PUBLIC HLIE_STATIC)
endif ()
