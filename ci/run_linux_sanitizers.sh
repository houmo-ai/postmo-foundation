#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:-all}
CMAKE_BIN=${CMAKE_BIN:-cmake}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

select_compiler_pair() {
    if [[ -n "${HLIE_SANITIZER_CC:-}" || -n "${HLIE_SANITIZER_CXX:-}" ]]; then
        if [[ -z "${HLIE_SANITIZER_CC:-}" || -z "${HLIE_SANITIZER_CXX:-}" ]]; then
            echo "HLIE_SANITIZER_CC and HLIE_SANITIZER_CXX must be set together" >&2
            exit 2
        fi
        SANITIZER_CC=${HLIE_SANITIZER_CC}
        SANITIZER_CXX=${HLIE_SANITIZER_CXX}
        return
    fi

    # C and C++ must use the same sanitizer runtime. Some build hosts expose
    # cc=gcc-9 and c++=g++-13, which otherwise loads two incompatible libasan
    # versions into one process.
    local version
    for version in {20..9}; do
        if command -v "gcc-${version}" >/dev/null 2>&1 &&
           command -v "g++-${version}" >/dev/null 2>&1; then
            SANITIZER_CC=$(command -v "gcc-${version}")
            SANITIZER_CXX=$(command -v "g++-${version}")
            return
        fi
    done
    if command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
        SANITIZER_CC=$(command -v clang)
        SANITIZER_CXX=$(command -v clang++)
        return
    fi

    SANITIZER_CC=$(command -v cc)
    SANITIZER_CXX=$(command -v c++)
}

select_compiler_pair
CMAKE_COMPILER_ARGS=(
    "-DCMAKE_C_COMPILER=${SANITIZER_CC}"
    "-DCMAKE_CXX_COMPILER=${SANITIZER_CXX}"
)
echo "==> Sanitizer compiler pair: CC=${SANITIZER_CC}, CXX=${SANITIZER_CXX}"

prepare_build_directory() {
    local build_dir=$1
    local cache="${build_dir}/CMakeCache.txt"
    # A fresh sanitizer build has no cache yet; that is the normal path, not
    # an error under the script's `set -e` policy.
    [[ -f "${cache}" ]] || return 0

    local cached_cc cached_cxx
    cached_cc=$(grep '^CMAKE_C_COMPILER:' "${cache}" | head -1 | cut -d= -f2- || true)
    cached_cxx=$(grep '^CMAKE_CXX_COMPILER:' "${cache}" | head -1 | cut -d= -f2- || true)
    if [[ -n "${cached_cc}" && -n "${cached_cxx}" ]] &&
       { [[ "$(readlink -f "${cached_cc}")" != "$(readlink -f "${SANITIZER_CC}")" ]] ||
         [[ "$(readlink -f "${cached_cxx}")" != "$(readlink -f "${SANITIZER_CXX}")" ]]; }; then
        echo "==> Compiler pair changed; recreating ${build_dir}"
        "${CMAKE_BIN}" -E remove_directory "${build_dir}"
    fi
}

usage() {
    echo "Usage: $0 [all|asan-ubsan|tsan]"
}

run_asan_ubsan() {
    local build_dir="${REPO_ROOT}/cmake-build-sanitizers-asan-ubsan"
    prepare_build_directory "${build_dir}"
    echo "==> Configuring ASAN+UBSAN in ${build_dir}"
    "${CMAKE_BIN}" -S "${REPO_ROOT}" -B "${build_dir}" \
        "${CMAKE_COMPILER_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DHLIE_BUILD_TESTS=ON \
        -DHLIE_SANITIZE_ADDRESS=ON \
        -DHLIE_SANITIZE_UNDEFINED=ON \
        -DHLIE_SANITIZE_THREAD=OFF

    "${CMAKE_BIN}" --build "${build_dir}" \
        --target hlie-unit-tests hlie-worker-supervisor-component-tests \
        --parallel "${BUILD_JOBS}"

    ASAN_OPTIONS=${ASAN_OPTIONS:-halt_on_error=1:abort_on_error=1:detect_leaks=1} \
    UBSAN_OPTIONS=${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1} \
        ctest --test-dir "${build_dir}" --output-on-failure \
        -R '^(hlie\.unit|hlie\.stress\.control-plane|hlie\.component\.worker-supervisor)$'
}

run_tsan() {
    local build_dir="${REPO_ROOT}/cmake-build-sanitizers-tsan"
    prepare_build_directory "${build_dir}"
    echo "==> Configuring proprietary-SDK-free TSAN control-plane tests in ${build_dir}"
    "${CMAKE_BIN}" -S "${REPO_ROOT}" -B "${build_dir}" \
        "${CMAKE_COMPILER_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DHLIE_TESTS_ONLY=ON \
        -DHLIE_SANITIZE_ADDRESS=OFF \
        -DHLIE_SANITIZE_UNDEFINED=OFF \
        -DHLIE_SANITIZE_THREAD=ON

    "${CMAKE_BIN}" --build "${build_dir}" \
        --target hlie-unit-tests \
        --parallel "${BUILD_JOBS}"

    TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1} \
        ctest --test-dir "${build_dir}" --output-on-failure
}

case "${MODE}" in
    all)
        run_asan_ubsan
        run_tsan
        ;;
    asan-ubsan)
        run_asan_ubsan
        ;;
    tsan)
        run_tsan
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
