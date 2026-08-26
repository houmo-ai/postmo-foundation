#!/usr/bin/env bash

function update-alternatives-clang() {
    local version=$1
    local priority=$2

    update-alternatives \
        --install /usr/bin/clang                     clang                     /usr/bin/clang-${version} ${priority} \
        --slave   /usr/bin/clang++                   clang++                   /usr/bin/clang++-${version}  \
        --slave   /usr/bin/asan_symbolize            asan_symbolize            /usr/bin/asan_symbolize-${version} \
        --slave   /usr/bin/c-index-test              c-index-test              /usr/bin/c-index-test-${version} \
        --slave   /usr/bin/clang-check               clang-check               /usr/bin/clang-check-${version} \
        --slave   /usr/bin/clang-cl                  clang-cl                  /usr/bin/clang-cl-${version} \
        --slave   /usr/bin/clang-cpp                 clang-cpp                 /usr/bin/clang-cpp-${version} \
        --slave   /usr/bin/clang-format              clang-format              /usr/bin/clang-format-${version} \
        --slave   /usr/bin/clang-format-diff         clang-format-diff         /usr/bin/clang-format-diff-${version} \
        --slave   /usr/bin/clang-include-fixer       clang-include-fixer       /usr/bin/clang-include-fixer-${version} \
        --slave   /usr/bin/clang-offload-bundler     clang-offload-bundler     /usr/bin/clang-offload-bundler-${version} \
        --slave   /usr/bin/clang-query               clang-query               /usr/bin/clang-query-${version} \
        --slave   /usr/bin/clang-rename              clang-rename              /usr/bin/clang-rename-${version} \
        --slave   /usr/bin/clang-reorder-fields      clang-reorder-fields      /usr/bin/clang-reorder-fields-${version} \
        --slave   /usr/bin/clang-tidy                clang-tidy                /usr/bin/clang-tidy-${version} \
        --slave   /usr/bin/clang-apply-replacements  clang-apply-replacements  /usr/bin/clang-apply-replacements-${version} \
        --slave   /usr/bin/clang-change-namespace    clang-change-namespace    /usr/bin/clang-change-namespace-${version} \
        --slave   /usr/bin/clang-doc                 clang-doc                 /usr/bin/clang-doc-${version} \
        --slave   /usr/bin/clang-extdef-mapping      clang-extdef-mapping      /usr/bin/clang-extdef-mapping-${version} \
        --slave   /usr/bin/clang-linker-wrapper      clang-linker-wrapper      /usr/bin/clang-linker-wrapper-${version} \
        --slave   /usr/bin/clang-move                clang-move                /usr/bin/clang-move-${version} \
        --slave   /usr/bin/clang-nvlink-wrapper      clang-nvlink-wrapper      /usr/bin/clang-nvlink-wrapper-${version} \
        --slave   /usr/bin/clang-offload-wrapper     clang-offload-wrapper     /usr/bin/clang-offload-wrapper-${version} \
        --slave   /usr/bin/clang-refactor            clang-refactor            /usr/bin/clang-refactor-${version} \
        --slave   /usr/bin/clang-repl                clang-repl                /usr/bin/clang-repl-${version} \
        --slave   /usr/bin/clang-scan-deps           clang-scan-deps           /usr/bin/clang-scan-deps-${version} \
        --slave   /usr/bin/python                    python3                   /usr/bin/python3.9 \
        --slave   /usr/bin/python                    python3                   /usr/bin/python3.9
}

function update-alternatives-gcc() {
    local version=$1
    local priority=$2

    update-alternatives \
    --install /usr/bin/gcc                       gcc                       /usr/bin/gcc-${version} ${priority} \
	--slave   /usr/bin/gcc-ar                    gcc-ar                    /usr/bin/gcc-ar-${version}  \
	--slave   /usr/bin/gcc-nm                    gcc-nm                    /usr/bin/gcc-nm-${version}  \
	--slave   /usr/bin/gcc-ranlib                gcc-ranlib                /usr/bin/gcc-ranlib-${version}  \
	--slave   /usr/bin/gcov                	     gcov                      /usr/bin/gcov-${version}  \
	--slave   /usr/bin/gcov-dump                 gcov-dump                 /usr/bin/gcov-dump-${version}  \
	--slave   /usr/bin/g++                       g++                       /usr/bin/g++-${version}
}
