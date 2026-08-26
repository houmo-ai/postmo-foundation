# C++ tests

The C++ test suite uses the vendored Catch2 framework and runs through CTest.
The fast path deliberately avoids configuring or linking proprietary inference
libraries. It builds the control-plane subset used by TSAN:

```powershell
cmake -S . -B build-tests -DHLIE_TESTS_ONLY=ON
cmake --build build-tests --config Release --target hlie-unit-tests
ctest --test-dir build-tests -C Release --output-on-failure
```

For a normal product build with tests enabled:

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CTest labels separate fast unit tests from optional component tests:

```powershell
ctest --test-dir build-tests -C Release -L unit
ctest --test-dir build-tests -C Release -L component
ctest --test-dir build-tests -C Release -L stress
```

`hlie.stress.control-plane` exercises the ServiceSlot/Registry lifecycle and
device reservation ledger under thousands of concurrent interleavings. It is
part of the normal CTest set and does not require accelerator hardware.

## Linux sanitizers

The rootless sanitizer runner keeps ASAN/UBSAN and TSAN in separate build
directories. ASAN+UBSAN covers the normal unit and Fake Worker component tests;
TSAN uses `HLIE_TESTS_ONLY=ON` so proprietary TCIM/HAL/inference libraries are
not loaded into the instrumented process:

```bash
bash ci/run_linux_sanitizers.sh all
bash ci/run_linux_sanitizers.sh asan-ubsan
bash ci/run_linux_sanitizers.sh tsan
```

The equivalent CMake switches are `HLIE_SANITIZE_ADDRESS`,
`HLIE_SANITIZE_UNDEFINED`, and `HLIE_SANITIZE_THREAD`. TSAN must use a separate
build from ASAN/UBSAN. The runner automatically selects a matching GCC C/C++
pair; override it with `HLIE_SANITIZER_CC` and `HLIE_SANITIZER_CXX` when needed.

If the sibling `hlaw-mirage/bin/hlaw-mirage(.exe)` is present, CMake adds the
Mirage HTTP component suite automatically. An explicit binary can be selected
with:

```powershell
cmake -S . -B build-tests `
  -DHLIE_TESTS_ONLY=ON `
  -DHLIE_MIRAGE_EXECUTABLE=D:/tools/hlaw-mirage.exe
```

Each component test launches an isolated Mirage process on an OS-assigned port;
it never depends on a shared developer instance.

## Real-hardware P0 smoke

The real-hardware gate is explicit and is not included in the default CTest
set. Configure the model paths, then build `hlie-p0-hardware-smoke`:

```powershell
cmake -S . -B build `
  -DBUILD_TESTING=ON `
  -DHLIE_P0_HARDWARE_PRIMARY_MODEL=E:/models/model.gguf `
  -DHLIE_P0_HARDWARE_PRIMARY_MMPROJ=E:/models/mmproj_448.gguf `
  -DHLIE_P0_HARDWARE_WORKER_MMPROJ=E:/models/mmproj_896.gguf

cmake --build build --config Release --target hlie-p0-hardware-smoke
```

`HLIE_P0_HARDWARE_PRIMARY_MMPROJ` and `HLIE_P0_HARDWARE_WORKER_MMPROJ` are
optional for text-only models; omit them when the corresponding runtime does
not use a multimodal projector.

The target starts an isolated master (port 17901 by default), performs local
and dynamic-Worker inference, injects pre-READY and post-READY Worker crashes,
checks resource rejection and ledger convergence, verifies generation recovery
and graceful shutdown, and audits INFO/WARN/ERROR logging. Artifacts are saved
under `artifacts/p0_hardware_smoke/<timestamp>`.

Useful cache variables are `HLIE_P0_HARDWARE_WORKER_MODEL` (defaults to the
primary GGUF), `HLIE_P0_HARDWARE_PRIMARY_DEVICE`,
`HLIE_P0_HARDWARE_WORKER_DEVICE`, and `HLIE_P0_HARDWARE_PORT`.

The explicit lifecycle soak defaults to 500 dynamic Worker cycles and runs one
proxy inference every 10 cycles:

```powershell
cmake --build build --config Release --target hlie-p0-hardware-soak
```

Use `HLIE_P0_HARDWARE_SOAK_CYCLES` and
`HLIE_P0_HARDWARE_SOAK_INFERENCE_EVERY` for shorter developer runs. The soak
checks the ledger after every cycle, requires one-time launch manifests to be
removed, samples parent RSS/private bytes/handles/threads, and rejects both
large total growth and continued growth in the final 20% of the run.

`HLIE_BUILD_TESTS=ON` remains accepted as a compatibility alias for
`BUILD_TESTING=ON`.
