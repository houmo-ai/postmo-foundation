# C++ tests

The C++ test suite uses the vendored Catch2 framework and runs through CTest.
The fast path deliberately avoids configuring proprietary inference libraries:

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
```

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

`HLIE_BUILD_TESTS=ON` remains accepted as a compatibility alias for
`BUILD_TESTING=ON`.
