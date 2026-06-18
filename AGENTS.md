# hi.cpp Agent Notes

## Build

- Preferred CMake executable on this machine:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe`
- Common Release build command:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe --build D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio --target houmo-llm-hi-cpp-win --config Release`
- Reconfigure the existing build directory when CMake files or new globbed `.cpp` files change:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe -S D:\workspace\wtt\houmo\llm\hi.cpp -B D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio`
- The generated Visual Studio build directory is:
  `D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio`

## Notes

- `src/hlie_base/CMakeLists.txt` uses `file(GLOB ...)` for base sources. After adding new `.cpp` files, rerun CMake configure if the generated project does not pick them up automatically.
