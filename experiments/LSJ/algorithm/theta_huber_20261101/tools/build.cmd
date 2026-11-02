@echo off
setlocal
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "CTEST=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
set "SOURCE=%~dp0.."
set "BUILD=%~dp0..\..\..\.tmp\theta_huber_20261101_build"
"%CMAKE%" -S "%SOURCE%" -B "%BUILD%" -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/Users/SSAFY/opencv/build
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%BUILD%" --config Release -- /m:1
if errorlevel 1 exit /b 1
"%CTEST%" --test-dir "%BUILD%" -C Release --output-on-failure
if errorlevel 1 exit /b 1
exit /b 0
