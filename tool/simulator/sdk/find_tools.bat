@echo off
rem ===================================================================================
rem find_tools.bat - locate Visual Studio (MSBuild, CMake) and OpenCV
rem   sets : VSDIR, MSBUILD, CMAKE, OCV (OpenCV ...\build folder)
rem   OpenCV : env OPENCV_DIR (...\build or ...\build\x64\vc15) -> else C:\opencv\build
rem ===================================================================================
set "VSDIR="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [오류] vswhere.exe 가 없음 - Visual Studio 2019 이상 + "C++ 데스크톱 개발" 설치 필요
  exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo [오류] C++ 도구가 설치된 Visual Studio 를 찾지 못함
  exit /b 1
)
set "MSBUILD=%VSDIR%\MSBuild\Current\Bin\amd64\MSBuild.exe"
if not exist "%MSBUILD%" set "MSBUILD=%VSDIR%\MSBuild\Current\Bin\MSBuild.exe"
set "CMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE%" (
  set "CMAKE="
  for /f "delims=" %%c in ('where cmake 2^>nul') do if not defined CMAKE set "CMAKE=%%c"
)

if defined OPENCV_DIR (set "OCV=%OPENCV_DIR%") else (set "OCV=C:\opencv\build")
if exist "%OCV%\..\..\include\opencv2" for %%d in ("%OCV%\..\..") do set "OCV=%%~fd"
if not exist "%OCV%\x64\vc15\bin\opencv_world3415.dll" (
  echo [오류] OpenCV 3.4.15 를 찾지 못함 : %OCV%
  echo        환경 변수 OPENCV_DIR 에 OpenCV 의 build 폴더를 넣어 주세요 (예 C:\opencv\build^)
  exit /b 1
)
echo Visual Studio : %VSDIR%
echo OpenCV        : %OCV%
exit /b 0
