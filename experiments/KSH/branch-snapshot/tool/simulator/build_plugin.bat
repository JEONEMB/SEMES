@echo off
chcp 65001 >nul
rem ===================================================================================
rem build_plugin.bat - build team code in sdk\team\<name>\ into plugins\<name>\<name>.dll
rem   then pick it in the simulator's estimator box (console : --estimator <name>)
rem   needs : Visual Studio 2019+ (Desktop C++, CMake tools), OpenCV 3.4.15
rem   (comments are ASCII on purpose: cmd misreads lines with multi-byte UTF-8 characters)
rem ===================================================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
call "%ROOT%\sdk\find_tools.bat" || goto :fail
if not defined CMAKE (
  echo [오류] CMake 를 찾지 못함 - Visual Studio 설치 관리자에서 "Windows용 C++ CMake 도구" 추가
  goto :fail
)
echo.
echo [1/2] 구성 : "%ROOT%\sdk"
"%CMAKE%" -S "%ROOT%\sdk" -B "%ROOT%\sdk\build" -A x64 -DSIM_OPENCV_ROOT="%OCV%" -DSIM_PLUGIN_OUT_DIR="%ROOT%\plugins" || goto :fail
echo [2/2] 빌드 (Release)
"%CMAKE%" --build "%ROOT%\sdk\build" --config Release || goto :fail
echo.
echo 완료 - 만든 플러그인 :
dir /s /b "%ROOT%\plugins\*.dll"
if "%~1"=="" pause
exit /b 0
:fail
echo.
echo [실패] 위 오류를 확인하세요.
if "%~1"=="" pause
exit /b 1
