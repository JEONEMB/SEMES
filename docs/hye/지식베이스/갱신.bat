@echo off
chcp 65001 > nul
setlocal
REM 지식베이스 갱신: 사실/*.md 를 읽어 지식지도.md 와 Obsidian 캔버스를 다시 쓴다.
REM 사용법:  갱신.bat                      (기본 캔버스로)
REM          갱신.bat "D:\다른\경로.canvas"  (다른 캔버스로)

set "CM=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "SRC=%~dp0생성기"
set "BLD=%TEMP%\hyebuild\kbgen"

if not exist "%BLD%\Release\kbgen.exe" (
  echo [1/2] 생성기 빌드
  "%CM%" -S "%SRC%" -B "%BLD%" -G "Visual Studio 18 2026" -A x64 -T v141 > nul
  "%CM%" --build "%BLD%" --config Release > nul
) else (
  echo [1/2] 생성기 다시 빌드
  "%CM%" --build "%BLD%" --config Release > nul
)
if errorlevel 1 ( echo 빌드 실패 & exit /b 1 )

echo [2/2] 지도 갱신
"%BLD%\Release\kbgen.exe" %1
endlocal
