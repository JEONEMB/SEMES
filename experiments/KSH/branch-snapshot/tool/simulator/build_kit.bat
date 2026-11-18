@echo off
chcp 65001 >nul
rem ===================================================================================
rem build_kit.bat - build the simulator distribution kit (Release x64)
rem   1) MSBuild : console\Simulator.vcxproj, mfc\SimulatorMFC.vcxproj -> build\bin
rem   2) CMake   : plugins (sdk\team\*, example ksh_plugin) -> plugins  (skipped without CMake)
rem   3) dist\SimulatorKit : exe, DLLs (OpenCV, VC++ runtime), config, world, SDK, docs + zip
rem   The previous kit is not deleted: it is moved to dist\_backup\
rem   usage : build_kit.bat           (pauses at the end)
rem           build_kit.bat nopause
rem   (comments are ASCII on purpose: cmd misreads lines with multi-byte UTF-8 characters)
rem ===================================================================================
setlocal EnableExtensions EnableDelayedExpansion
set "SIM=%~dp0"
if "%SIM:~-1%"=="\" set "SIM=%SIM:~0,-1%"
set "OUT=%SIM%\build\bin"
set "KIT=%SIM%\dist\SimulatorKit"
rem alignment sources : repo root (PJT\include, PJT\src\modules) first, else old ..\Alignment
set "ALIGN=%SIM%\..\Alignment"
if exist "%SIM%\..\..\src\modules" for %%a in ("%SIM%\..\..") do set "ALIGN=%%~fa"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%t"

call "%SIM%\sdk\find_tools.bat" || goto :fail

echo.
echo [1/3] exe 빌드 (Release x64)
for %%p in (console\Simulator.vcxproj mfc\SimulatorMFC.vcxproj) do (
  "%MSBUILD%" "%SIM%\%%p" /m /nologo /v:minimal /p:Configuration=Release /p:Platform=x64 /p:OpenCVDir="%OCV%" /p:OutDir="%OUT%\\" /p:IntDir="%SIM%\build\obj\%%~np\\"
  if errorlevel 1 ( echo [오류] 빌드 실패 : %%p & goto :fail )
)

echo.
echo [2/3] 플러그인 빌드
if defined CMAKE (
  "%CMAKE%" -S "%SIM%\sdk" -B "%SIM%\build\plugins" -A x64 -DSIM_OPENCV_ROOT="%OCV%" -DSIM_PLUGIN_OUT_DIR="%SIM%\plugins" -DSIM_ALIGNMENT_DIR="%ALIGN%"
  if errorlevel 1 ( echo [오류] 플러그인 구성 실패 & goto :fail )
  "%CMAKE%" --build "%SIM%\build\plugins" --config Release
  if errorlevel 1 ( echo [오류] 플러그인 빌드 실패 & goto :fail )
) else (
  echo CMake 가 없어 플러그인은 건너뜀 - 내장 추정기만 들어감
)

echo.
echo [3/3] 키트 모으기 : %KIT%
if exist "%KIT%" (
  if not exist "%SIM%\dist\_backup" mkdir "%SIM%\dist\_backup"
  move "%KIT%" "%SIM%\dist\_backup\SimulatorKit_%TS%" >nul || ( echo [오류] 이전 키트를 옮기지 못함 - 열려 있는 파일을 닫아 주세요 & goto :fail )
)
for %%d in ("" config world plugins sdk docs results data data\img) do mkdir "%KIT%\%%~d" 2>nul
copy /y "%OUT%\Simulator.exe" "%KIT%\" >nul || goto :fail
copy /y "%OUT%\SimulatorMFC.exe" "%KIT%\" >nul || goto :fail
copy /y "%OCV%\x64\vc15\bin\opencv_world3415.dll" "%KIT%\" >nul || goto :fail
rem VC++ runtime DLLs next to the exe (app-local) so PCs without Visual Studio can run it
set "REDIST="
for /d %%v in ("%VSDIR%\VC\Redist\MSVC\14.*") do set "REDIST=%%v"
if defined REDIST (
  for /d %%c in ("!REDIST!\x64\Microsoft.VC*.CRT") do copy /y "%%c\*.dll" "%KIT%\" >nul
  for /d %%c in ("!REDIST!\x64\Microsoft.VC*.MFC") do copy /y "%%c\mfc140u.dll" "%KIT%\" >nul
) else (
  echo [주의] VC++ 재배포 파일을 찾지 못함 - 받는 PC 에 "Visual C++ 재배포 가능 패키지(x64)" 가 필요할 수 있음
)
copy /y "%SIM%\config\simulator.yml" "%KIT%\config\" >nul
copy /y "%ALIGN%\config\*.yml" "%KIT%\config\" >nul
robocopy "%SIM%\world" "%KIT%\world" /E /XD _backup /NFL /NDL /NJH /NJS /NP >nul
if exist "%SIM%\plugins" robocopy "%SIM%\plugins" "%KIT%\plugins" /E /NFL /NDL /NJH /NJS /NP >nul
robocopy "%SIM%\sdk" "%KIT%\sdk" /E /XD build /NFL /NDL /NJH /NJS /NP >nul
copy /y "%SIM%\docs\*.md" "%KIT%\docs\" >nul
copy /y "%SIM%\kit\README.md" "%KIT%\README.md" >nul
copy /y "%SIM%\kit\data_README.md" "%KIT%\data\img\README.md" >nul
copy /y "%SIM%\kit\plugins_README.md" "%KIT%\plugins\README.md" >nul
copy /y "%SIM%\build_plugin.bat" "%KIT%\" >nul

echo.
echo 자가 점검 : 키트에서 콘솔 앱 실행
pushd "%KIT%"
Simulator.exe --list-plugins
if errorlevel 1 ( popd & echo [오류] 키트의 Simulator.exe 실행 실패 & goto :fail )
popd

powershell -NoProfile -Command "Compress-Archive -Path '%KIT%\*' -DestinationPath '%SIM%\dist\SimulatorKit_%TS%.zip'"
echo.
echo 완료
echo   키트 : %KIT%
echo   압축 : %SIM%\dist\SimulatorKit_%TS%.zip  (팀원에게 전달)
if /i not "%~1"=="nopause" pause
exit /b 0

:fail
echo.
echo [실패] 위 오류를 확인하세요.
if /i not "%~1"=="nopause" pause
exit /b 1
