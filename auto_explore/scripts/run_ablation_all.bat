@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
call ".\run_ablation.bat" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
