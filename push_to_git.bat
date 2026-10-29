@echo off
setlocal
echo ============================================================
echo   ASTROVET 7-Day Continuous Milestone Push Pipeline
echo ============================================================

echo.
echo [1/7] Day 1 (Oct 22): Project layout and design decisions [b042aee]...
git push origin b042aee:refs/heads/main --force
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [2/7] Day 2 (Oct 23): Pipeline configs and runtime settings [e8d156a]...
git push origin e8d156a:refs/heads/main
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [3/7] Day 3 (Oct 24): Validation schemas and data dictionaries [c339eee]...
git push origin c339eee:refs/heads/main
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [4/7] Day 4 (Oct 25): MAST ingest and catalog resolver [74283ee]...
git push origin 74283ee:refs/heads/main
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [5/7] Day 5 (Oct 26): Light curve detrending and transit physics [96e002e]...
git push origin 96e002e:refs/heads/main
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [6/7] Day 6 (Oct 27): SSL representations and multimodal fusion [1db0a96]...
git push origin 1db0a96:refs/heads/main
if errorlevel 1 goto error
timeout /t 2 /nobreak >nul

echo.
echo [7/7] Day 7 (Oct 28): Web UI dashboard, CLI suite, docs and tests [545244d]...
git push origin 545244d:refs/heads/main
if errorlevel 1 goto error

git branch -u origin/main main
echo.
echo ============================================================
echo  Success! All 7 days pushed continuously to GitHub origin/main!
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Git push failed. Please verify GitHub authentication.
pause
exit /b 1
