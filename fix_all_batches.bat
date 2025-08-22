@echo off
echo ========================================
echo FINAL FIX - Renaming Files
echo ========================================
echo.

cd /d "%~dp0"

echo Renaming model files...
echo ----------------------

if exist "docs\models\ann.md" (
    ren "docs\models\ann.md" "artificial-neural-network.md"
    echo [DONE] Renamed ann.md to artificial-neural-network.md
) else (
    echo [SKIP] ann.md not found or already renamed
)

if exist "docs\models\mlr.md" (
    ren "docs\models\mlr.md" "multiple-linear-regression.md"
    echo [DONE] Renamed mlr.md to multiple-linear-regression.md
) else (
    echo [SKIP] mlr.md not found or already renamed
)

if exist "docs\models\slr.md" (
    ren "docs\models\slr.md" "simple-linear-regression.md"
    echo [DONE] Renamed slr.md to simple-linear-regression.md
) else (
    echo [SKIP] slr.md not found or already renamed
)

echo.
echo Renaming resources file...
echo -------------------------

if exist "docs\resources\resources-libraries.md" (
    ren "docs\resources\resources-libraries.md" "libraries.md"
    echo [DONE] Renamed resources-libraries.md to libraries.md
) else (
    echo [SKIP] resources-libraries.md not found or already renamed
)

echo.
echo Fixing image paths in all markdown files...
echo ------------------------------------------

REM Create PowerShell script to fix image paths
(
echo $files = Get-ChildItem -Path docs -Filter *.md -Recurse
echo foreach ^($file in $files^) {
echo     $content = Get-Content $file.FullName -Raw
echo     # Fix excessive ../../../ in image paths
echo     $content = $content -replace '\.\./\.\./\.\./\.\./\.\./assets/images/', '../assets/images/'
echo     $content = $content -replace '\.\./\.\./\.\./\.\./assets/images/', '../assets/images/'
echo     $content = $content -replace '\.\./\.\./\.\./assets/images/', '../assets/images/'
echo     $content = $content -replace '\.\./\.\./assets/images/', '../assets/images/'
echo     # Fix for docs root
echo     if ^($file.Directory.Name -eq 'docs'^) {
echo         $content = $content -replace '\.\./\.\./assets/images/', 'assets/images/'
echo     }
echo     Set-Content $file.FullName $content -NoNewline
echo }
echo Write-Host "Fixed image paths"
) > fix_paths.ps1

powershell -ExecutionPolicy Bypass -File fix_paths.ps1
del fix_paths.ps1

echo.
echo ========================================
echo VERIFICATION
echo ========================================
echo.

echo Checking if files exist with correct names:
echo -------------------------------------------

set all_good=1

if exist "docs\models\artificial-neural-network.md" (
    echo [OK] artificial-neural-network.md
) else (
    echo [MISSING] artificial-neural-network.md
    set all_good=0
)

if exist "docs\models\multiple-linear-regression.md" (
    echo [OK] multiple-linear-regression.md
) else (
    echo [MISSING] multiple-linear-regression.md
    set all_good=0
)

if exist "docs\models\simple-linear-regression.md" (
    echo [OK] simple-linear-regression.md
) else (
    echo [MISSING] simple-linear-regression.md
    set all_good=0
)

if exist "docs\resources\libraries.md" (
    echo [OK] libraries.md
) else (
    echo [MISSING] libraries.md
    set all_good=0
)

echo.
if %all_good%==1 (
    echo ========================================
    echo SUCCESS! All files renamed correctly!
    echo ========================================
    echo.
    echo Now run: mkdocs serve
    echo Then open: http://localhost:8000
) else (
    echo ========================================
    echo ERROR: Some files are still missing
    echo ========================================
)

echo.
pause