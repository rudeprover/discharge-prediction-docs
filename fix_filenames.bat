@echo off
echo ========================================
echo Fixing MkDocs File Names
echo ========================================
echo.

cd /d "%~dp0"

REM Fix model file names
echo Fixing model file names...
if exist "docs\models\ann.md" (
    ren "docs\models\ann.md" "artificial-neural-network.md"
    echo Renamed ann.md to artificial-neural-network.md
)

if exist "docs\models\mlr.md" (
    ren "docs\models\mlr.md" "multiple-linear-regression.md"
    echo Renamed mlr.md to multiple-linear-regression.md
)

if exist "docs\models\slr.md" (
    ren "docs\models\slr.md" "simple-linear-regression.md"
    echo Renamed slr.md to simple-linear-regression.md
)

REM Fix resources file name
echo.
echo Fixing resources file name...
if exist "docs\resources\resources-libraries.md" (
    ren "docs\resources\resources-libraries.md" "libraries.md"
    echo Renamed resources-libraries.md to libraries.md
)

REM Fix fundamentals file names
echo.
echo Fixing fundamentals file names...
if exist "docs\fundamentals\feature-engineering.md" (
    echo feature-engineering.md already correct
)

if exist "docs\fundamentals\performance-metrics.md" (
    echo performance-metrics.md already correct
)

REM Fix setup file names
echo.
echo Fixing setup file names...
if exist "docs\setup\data-import.md" (
    echo data-import.md already correct
)

if exist "docs\setup\installation.md" (
    echo installation.md already correct
)

REM Check if index.md exists in root (might be in wrong place)
echo.
echo Checking index.md location...
if exist "index.md" (
    if not exist "docs\index.md" (
        move "index.md" "docs\index.md"
        echo Moved index.md to docs folder
    )
)

REM Verify all required files exist
echo.
echo ========================================
echo Verifying file structure...
echo ========================================

set missing=0

if exist "docs\index.md" (
    echo [OK] docs\index.md
) else (
    echo [MISSING] docs\index.md - Creating blank file...
    echo # Discharge Prediction with Python > "docs\index.md"
    echo. >> "docs\index.md"
    echo Welcome to the documentation! >> "docs\index.md"
    set missing=1
)

if exist "docs\models\simple-linear-regression.md" (
    echo [OK] docs\models\simple-linear-regression.md
) else (
    echo [MISSING] docs\models\simple-linear-regression.md
    set missing=1
)

if exist "docs\models\multiple-linear-regression.md" (
    echo [OK] docs\models\multiple-linear-regression.md
) else (
    echo [MISSING] docs\models\multiple-linear-regression.md
    set missing=1
)

if exist "docs\models\artificial-neural-network.md" (
    echo [OK] docs\models\artificial-neural-network.md
) else (
    echo [MISSING] docs\models\artificial-neural-network.md
    set missing=1
)

if exist "docs\resources\libraries.md" (
    echo [OK] docs\resources\libraries.md
) else (
    echo [MISSING] docs\resources\libraries.md
    set missing=1
)

if exist "docs\fundamentals\performance-metrics.md" (
    echo [OK] docs\fundamentals\performance-metrics.md
) else (
    echo [MISSING] docs\fundamentals\performance-metrics.md
    set missing=1
)

if exist "docs\fundamentals\feature-engineering.md" (
    echo [OK] docs\fundamentals\feature-engineering.md
) else (
    echo [MISSING] docs\fundamentals\feature-engineering.md
    set missing=1
)

if exist "docs\setup\installation.md" (
    echo [OK] docs\setup\installation.md
) else (
    echo [MISSING] docs\setup\installation.md
    set missing=1
)

if exist "docs\setup\data-import.md" (
    echo [OK] docs\setup\data-import.md
) else (
    echo [MISSING] docs\setup\data-import.md
    set missing=1
)

echo.
if %missing%==0 (
    echo ========================================
    echo SUCCESS: All files are correctly named!
    echo ========================================
    echo.
    echo You can now run: mkdocs serve
) else (
    echo ========================================
    echo WARNING: Some files are still missing
    echo ========================================
    echo Please check the files marked as [MISSING] above
)

echo.
pause