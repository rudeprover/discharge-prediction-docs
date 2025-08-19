@echo off
echo ========================================
echo Fixing MkDocs Documentation Issues
echo ========================================
echo.

REM Step 1: Fix file extensions
echo Step 1: Fixing file extensions...
echo --------------------------------

REM Root directory files
if exist mkdocs-config.txt (
    ren mkdocs-config.txt mkdocs.yml
    echo Renamed mkdocs-config.txt to mkdocs.yml
)
if exist mkdocs.txt (
    ren mkdocs.txt mkdocs.yml
    echo Renamed mkdocs.txt to mkdocs.yml
)
if exist requirements-txt.txt (
    ren requirements-txt.txt requirements.txt
    echo Renamed requirements-txt.txt to requirements.txt
)
if exist requirements.txt.txt (
    ren requirements.txt.txt requirements.txt
    echo Renamed requirements.txt.txt to requirements.txt
)
if exist deploy-script.txt (
    ren deploy-script.txt deploy.sh
    echo Renamed deploy-script.txt to deploy.sh
)
if exist README.txt (
    ren README.txt README.md
    echo Renamed README.txt to README.md
)
if exist gitignore.txt (
    ren gitignore.txt .gitignore
    echo Renamed gitignore.txt to .gitignore
)

REM GitHub Actions
if exist .github\workflows\ci.yml.txt (
    ren .github\workflows\ci.yml.txt ci.yml
    echo Renamed ci.yml.txt to ci.yml
)
if exist .github\workflows\ci.txt (
    ren .github\workflows\ci.txt ci.yml
    echo Renamed ci.txt to ci.yml
)

REM Docs files
if exist docs\index.txt (
    ren docs\index.txt index.md
    echo Renamed index.txt to index.md
)

REM Setup files
if exist docs\setup\installation.txt (
    ren docs\setup\installation.txt installation.md
)
if exist docs\setup\data-import.txt (
    ren docs\setup\data-import.txt data-import.md
)

REM Fundamentals files
if exist docs\fundamentals\performance-metrics.txt (
    ren docs\fundamentals\performance-metrics.txt performance-metrics.md
)
if exist docs\fundamentals\feature-engineering.txt (
    ren docs\fundamentals\feature-engineering.txt feature-engineering.md
)

REM Models files
if exist docs\models\simple-linear-regression.txt (
    ren docs\models\simple-linear-regression.txt simple-linear-regression.md
)
if exist docs\models\multiple-linear-regression.txt (
    ren docs\models\multiple-linear-regression.txt multiple-linear-regression.md
)
if exist docs\models\artificial-neural-network.txt (
    ren docs\models\artificial-neural-network.txt artificial-neural-network.md
)

REM Resources files
if exist docs\resources\libraries.txt (
    ren docs\resources\libraries.txt libraries.md
)

REM Assets files
if exist docs\assets\css\extra.css.txt (
    ren docs\assets\css\extra.css.txt extra.css
)
if exist docs\assets\js\extra.js.txt (
    ren docs\assets\js\extra.js.txt extra.js
)

echo.
echo Step 2: Fixing broken links in markdown files...
echo ------------------------------------------------

REM Create a PowerShell script to fix the links
echo Creating PowerShell script to fix links...
(
echo # Fix broken links in markdown files
echo.
echo # Fix links in models/simple-linear-regression.md
echo $file = 'docs\models\simple-linear-regression.md'
echo if ^(Test-Path $file^) {
echo     $content = Get-Content $file -Raw
echo     $content = $content -replace '\.\./fundamentals/performance-metrics\.md', '../fundamentals/performance-metrics.md'
echo     $content = $content -replace '\.\./fundamentals/feature-engineering\.md', '../fundamentals/feature-engineering.md'
echo     $content = $content -replace 'multiple-linear-regression\.md', 'multiple-linear-regression.md'
echo     $content = $content -replace 'artificial-neural-network\.md', 'artificial-neural-network.md'
echo     Set-Content $file $content -NoNewline
echo     Write-Host "Fixed links in $file"
echo }
echo.
echo # Fix links in models/multiple-linear-regression.md
echo $file = 'docs\models\multiple-linear-regression.md'
echo if ^(Test-Path $file^) {
echo     $content = Get-Content $file -Raw
echo     $content = $content -replace 'simple-linear-regression\.md', 'simple-linear-regression.md'
echo     $content = $content -replace 'artificial-neural-network\.md', 'artificial-neural-network.md'
echo     $content = $content -replace '\.\./resources/libraries\.md', '../resources/libraries.md'
echo     Set-Content $file $content -NoNewline
echo     Write-Host "Fixed links in $file"
echo }
echo.
echo # Fix links in models/artificial-neural-network.md
echo $file = 'docs\models\artificial-neural-network.md'
echo if ^(Test-Path $file^) {
echo     $content = Get-Content $file -Raw
echo     $content = $content -replace 'multiple-linear-regression\.md', 'multiple-linear-regression.md'
echo     $content = $content -replace '\.\./resources/libraries\.md', '../resources/libraries.md'
echo     $content = $content -replace '\.\./models/simple-linear-regression\.md', 'simple-linear-regression.md'
echo     Set-Content $file $content -NoNewline
echo     Write-Host "Fixed links in $file"
echo }
echo.
echo # Fix links in resources/resources-libraries.md if it exists
echo $oldfile = 'docs\resources\resources-libraries.md'
echo $newfile = 'docs\resources\libraries.md'
echo if ^(Test-Path $oldfile^) {
echo     if ^(Test-Path $newfile^) {
echo         Remove-Item $oldfile
echo     } else {
echo         Rename-Item $oldfile $newfile
echo     }
echo     Write-Host "Fixed resources file name"
echo }
echo.
echo # Fix links in setup/installation.md
echo $file = 'docs\setup\installation.md'
echo if ^(Test-Path $file^) {
echo     $content = Get-Content $file -Raw
echo     $content = $content -replace '\.\./index\.md', '../index.md'
echo     $content = $content -replace 'data-import\.md', 'data-import.md'
echo     $content = $content -replace '\.\./fundamentals/performance-metrics\.md', '../fundamentals/performance-metrics.md'
echo     Set-Content $file $content -NoNewline
echo     Write-Host "Fixed links in $file"
echo }
echo.
echo # Fix image paths
echo $files = Get-ChildItem -Path docs -Filter *.md -Recurse
echo foreach ^($file in $files^) {
echo     $content = Get-Content $file.FullName -Raw
echo     $content = $content -replace '\.\./assets/images/', '../assets/images/'
echo     $content = $content -replace 'assets/images/', '../assets/images/'
echo     Set-Content $file.FullName $content -NoNewline
echo }
echo Write-Host "Fixed image paths in all markdown files"
) > fix_links.ps1

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File fix_links.ps1

REM Clean up
del fix_links.ps1

echo.
echo Step 3: Verifying file structure...
echo ------------------------------------

if exist mkdocs.yml (
    echo [OK] mkdocs.yml exists
) else (
    echo [ERROR] mkdocs.yml not found!
)

if exist docs\index.md (
    echo [OK] docs\index.md exists
) else (
    echo [ERROR] docs\index.md not found!
)

if exist requirements.txt (
    echo [OK] requirements.txt exists
) else (
    echo [ERROR] requirements.txt not found!
)

echo.
echo ========================================
echo All fixes completed!
echo ========================================
echo.
echo You can now run: mkdocs serve
echo.
pause