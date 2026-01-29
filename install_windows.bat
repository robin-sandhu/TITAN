@echo off
REM ==============================================================================
REM TITAN - Windows Installer Script (Developed by Robin Sandhu)
REM A Standardized Framework for Clinical Prediction Model Development
REM ==============================================================================
REM
REM This script sets up TITAN and all dependencies on Windows.
REM
REM Usage:
REM   Right-click and "Run as Administrator" (recommended)
REM   Or double-click to run
REM
REM ==============================================================================

echo ==============================================================
echo   TITAN - Windows Installation
echo   Developed by Robin Sandhu
echo ==============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM ==============================================================================
REM Step 1: Create Virtual Environment
REM ==============================================================================

echo.
echo Step 1: Creating virtual environment...

if exist titan_venv (
    echo [INFO] Virtual environment already exists
    set /p RECREATE="Do you want to recreate it? (y/n): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q titan_venv
        python -m venv titan_venv
        echo [OK] Virtual environment recreated
    ) else (
        echo [OK] Using existing virtual environment
    )
) else (
    python -m venv titan_venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
call titan_venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM ==============================================================================
REM Step 2: Upgrade pip
REM ==============================================================================

echo.
echo Step 2: Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded

REM ==============================================================================
REM Step 3: Install Core Dependencies
REM ==============================================================================

echo.
echo Step 3: Installing core dependencies (this may take several minutes)...

echo Installing numpy, pandas, scipy...
pip install numpy pandas scipy >nul 2>&1
echo [OK] Core math packages installed

echo Installing scikit-learn...
pip install scikit-learn >nul 2>&1
echo [OK] scikit-learn installed

echo Installing matplotlib, seaborn...
pip install matplotlib seaborn >nul 2>&1
echo [OK] Visualization packages installed

echo Installing shap...
pip install shap >nul 2>&1
echo [OK] SHAP installed

echo Installing statsmodels, pingouin, lifelines...
pip install statsmodels pingouin lifelines >nul 2>&1
echo [OK] Statistics packages installed

echo Installing fpdf2, openpyxl, networkx, joblib...
pip install fpdf2 openpyxl networkx joblib >nul 2>&1
echo [OK] Utility packages installed

REM ==============================================================================
REM Step 4: Install Medical NER (Optional)
REM ==============================================================================

echo.
set /p INSTALL_NER="Step 4: Install medical NER support (UMLS)? This is optional but recommended. (y/n): "
if /i "%INSTALL_NER%"=="y" (
    echo Installing spacy and scispacy (this may take a while)...
    pip install spacy >nul 2>&1
    pip install scispacy >nul 2>&1
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz >nul 2>&1
    echo [OK] Medical NER installed
) else (
    echo [INFO] Skipping medical NER (TITAN will work without it)
)

REM ==============================================================================
REM Step 5: Verify Installation
REM ==============================================================================

echo.
echo Step 5: Verifying installation...

python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('[OK] Core packages verified')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Core package verification failed
    pause
    exit /b 1
)

REM Check SHAP separately (may have warnings)
python -c "import shap; print('[OK] SHAP verified')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] SHAP verification had warnings (may still work)
)

REM Check if TITAN.py exists
if exist TITAN.py (
    python -c "from TITAN import run_infinity_on_file; print('[OK] TITAN.py importable')" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] TITAN import check had warnings (may still work)
    )
) else (
    echo [WARNING] TITAN.py not found in current directory
)

REM ==============================================================================
REM Step 6: Create Launcher Scripts
REM ==============================================================================

echo.
echo Step 6: Creating launcher scripts...

REM CLI launcher
(
echo @echo off
echo cd /d "%%~dp0"
echo call titan_venv\Scripts\activate.bat
echo python TITAN.py %%*
echo pause
) > run_titan.bat
echo [OK] CLI launcher created: run_titan.bat

REM GUI launcher
(
echo @echo off
echo cd /d "%%~dp0"
echo call titan_venv\Scripts\activate.bat
echo python TITAN_GUI.py %%*
) > run_titan_gui.bat
echo [OK] GUI launcher created: run_titan_gui.bat

REM GUI launcher (no console)
(
echo @echo off
echo cd /d "%%~dp0"
echo call titan_venv\Scripts\activate.bat
echo start "" pythonw TITAN_GUI.py
) > run_titan_gui_noconsole.bat
echo [OK] GUI launcher (no console) created: run_titan_gui_noconsole.bat

REM ==============================================================================
REM Step 7: Create Desktop Shortcut (Optional)
REM ==============================================================================

echo.
set /p CREATE_SHORTCUT="Create desktop shortcut? (y/n): "
if /i "%CREATE_SHORTCUT%"=="y" (
    set "SCRIPT_DIR=%CD%"
    set "DESKTOP=%USERPROFILE%\Desktop"
    
    REM Create VBS script to make shortcut
    echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
    echo sLinkFile = "%DESKTOP%\TITAN.lnk" >> CreateShortcut.vbs
    echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
    echo oLink.TargetPath = "%SCRIPT_DIR%\run_titan_gui_noconsole.bat" >> CreateShortcut.vbs
    echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> CreateShortcut.vbs
    echo oLink.Description = "TITAN - Developed by Robin Sandhu" >> CreateShortcut.vbs
    echo oLink.Save >> CreateShortcut.vbs
    
    cscript //nologo CreateShortcut.vbs
    del CreateShortcut.vbs
    
    echo [OK] Desktop shortcut created
)

REM ==============================================================================
REM Complete
REM ==============================================================================

echo.
echo ==============================================================
echo   TITAN Installation Complete!
echo ==============================================================
echo.
echo To run TITAN:
echo   Command Line:  Double-click run_titan.bat
echo   GUI:           Double-click run_titan_gui.bat
echo.
echo Or activate the environment manually:
echo   titan_venv\Scripts\activate.bat
echo   python TITAN.py
echo.
echo For help, see USER_MANUAL.md
echo ==============================================================
echo.
pause
