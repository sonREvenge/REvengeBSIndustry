@echo off
setlocal enabledelayedexpansion
title Brawl Industry - Setup
cd /d "%~dp0"

echo Brawl Industry - Setup
echo.

echo [1/4] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable dans le PATH.
    echo Telecharge Python 3.10+ sur https://www.python.org/downloads/
    echo Coche "Add Python to PATH" pendant l'installation.
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if !PYMAJ! LSS 3 (
    echo [ERREUR] Python !PYVER! detecte - version 3.10+ requise.
    pause & exit /b 1
)
if !PYMAJ! EQU 3 if !PYMIN! LSS 10 (
    echo [ERREUR] Python !PYVER! detecte - version 3.10+ requise.
    pause & exit /b 1
)
echo [OK] Python !PYVER!

echo [2/4] Mise a jour de pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARN] Impossible de mettre a jour pip, on continue quand meme.
)
echo [OK] pip pret

echo [3/4] Installation des dependances (peut prendre 1-2 minutes)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installation des dependances a echoue.
    echo Essaie de relancer ce fichier en Administrateur.
    echo Si le probleme persiste, lance manuellement :
    echo   python -m pip install -r requirements.txt
    pause & exit /b 1
)
echo [OK] Dependances installees

echo [4/4] Verification de ADB...
adb version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARN] ADB introuvable dans le PATH.
    echo Telecharge Android Platform Tools :
    echo   https://developer.android.com/tools/releases/platform-tools
    echo Extrais le dossier et ajoute-le a la variable PATH.
    echo.
    echo Le bot ne pourra pas demarrer sans ADB.
    echo.
) else (
    for /f "tokens=*" %%v in ('adb version 2^>^&1 ^| findstr /i "version"') do echo [OK] %%v
)

echo.
echo Configuration du bot
python main.py --setup

echo.
echo Setup termine. Lance start.bat pour demarrer le bot.
pause
