@echo off
rem ============================================================================
rem  CODEEP - instalador/lancador para Windows 10/11 (duplo clique)
rem
rem  Chama o Install-CODEEP.ps1 (PowerShell) sem exigir que o usuario abra um
rem  terminal ou digite comandos: copia o programa para %USERPROFILE%\CODEEP,
rem  instala o Python 3.10 se faltar (so para o usuario atual, sem administrador),
rem  cria o ambiente virtual e os atalhos CODEEP na Area de Trabalho e no Menu
rem  Iniciar e abre o programa.
rem
rem  So e preciso na primeira vez: depois use o atalho CODEEP.
rem  (Se o Windows mostrar "O Windows protegeu o computador": Mais informacoes >
rem  Executar assim mesmo.)
rem ============================================================================
setlocal EnableExtensions
title CODEEP - Instalacao
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-CODEEP.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [CODEEP] A instalacao nao terminou ^(codigo %RC%^). Leia as mensagens acima.
    pause
)
exit /b %RC%
