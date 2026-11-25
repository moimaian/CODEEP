# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# CODEEP – Deep Learning Pipeline for Ligand Generation
# Copyright (C) 2024–2026 Moisés Maia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Preparação do ambiente do CODEEP no Linux e no Windows — SÓ biblioteca padrão (sem Qt).

Existe separado de CODEEP_requirements.py porque esse módulo importa o PyQt5 já no topo, e o PyQt5 é
justamente o que o bootstrap precisa instalar antes de tudo. Aqui ficam:

  • o venv dedicado (~/.venv/CODEEP) e o Python que o cria (no Windows, o 3.10.11 exato do python.org,
    instalado só para o usuário atual, sem administrador);
  • o runtime Visual C++ de que as DLLs do Qt5 precisam no Windows;
  • os atalhos CODEEP.lnk (Área de Trabalho e Menu Iniciar) que abrem o programa com pythonw.exe, sem
    console — "python CODEEP.py --setup" os cria (é o que Install-CODEEP.ps1 chama);
  • os programas externos fora do pip — Java 17+, P2Rank e xtb — escolhendo o arquivo certo para o sistema
    em que roda: tar.gz/tar.xz no Linux, winget e zip no Windows.

No Linux o comportamento é o de sempre; tudo o que é do Windows fica atrás de `IS_WINDOWS`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from MODULES import platform_utils
from MODULES.platform_utils import IS_WINDOWS

APP_NAME = "CODEEP"
APP_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = APP_DIR / "tools"

# ── Python do venv ────────────────────────────────────────────────────────────
TARGET_VENV_PYTHON_WINDOWS = "3.10.11"
PYTHON_WINDOWS_INSTALLER_URL = (
    f"https://www.python.org/ftp/python/{TARGET_VENV_PYTHON_WINDOWS}/python-{TARGET_VENV_PYTHON_WINDOWS}-amd64.exe"
)
PYTHON_WINGET_ID = "Python.Python.3.10"
VCREDIST_WINGET_ID = "Microsoft.VCRedist.2015+.x64"
VCREDIST_DOWNLOAD_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
JAVA_WINGET_ID = "Microsoft.OpenJDK.17"
# códigos de saída do winget que significam "já instalado" / "nada a atualizar"
_WINGET_ALREADY_OK = {0, 0x8A150061 - (1 << 32), 0x8A15010D - (1 << 32)}

# Pacotes mínimos para abrir a janela (o resto é instalado pela janela Install Requirements)
ESSENTIAL_PACKAGES = ["PyQt5==5.15.11", "psutil"]

# ── Programas externos ────────────────────────────────────────────────────────
P2RANK_VERSION = "2.5.1"
P2RANK_URL_TMPL = "https://github.com/rdk/p2rank/releases/download/{v}/p2rank_{v}.tar.gz"
JAVA_MIN_MAJOR = 17
JAVA_INSTALL_HINT_LINUX = "sudo apt install openjdk-17-jre-headless"
JAVA_INSTALL_HINT_WINDOWS = f"winget install --id {JAVA_WINGET_ID} -e"
XTB_VERSION = "6.7.1"
XTB_URL_LINUX = "https://github.com/grimme-lab/xtb/releases/download/v{v}/xtb-{v}-linux-x86_64.tar.xz"
# o release v6.7.1 publica o build do Windows com o sufixo "pre"; o zip abre em xtb-6.7.1/bin/xtb.exe
XTB_URL_WINDOWS = "https://github.com/grimme-lab/xtb/releases/download/v{v}/xtb-{v}pre-windows-x86_64.zip"


def java_install_hint() -> str:
    return JAVA_INSTALL_HINT_WINDOWS if IS_WINDOWS else JAVA_INSTALL_HINT_LINUX


# ==========================================================================
# ─── HELPERS ──────────────────────────────────────────────────────────────
# ==========================================================================

def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def assume_yes() -> bool:
    """CODEEP_ASSUME_YES=1 responde "S" às confirmações (Install-CODEEP.ps1 usa, para não precisar de terminal)."""
    return os.environ.get("CODEEP_ASSUME_YES", "").strip().lower() in ("1", "true", "yes", "s", "y")


def stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    if assume_yes():
        print(f"{message} [S/n]: S")
        return True
    if not stdin_is_tty():
        return default_yes
    try:
        resp = input(f"{message} [{'S/n' if default_yes else 's/N'}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not resp:
        return default_yes
    return resp in ("s", "sim", "y", "yes")


def download_file(url: str, destination: str | Path, log: Callable[[str], None] = print) -> bool:
    """Download só com a biblioteca padrão (o Windows não tem wget)."""
    import urllib.request
    try:
        log(f"$ download {url}")
        urllib.request.urlretrieve(url, str(destination))
        return Path(destination).is_file() and Path(destination).stat().st_size > 0
    except Exception as exc:
        log(f"FAILED  download: {url}: {exc}")
        return False


# ==========================================================================
# ─── WINDOWS: winget, Visual C++, Python 3.10 ─────────────────────────────
# ==========================================================================

def winget_available() -> bool:
    return IS_WINDOWS and _which("winget") is not None


def winget_install(pkg_id: str, extra: Optional[list] = None, log: Callable[[str], None] = print) -> bool:
    """winget install -e --id <pkg_id> ...; True se instalou ou se já estava instalado."""
    command = ["winget", "install", "--id", pkg_id, "-e", "--silent",
               "--accept-package-agreements", "--accept-source-agreements"] + list(extra or [])
    log("$ " + " ".join(command))
    res = _run(command)
    code = res.returncode
    if code > 0x7FFFFFFF:
        code -= 1 << 32
    return code in _WINGET_ALREADY_OK


def vcredist_installed() -> bool:
    """True se o runtime Visual C++ 2015-2022 (x64) está instalado (chave de registro do instalador)."""
    if not IS_WINDOWS:
        return True
    try:
        import winreg
    except ImportError:
        return True
    for key in (r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                if int(winreg.QueryValueEx(handle, "Installed")[0]) == 1:
                    return True
        except OSError:
            continue
    return False


def ensure_vcredist_windows(log: Callable[[str], None] = print) -> bool:
    """Garante o runtime Visual C++ (x64) que as DLLs do Qt5 e do PyTorch exigem: sem ele o import falha com
    "DLL load failed". Usa o winget e, sem ele, o instalador oficial da Microsoft; o Windows pede a
    confirmação de administrador (UAC) uma vez."""
    if not IS_WINDOWS or vcredist_installed():
        return True
    log(f"[{APP_NAME}] Visual C++ runtime (x64) not found - the Qt5/PyTorch libraries need it. Installing...")
    ok = winget_install(VCREDIST_WINGET_ID, log=log) if winget_available() else False
    if not ok and not vcredist_installed():
        installer = os.path.join(tempfile.gettempdir(), "codeep_vc_redist.x64.exe")
        if download_file(VCREDIST_DOWNLOAD_URL, installer, log):
            # 1638: já há uma versão mais nova; 3010: pede reinício do Windows
            ok = _run([installer, "/install", "/quiet", "/norestart"]).returncode in (0, 1638, 3010)
    if vcredist_installed() or ok:
        log(f"[{APP_NAME}] Visual C++ runtime available.")
        return True
    log(f"[{APP_NAME}] Could not install the Visual C++ runtime. Install it by hand:\n"
        f"         winget install --id {VCREDIST_WINGET_ID} -e\n         or {VCREDIST_DOWNLOAD_URL}")
    return False


def _python_version_string(pybin: str) -> Optional[str]:
    try:
        res = _run([pybin, "-c", "import platform; print(platform.python_version())"], timeout=30)
        return res.stdout.strip().splitlines()[-1] if res.returncode == 0 and res.stdout.strip() else None
    except Exception:
        return None


def _windows_python_candidates() -> List[str]:
    """Pythons 3.10 prováveis no Windows (py launcher, pastas padrão dos instaladores, PATH)."""
    found: List[str] = []
    if _which("py"):
        res = _run(["py", "-3.10", "-c", "import sys; print(sys.executable)"])
        if res.returncode == 0 and res.stdout.strip():
            found.append(res.stdout.strip().splitlines()[-1])
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        found.append(os.path.join(local, "Programs", "Python", "Python310", "python.exe"))
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(variable, "")
        if base:
            found.append(os.path.join(base, "Python310", "python.exe"))
    for name in ("python", "python3"):
        path = _which(name)
        if path:
            found.append(path)
    found.append(sys.executable)
    unique: List[str] = []; seen = set()
    for path in found:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen and os.path.isfile(path):
            seen.add(key); unique.append(path)
    return unique


def find_windows_python(version: str = TARGET_VENV_PYTHON_WINDOWS) -> Optional[str]:
    for path in _windows_python_candidates():
        if _python_version_string(path) == version:
            return path
    return None


def ensure_python310_windows(version: str = TARGET_VENV_PYTHON_WINDOWS, log: Callable[[str], None] = print) -> str:
    """Windows: devolve o python.exe da versão exata (3.10.11). Se não existir, instala pelo winget
    (--scope user) ou, sem ele, pelo instalador do python.org (por usuário, sem administrador)."""
    found = find_windows_python(version)
    if found:
        return found
    log(f"[{APP_NAME}] Python {version} was not found on this Windows.")
    if not prompt_yes_no(f"[{APP_NAME}] Install Python {version} now?"):
        raise RuntimeError(f"Python {version} is required to create the {APP_NAME} environment.")
    if winget_available():
        winget_install(PYTHON_WINGET_ID, ["--version", version, "--scope", "user"], log=log)
        found = find_windows_python(version)
        if found:
            return found
    installer = os.path.join(tempfile.gettempdir(), f"python-{version}-amd64.exe")
    if download_file(PYTHON_WINDOWS_INSTALLER_URL, installer, log):
        _run([installer, "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
              "Include_test=0", "Include_pip=1"])
        found = find_windows_python(version)
        if found:
            return found
    raise RuntimeError(
        f"Could not install Python {version}. Install it by hand and start {APP_NAME} again:\n"
        f"  winget install --id {PYTHON_WINGET_ID} -e --version {version}\n"
        f"  or {PYTHON_WINDOWS_INSTALLER_URL}")


# ==========================================================================
# ─── VENV ─────────────────────────────────────────────────────────────────
# ==========================================================================

def venv_paths(env_name: str = APP_NAME) -> dict:
    """Caminhos do venv dedicado: Linux ~/.venv/CODEEP/bin/python; Windows %USERPROFILE%\\.venv\\CODEEP\\Scripts."""
    venv_dir = platform_utils.default_venv_dir(env_name)
    python = platform_utils.venv_python(venv_dir)
    scripts = platform_utils.venv_scripts_dir(venv_dir)
    return {
        "app_dir": str(APP_DIR), "venv_dir": venv_dir, "python": python,
        "pip": os.path.join(scripts, "pip.exe" if IS_WINDOWS else "pip"),
        "pythonw": platform_utils.venv_pythonw(venv_dir),
    }


def find_system_python() -> str:
    """Python fora de qualquer venv, para criar o venv do CODEEP (Linux)."""
    if not (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        return sys.executable
    for candidate in ("python3", "python3.12", "python3.11", "python3.10", "python3.9", "python"):
        found = _which(candidate)
        if found:
            return found
    return sys.executable


def ensure_venv(env_name: str = APP_NAME, log: Callable[[str], None] = print) -> dict:
    """Cria o venv se não existir. No Windows garante antes o runtime Visual C++ e usa o Python 3.10.11."""
    p = venv_paths(env_name)
    venv_dir = Path(p["venv_dir"])
    if IS_WINDOWS:
        ensure_vcredist_windows(log)
    recreate = False
    if venv_dir.is_dir() and Path(p["python"]).is_file() and IS_WINDOWS:
        recreate = _python_version_string(p["python"]) != TARGET_VENV_PYTHON_WINDOWS
    if recreate:
        shutil.rmtree(venv_dir, ignore_errors=True)
    if not venv_dir.is_dir() or not Path(p["python"]).is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        base_python = ensure_python310_windows(log=log) if IS_WINDOWS else find_system_python()
        res = _run([base_python, "-m", "venv", str(venv_dir)])
        if res.returncode != 0:
            raise RuntimeError(f"Falha ao criar venv com {base_python}:\n{res.stdout}\n"
                               + ("" if IS_WINDOWS else "Verifique se 'python3-venv' está instalado:\n  sudo apt install python3-venv"))
        _run([p["python"], "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    return p


def running_in_venv(env_name: str = APP_NAME) -> bool:
    """O Python atual é o do venv do CODEEP? No Windows o atalho abre com pythonw.exe: é o mesmo venv."""
    try:
        paths = venv_paths(env_name)
        if IS_WINDOWS:
            allowed = {os.path.normcase(os.path.abspath(paths["python"])), os.path.normcase(os.path.abspath(paths["pythonw"]))}
            return os.path.normcase(os.path.abspath(sys.executable)) in allowed
        return Path(sys.executable).resolve() == Path(paths["python"]).resolve()
    except Exception:
        return False


def has_module(pybin: str, module_name: str) -> bool:
    return _run([pybin, "-c", f"import {module_name}"]).returncode == 0


# ==========================================================================
# ─── ATALHOS DO WINDOWS ───────────────────────────────────────────────────
# ==========================================================================

def _windows_known_folder(csidl: int) -> str:
    """Pasta especial do Windows (Área de Trabalho, Menu Iniciar), mesmo redirecionada pelo OneDrive."""
    try:
        import ctypes
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer) == 0 and buffer.value:  # type: ignore[attr-defined]
            return buffer.value
    except Exception:
        pass
    return ""


_SHORTCUT_POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
foreach ($path in ($env:CODEEP_LNK_PATHS -split '\|')) {
  if (-not $path) { continue }
  $link = $shell.CreateShortcut($path)
  $link.TargetPath = $env:CODEEP_LNK_TARGET
  $q = [char]34
  $link.Arguments = '-X utf8 ' + $q + $env:CODEEP_LNK_SCRIPT + $q
  $link.WorkingDirectory = $env:CODEEP_LNK_WORKDIR
  $link.Description = 'CODEEP - Deep Learning Pipeline for Ligand Generation'
  if ($env:CODEEP_LNK_ICON -and (Test-Path $env:CODEEP_LNK_ICON)) { $link.IconLocation = $env:CODEEP_LNK_ICON }
  $link.Save()
}
"""


def windows_shortcut_paths() -> List[str]:
    """Onde ficam os atalhos CODEEP.lnk: Área de Trabalho e Menu Iniciar > Programas."""
    paths = []
    for csidl in (0x10, 0x02):          # CSIDL_DESKTOPDIRECTORY, CSIDL_PROGRAMS
        folder = _windows_known_folder(csidl)
        if folder:
            paths.append(os.path.join(folder, "CODEEP.lnk"))
    return paths


def ensure_windows_shortcuts(script_path: str | Path, env_name: str = APP_NAME, force: bool = False,
                             log: Callable[[str], None] = print) -> bool:
    """Cria/atualiza CODEEP.lnk na Área de Trabalho e no Menu Iniciar. O atalho abre o CODEEP.py com o
    pythonw.exe do venv (sem janela de console; a saída vai para o botão Terminal). Só recria quando falta um
    atalho ou o destino mudou. Devolve True se os atalhos existem."""
    if not IS_WINDOWS:
        return False
    paths = venv_paths(env_name)
    target = paths["pythonw"] if os.path.isfile(paths["pythonw"]) else paths["python"]
    if not os.path.isfile(target):
        return False
    script = os.path.abspath(str(script_path)); app_dir = os.path.dirname(script)
    icon = os.path.join(app_dir, "ICONS", "CODEEP.ico")
    links = windows_shortcut_paths()
    if not links:
        return False
    signature = "\n".join([target, script, icon if os.path.isfile(icon) else ""] + links)
    marker = os.path.join(paths["venv_dir"], ".codeep_shortcuts")
    try:
        current = Path(marker).read_text(encoding="utf-8")
    except Exception:
        current = ""
    if not force and current == signature and all(os.path.isfile(link) for link in links):
        return True
    environment = os.environ.copy()
    environment.update({"CODEEP_LNK_PATHS": "|".join(links), "CODEEP_LNK_TARGET": target, "CODEEP_LNK_SCRIPT": script,
                        "CODEEP_LNK_WORKDIR": app_dir, "CODEEP_LNK_ICON": icon if os.path.isfile(icon) else ""})
    try:
        res = _run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-Command", _SHORTCUT_POWERSHELL], env=environment)
    except Exception as exc:
        log(f"[{APP_NAME}] Could not create the shortcuts: {exc}")
        return False
    if res.returncode != 0:
        log(f"[{APP_NAME}] Could not create the CODEEP.lnk shortcuts:\n{res.stdout}")
        return False
    try:
        Path(marker).write_text(signature, encoding="utf-8")
    except Exception:
        pass
    log(f"[{APP_NAME}] CODEEP.lnk shortcuts created on the Desktop and in the Start Menu.")
    return all(os.path.isfile(link) for link in links)


def restart_codeep() -> bool:
    """Reinicia o CODEEP num processo novo, para importar as bibliotecas recém-instaladas. Linux: substitui o
    processo (execve) e não retorna. Windows: inicia o novo processo destacado, com pythonw.exe, e devolve True
    — quem chamou encerra o atual. CODEEP_RESTARTED=1 impede um laço de reinícios."""
    script = str(APP_DIR / "CODEEP.py")
    if not os.path.isfile(script):
        return False
    environment = os.environ.copy(); environment["CODEEP_RESTARTED"] = "1"; environment["CODEEP_VENV_ACTIVE"] = "1"
    arguments = [sys.executable, script] + [a for a in sys.argv[1:] if a != "--setup"]
    try:
        if IS_WINDOWS:
            pythonw = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "pythonw.exe")
            if os.path.isfile(pythonw):
                arguments[0] = pythonw
            environment["PYTHONUTF8"] = "1"
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            subprocess.Popen(arguments, env=environment, cwd=str(APP_DIR), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
            return True
        os.execve(sys.executable, arguments, environment)
    except Exception as exc:
        print(f"[{APP_NAME}] Could not restart: {exc}")
    return False


# ==========================================================================
# ─── BOOTSTRAP (Windows) E --setup ────────────────────────────────────────
# ==========================================================================

def bootstrap_windows(interactive: bool = True) -> bool:
    """Windows: garante o venv e o PyQt5 e, se o Python atual não é o do venv, reabre o CODEEP dentro dele.
    Devolve True se pode seguir NESTE processo; se reabrir, encerra o atual com o código do filho."""
    env_name = APP_NAME
    if os.environ.get(f"{env_name}_VENV_ACTIVE") == "1" and running_in_venv(env_name):
        return True
    paths = venv_paths(env_name)
    if running_in_venv(env_name) and has_module(paths["python"], "PyQt5"):
        # já dentro do venv (atalho/pythonw): marca o ambiente para que os processos filhos (multiprocessing
        # "spawn" reimporta o CODEEP.py) não repitam o bootstrap
        os.environ[f"{env_name}_VENV_ACTIVE"] = "1"
        return True
    venv_ok = os.path.isdir(paths["venv_dir"]) and os.path.isfile(paths["python"])
    if interactive and not venv_ok and not stdin_is_tty() and not assume_yes():
        print(f"[{APP_NAME}] A primeira execução precisa de um terminal para preparar o ambiente "
              f"({paths['venv_dir']}). Dê um duplo clique em CODEEP_Windows.bat ou execute Install-CODEEP.ps1.")
        return False
    if interactive and not running_in_venv(env_name):
        print("\n" + "═" * 62)
        print(f"  {APP_NAME} — Configuração do ambiente Python")
        print("═" * 62)
        print(f"  Python atual   : {sys.executable}")
        print(f"  Ambiente alvo  : {paths['venv_dir']}")
        print(f"  Python do venv : {TARGET_VENV_PYTHON_WINDOWS}")
        if not prompt_yes_no("  Continuar?"):
            return False
    ensure_venv(env_name)
    if not has_module(paths["python"], "PyQt5"):
        print(f"[{APP_NAME}] Instalando o PyQt5 no venv...")
        res = _run([paths["python"], "-m", "pip", "install", "--upgrade", ESSENTIAL_PACKAGES[0]])
        if res.returncode != 0:
            print(res.stdout)
            return False
    env = os.environ.copy(); env[f"{env_name}_VENV_ACTIVE"] = "1"; env["PYTHONUTF8"] = "1"
    exe = paths["python"] if stdin_is_tty() or not os.path.isfile(paths["pythonw"]) else paths["pythonw"]
    # o Windows não tem exec(): executa o filho (pythonw sem terminal) e espera
    sys.exit(subprocess.call([exe] + sys.argv, env=env))


def run_setup() -> int:
    """`python CODEEP.py --setup` (usado pelo Install-CODEEP.ps1): prepara tudo sem abrir a interface — venv,
    runtime Visual C++, pacotes essenciais e, no Windows, os atalhos CODEEP.lnk. Devolve 0 se deu certo."""
    try:
        paths = ensure_venv(APP_NAME)
        print(f"[{APP_NAME}] Instalando os pacotes essenciais: {', '.join(ESSENTIAL_PACKAGES)}")
        res = _run([paths["python"], "-m", "pip", "install", "--upgrade"] + ESSENTIAL_PACKAGES)
        if res.returncode != 0:
            print(res.stdout)
            return 1
        Path(paths["venv_dir"], ".codeep_minimal_done").write_text("ok", encoding="utf-8")
        if IS_WINDOWS:
            ensure_windows_shortcuts(APP_DIR / "CODEEP.py", force=True)
        print(f"[{APP_NAME}] Preparação concluída: {paths['venv_dir']}")
        return 0
    except Exception as exc:
        print(f"[{APP_NAME}] Falha na preparação: {exc}")
        return 1


# ==========================================================================
# ─── PROGRAMAS EXTERNOS: Java, P2Rank, xtb ────────────────────────────────
# ==========================================================================

def detect_java() -> dict:
    """Versão do Java disponível e se atende ao mínimo exigido pelo P2Rank."""
    java = platform_utils.find_java()
    if not java:
        return {"found": False, "version": None, "major": None, "ok": False}
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10,
                             encoding="utf-8", errors="replace")
        txt = (out.stderr or out.stdout or "").splitlines()
        ver = txt[0].split('"')[1] if txt and '"' in txt[0] else "?"
    except Exception:
        ver = "?"
    parts = re.split(r"[._\-+]", ver)
    try:
        major = int(parts[1]) if parts[0] == "1" and len(parts) > 1 else int(parts[0])
    except (ValueError, IndexError):
        major = None
    return {"found": True, "version": ver, "major": major, "ok": major is not None and major >= JAVA_MIN_MAJOR}


def install_java(log: Callable[[str], None] = print) -> bool:
    """Windows: instala o OpenJDK 17 pelo winget. Linux: não instala (exige root) — mostra o comando."""
    if not IS_WINDOWS:
        log(f"WARNING  Java {JAVA_MIN_MAJOR}+ requires root on Linux. Run in a terminal:\n         {JAVA_INSTALL_HINT_LINUX}")
        return False
    if not winget_available():
        log("FAILED  winget not available. Install Java 17+ by hand: https://learn.microsoft.com/java/openjdk/download")
        return False
    ok = winget_install(JAVA_WINGET_ID, log=log)
    if ok and detect_java()["ok"]:
        log("OK  Java installed.")
        return True
    log("FAILED  Java installation. Run in a terminal:\n         " + JAVA_INSTALL_HINT_WINDOWS)
    return detect_java()["ok"]


def detect_p2rank() -> Optional[str]:
    return platform_utils.find_p2rank(TOOLS_DIR)


def install_p2rank(version: str = P2RANK_VERSION, log: Callable[[str], None] = print) -> bool:
    """Baixa e extrai o P2Rank em <projeto>/tools/. Não requer administrador. O mesmo tar.gz traz `prank`
    (bash) e `prank.bat` (Windows). O Java 17+ é instalado à parte (winget no Windows)."""
    import tarfile
    url = P2RANK_URL_TMPL.format(v=version)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tarball = TOOLS_DIR / f"p2rank_{version}.tar.gz"
    try:
        if not download_file(url, tarball, log):
            return False
        log(f"$ extract {tarball.name} -> {TOOLS_DIR}")
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(TOOLS_DIR)
        tarball.unlink(missing_ok=True)
    except Exception as exc:
        log(f"FAILED  P2Rank download/extract: {exc}")
        return False
    prank = detect_p2rank()
    if not prank:
        log("FAILED  P2Rank extracted but the launcher was not found in tools/")
        return False
    if not IS_WINDOWS:
        try:
            os.chmod(prank, 0o755)
        except OSError:
            pass
    java = detect_java()
    if not java["ok"]:
        atual = java["version"] or "ausente"
        log(f"WARNING  P2Rank instalado em {prank}, mas o Java disponível é {atual} e ele exige {JAVA_MIN_MAJOR}+.")
        log(f"         Instale com: {java_install_hint()}")
    else:
        log(f"OK  P2Rank {version} installed at {prank}")
    return True


def detect_xtb() -> Optional[str]:
    return platform_utils.find_xtb(TOOLS_DIR)


def install_xtb(version: str = XTB_VERSION, log: Callable[[str], None] = print) -> bool:
    """Baixa e extrai o xtb em <projeto>/tools/. Não requer administrador. Linux: tar.xz do release; Windows:
    zip `windows-x86_64` (xtb.exe + libiomp5md.dll, parâmetros GFN em share/xtb/)."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        url = XTB_URL_WINDOWS.format(v=version); archive = TOOLS_DIR / f"xtb-{version}-windows-x86_64.zip"
    else:
        url = XTB_URL_LINUX.format(v=version); archive = TOOLS_DIR / f"xtb-{version}-linux-x86_64.tar.xz"
    try:
        if not download_file(url, archive, log):
            return False
        log(f"$ extract {archive.name} -> {TOOLS_DIR}")
        if IS_WINDOWS:
            import zipfile
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(TOOLS_DIR)
        else:
            import tarfile
            with tarfile.open(archive, "r:xz") as tf:
                tf.extractall(TOOLS_DIR)
        archive.unlink(missing_ok=True)
    except Exception as exc:
        log(f"FAILED  xtb download/extract: {exc}")
        return False
    xtb_bin = detect_xtb()
    if not xtb_bin:
        log("FAILED  xtb extracted but 'bin/xtb' was not found in tools/")
        return False
    if not IS_WINDOWS:
        try:
            os.chmod(xtb_bin, 0o755)
        except OSError:
            pass
    log(f"OK  xtb {version} installed at {xtb_bin}")
    return True
