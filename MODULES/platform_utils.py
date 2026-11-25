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

"""O que muda entre Linux e Windows no CODEEP, num só lugar (só biblioteca padrão, sem Qt).

Os demais módulos perguntam aqui em que sistema estão rodando e onde ficam as ferramentas externas
(P2Rank, Java, xtb), em vez de cada um repetir o seu `if`. No Linux nada muda: tudo continua nos
mesmos caminhos de antes. Mesmo padrão do CODOC (MODULES/platform_utils.py).
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"


# ----------------------------------------------------------------------------- fluxos padrão e processos filhos
def ensure_std_streams() -> None:
    """Windows aberto pelo atalho (pythonw.exe): não há console e `sys.stdout`/`sys.stderr` são None, o que
    faz qualquer `print` levantar AttributeError. Troca por um destino nulo; o log da sessão (arquivo)
    continua sendo gravado por quem chama `_setup_terminal_capture`. No Linux não faz nada."""
    for nome in ("stdout", "stderr"):
        if getattr(sys, nome) is None:
            setattr(sys, nome, open(os.devnull, "w", encoding="utf-8"))
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    # console do Windows em cp1252 + "✅"/"Å" no texto: troca o caractere em vez de levantar UnicodeEncodeError
    for nome in ("stdout", "stderr"):
        try:
            getattr(sys, nome).reconfigure(errors="replace")
        except Exception:
            pass


def install_subprocess_defaults() -> None:
    """Windows: faz todo processo filho nascer sem janela de console e com um stdin válido.

    O CODEEP aberto pelo atalho roda em `pythonw.exe` (sem console): sem isso cada `pip`, `xtb`, `prank`
    ou `nvidia-smi` abriria uma janela preta e, sem stdin, o subprocess falharia com "handle inválido".
    Também decodifica a saída das ferramentas com `errors="replace"`, para que um byte fora da página de
    código do Windows não derrube a leitura do log. No Linux não faz nada."""
    if not IS_WINDOWS or getattr(subprocess.Popen, "_codeep_patched", False):
        return
    original_init = subprocess.Popen.__init__
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = no_window
        if kwargs.get("stdin") is None and (sys.stdin is None or not sys.stdin.isatty()):
            kwargs["stdin"] = subprocess.DEVNULL
        text_mode = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding") or kwargs.get("errors")
        if text_mode and kwargs.get("errors") is None:
            kwargs["errors"] = "replace"
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init  # type: ignore[method-assign]
    subprocess.Popen._codeep_patched = True    # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- pastas
def home_dir() -> str:
    return str(Path.home())


def state_dir() -> Path:
    """Onde ficam os arquivos de estado do CODEEP (log da sessão do botão Terminal)."""
    if IS_WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "CODEEP"
    return Path.home() / ".local" / "state" / "CODEEP"


def default_venv_dir(env_name: str = "CODEEP") -> str:
    """$HOME/.venv/CODEEP (Linux) ou %USERPROFILE%\\.venv\\CODEEP (Windows)."""
    return os.path.join(home_dir(), ".venv", env_name)


def venv_scripts_dir(venv_dir: str) -> str:
    """Pasta dos executáveis do venv: Scripts\\ no Windows, bin/ no Linux."""
    return os.path.join(venv_dir, "Scripts" if IS_WINDOWS else "bin")


def venv_python(venv_dir: Optional[str] = None) -> str:
    venv_dir = venv_dir or default_venv_dir()
    return os.path.join(venv_scripts_dir(venv_dir), "python.exe" if IS_WINDOWS else "python")


def venv_pythonw(venv_dir: Optional[str] = None) -> str:
    """Windows: pythonw.exe abre a interface sem console. Linux: o próprio python."""
    venv_dir = venv_dir or default_venv_dir()
    return os.path.join(venv_scripts_dir(venv_dir), "pythonw.exe") if IS_WINDOWS else venv_python(venv_dir)


def _program_dirs() -> list[str]:
    folders: list[str] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        value = os.environ.get(variable)
        if value and value not in folders:
            folders.append(value)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        folders.append(os.path.join(local, "Programs"))
    return folders


def default_workers(linux_default: int = 14) -> int:
    """Processos do DataLoader/Pool. No Windows o `spawn` reimporta o programa em cada processo (lento e
    pesado em memória); usa menos."""
    n = os.cpu_count() or 1
    return min(linux_default, max(0, n - 1)) if not IS_WINDOWS else min(4, max(0, n - 1))


# ----------------------------------------------------------------------------- executáveis
def exe_names(name: str) -> list[str]:
    """Nomes possíveis de um executável: `xtb` → ['xtb.exe', 'xtb'] no Windows; ['xtb'] no Linux."""
    return [name + ".exe", name] if IS_WINDOWS else [name]


def is_runnable(path: Path | str) -> bool:
    """Arquivo executável. No Windows `os.access(X_OK)` é verdadeiro para qualquer arquivo; basta existir."""
    p = Path(path)
    return p.is_file() and (IS_WINDOWS or os.access(p, os.X_OK))


# ----------------------------------------------------------------------------- Java (P2Rank)
def find_java() -> Optional[str]:
    """O `java`: no PATH, em JAVA_HOME e, no Windows, nas pastas onde os instaladores comuns o colocam.
    Logo depois de instalar o Java pelo winget o PATH deste processo ainda é o antigo, e por isso a procura."""
    found = shutil.which("java")
    if found:
        return found
    names = ["java.exe"] if IS_WINDOWS else ["java"]
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        for name in names:
            candidate = os.path.join(java_home, "bin", name)
            if os.path.isfile(candidate):
                return candidate
    if IS_WINDOWS:
        for base in _program_dirs():
            for pattern in ("Microsoft/jdk-*", "Eclipse Adoptium/*", "Java/*", "Zulu/*", "Amazon Corretto/*", "BellSoft/*"):
                for folder in sorted(glob.glob(os.path.join(base, *pattern.split("/"))), reverse=True):
                    candidate = os.path.join(folder, "bin", "java.exe")
                    if os.path.isfile(candidate):
                        return candidate
    return None


def java_environment() -> dict:
    """Ambiente para rodar programas Java (P2Rank): se o java só foi achado fora do PATH, o inclui."""
    environment = os.environ.copy()
    java = find_java()
    if java and not shutil.which("java"):
        bin_dir = os.path.dirname(java)
        environment["PATH"] = bin_dir + os.pathsep + environment.get("PATH", "")
        environment.setdefault("JAVA_HOME", os.path.dirname(bin_dir))
    return environment


# ----------------------------------------------------------------------------- P2Rank e xtb
def p2rank_launcher_names() -> list[str]:
    """O pacote do P2Rank traz `prank` (bash) e `prank.bat` (Windows)."""
    return ["prank.bat", "prank"] if IS_WINDOWS else ["prank"]


def find_in_dirs(patterns: list[str], bases: list[Path]) -> Optional[str]:
    """Primeiro arquivo executável que casa com algum padrão glob, em cada pasta-base (na ordem)."""
    for base in bases:
        if not Path(base).is_dir():
            continue
        for pattern in patterns:
            try:
                for cand in sorted(Path(base).glob(pattern)):
                    if is_runnable(cand):
                        return str(cand)
            except OSError:
                continue
    return None


def tool_search_roots(project_tools: Path) -> list[Path]:
    """Pastas onde procurar P2Rank/xtb instalados: <projeto>/tools, $HOME e, no Linux, /opt."""
    roots = [Path(project_tools), Path.home()]
    if not IS_WINDOWS:
        roots.append(Path("/opt"))
    return roots


def find_xtb(project_tools: Path) -> Optional[str]:
    """`xtb`: XTB_HOME, PATH, <projeto>/tools/xtb*/bin/xtb(.exe), $HOME e /opt (Linux)."""
    env_home = os.environ.get("XTB_HOME")
    if env_home:
        for name in exe_names("xtb"):
            cand = Path(env_home) / "bin" / name
            if is_runnable(cand):
                return str(cand)
    found = shutil.which("xtb")
    if found:
        return found
    return find_in_dirs([f"xtb*/bin/{n}" for n in exe_names("xtb")], tool_search_roots(project_tools))


def find_p2rank(project_tools: Path) -> Optional[str]:
    """Launcher do P2Rank: P2RANK_HOME, PATH, <projeto>/tools/p2rank*/prank(.bat), $HOME e /opt (Linux)."""
    env_home = os.environ.get("P2RANK_HOME")
    if env_home:
        for name in p2rank_launcher_names():
            cand = Path(env_home) / name
            if is_runnable(cand):
                return str(cand)
    for name in p2rank_launcher_names():
        found = shutil.which(name)
        if found:
            return found
    return find_in_dirs([f"p2rank*/{n}" for n in p2rank_launcher_names()], tool_search_roots(project_tools))


# ----------------------------------------------------------------------------- arquivos e pastas
def symlink_or_copy(src: Path | str, dst: Path | str) -> str:
    """Liga `dst` à pasta `src` sem copiar dados, na ordem: link simbólico → junção (Windows, sem
    privilégio) → cópia. Devolve o método usado: "symlink", "junction" ou "copy"."""
    src, dst = str(src), str(dst)
    try:
        os.symlink(src, dst, target_is_directory=os.path.isdir(src))
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if IS_WINDOWS and os.path.isdir(src):
        try:
            r = subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], capture_output=True, text=True)
            if r.returncode == 0 and os.path.isdir(dst):
                return "junction"
        except Exception:
            pass
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy"


def open_path(path: str) -> None:
    """Abre um arquivo ou pasta no programa padrão do sistema."""
    if IS_WINDOWS:
        os.startfile(path)  # type: ignore[attr-defined]
    elif IS_MAC:
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def follow_log_command(log_path: str) -> list[list[str]]:
    """Comandos candidatos para abrir um terminal que acompanha o log (`tail -F`)."""
    if IS_WINDOWS:
        ps = f"Get-Content -LiteralPath '{log_path}' -Wait -Tail 200 -Encoding UTF8"
        return [["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", ps]]
    return [
        ["x-terminal-emulator", "-e", "tail", "-n", "+1", "-F", log_path],
        ["gnome-terminal", "--", "tail", "-n", "+1", "-F", log_path],
        ["konsole", "-e", "tail", "-n", "+1", "-F", log_path],
        ["xfce4-terminal", "--title=CODEEP Terminal", "--command", f"tail -n +1 -F '{log_path}'"],
        ["xterm", "-T", "CODEEP Terminal", "-e", "tail", "-n", "+1", "-F", log_path],
    ]


# ----------------------------------------------------------------------------- hardware
def windows_cpu_name() -> str:
    """Nome do processador no Windows (registro), vazio se não conseguir."""
    if not IS_WINDOWS:
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return ""


def windows_cpu_mhz() -> float:
    """Clock nominal (MHz) no Windows, 0 se não conseguir."""
    if not IS_WINDOWS:
        return 0.0
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return float(winreg.QueryValueEx(k, "~MHz")[0])
    except Exception:
        return 0.0


def windows_memory_gb() -> tuple[float, float]:
    """(total, disponível) em GB no Windows, sem depender do psutil; (0, 0) se falhar."""
    if not IS_WINDOWS:
        return 0.0, 0.0
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = _MEMORYSTATUSEX(); s.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):  # type: ignore[attr-defined]
            return round(s.ullTotalPhys / 1e9, 1), round(s.ullAvailPhys / 1e9, 1)
    except Exception:
        pass
    return 0.0, 0.0


# ----------------------------------------------------------------------------- GPU NVIDIA
def find_nvidia_smi() -> Optional[str]:
    """`nvidia-smi`: no PATH e, no Windows, nas pastas onde o driver da NVIDIA o instala."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if IS_WINDOWS:
        candidates = [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")]
        for base in _program_dirs():
            candidates.append(os.path.join(base, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"))
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
    return None
