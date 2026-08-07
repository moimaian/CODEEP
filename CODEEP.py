#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# CODEEP – Deep Learning Pipeline for Ligand Generation
# Part of the CODOC · CODYN · CODEEP · CODRUG suite
# Copyright (C) 2024–2026 Moisés Maia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ==========================================================================================================================================
# ══════════════════════════ BOOTSTRAP PRÉ-QT ══════════════════════════════
#
# Fluxo completo na primeira execução (e em qualquer execução fora do venv):
#
#   1. Detecta se já está rodando dentro do venv ~/.venv/CODEEP
#      → Se sim: segue normalmente (PyQt5 já deve estar lá)
#
#   2. Se NÃO está no venv:
#      a. Pergunta ao usuário se deseja criar/usar o ambiente CODEEP
#      b. Cria ~/.venv/CODEEP com python -m venv (se não existir)
#      c. Atualiza pip/setuptools/wheel no venv
#      d. Instala PyQt5 no venv
#      e. Reinicia o processo inteiro usando o Python DO VENV via os.execve
#         → A partir daqui TUDO roda dentro do venv, incluindo Install Requirements
#
#   A variável de ambiente CODEEP_VENV_ACTIVE="1" protege contra loop infinito.
#   O venv é persistente: execuções futuras entram direto no passo 1 → seguem.
#
# ==========================================================================================================================================
from __future__ import annotations
import os, sys, subprocess
import logging

from ctypes.util import find_library

from pathlib import Path


# Caminho canônico do venv dedicado ao CODEEP
_VENV_DIR    = Path.home() / ".venv" / "CODEEP"
_VENV_PYTHON = _VENV_DIR / "bin" / "python"

log = logging.getLogger(__name__)

# ── Atalho no menu Linux ───────────────────────────────────────────────────
_APP_DIR      = Path(__file__).resolve().parent
_DESKTOP_DIR  = Path.home() / ".local" / "share" / "applications"
_DESKTOP_FILE = _DESKTOP_DIR / "CODEEP.desktop"

def _write_desktop() -> None:
    """Escreve (ou sobrescreve) o arquivo .desktop do CODEEP."""
    _DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    icon_path = _APP_DIR / "ICONS" / "CODEEP.png"
    content = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Name=CODEEP\n"
        "Comment=Deep Learning Pipeline for Ligand Generation\n"
        f"Exec=bash -i -c \"env PYTHONNOUSERSITE=1 '{_VENV_PYTHON}' '{_APP_DIR / 'CODEEP.py'}'\"\n"
        f"Icon={icon_path}\n"
        "Terminal=true\n"
        "Type=Application\n"
        "Categories=Qt;Science;Chemistry;Education;\n"
        "StartupNotify=false\n"
    )
    _DESKTOP_FILE.write_text(content, encoding="utf-8")
    _DESKTOP_FILE.chmod(0o755)


def _ensure_desktop() -> None:
    """Cria o atalho no menu Linux se não existir ou se o venv mudou de lugar."""
    try:
        if not _DESKTOP_FILE.exists():
            _write_desktop()
        else:
            # Recria se o caminho do venv no arquivo não bater com o atual
            # (usuário moveu a instalação ou recriou o venv)
            if str(_VENV_PYTHON) not in _DESKTOP_FILE.read_text(encoding="utf-8"):
                _write_desktop()
    except Exception:
        pass  # nunca travar a inicialização por causa do atalho

# ── Bootstrap ───────────────────────────────────────────────────
def _already_in_venv() -> bool:
    """Retorna True se o Python atual É o Python do venv CODEEP."""
    try:
        return Path(sys.executable).resolve() == _VENV_PYTHON.resolve()
    except Exception:
        return False

def _bootstrap_venv():
    """
    Garante que o CODEEP rode sempre dentro do venv ~/.venv/CODEEP.
    Chamada antes de qualquer import pesado ou Qt.
    """
    # Proteção contra loop (caso execve falhe e o processo seja re-chamado)
    if os.environ.get("CODEEP_VENV_ACTIVE") == "1":
        _ensure_desktop()
        return

    # Já está no venv correto — segue sem perguntar
    if _already_in_venv():
        os.environ["CODEEP_VENV_ACTIVE"] = "1"
        _ensure_desktop()
        return

    # ── Informa o usuário e pede confirmação ──────────────────────────────
    print("\n" + "═" * 62)
    print("  CODEEP — Configuração do Ambiente")
    print("═" * 62)
    print(f"  Python atual   : {sys.executable}")
    print(f"  Versão Python  : {sys.version.split()[0]}")
    print(f"  Ambiente alvo  : {_VENV_DIR}")

    if _VENV_DIR.is_dir() and _VENV_PYTHON.is_file():
        print("  Status         : ✔ venv já existe")
    else:
        print("  Status         : ○ venv será criado agora")

    print()
    print("  O CODEEP utiliza um ambiente virtual isolado (~/.venv/CODEEP)")
    print("  para evitar conflitos com outros projetos Python.")
    print()
    try:
        resp = input("  Deseja continuar? [S/n]: ").strip().lower()
        if resp in ("n", "nao", "não", "no"):
            print("\n[CODEEP] Cancelado pelo usuário.")
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        print("\n[CODEEP] Cancelado.")
        sys.exit(0)

    # ── Cria o venv se necessário ─────────────────────────────────────────
    if not (_VENV_DIR.is_dir() and _VENV_PYTHON.is_file()):
        print(f"\n[CODEEP] Criando venv em {_VENV_DIR} ...")
        _VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [sys.executable, "-m", "venv", str(_VENV_DIR)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if res.returncode != 0:
            print(f"[CODEEP] ❌ Falha ao criar venv:\n{res.stdout}")
            print("         Verifique se 'python3-venv' está instalado:")
            print("         sudo apt install python3-venv")
            sys.exit(1)
        print("[CODEEP] ✅ Venv criado.")

        # Atualiza pip/setuptools/wheel no venv
        print("[CODEEP] Atualizando pip/setuptools/wheel...")
        subprocess.run(
            [str(_VENV_PYTHON), "-m", "pip", "install", "--upgrade",
             "pip", "setuptools", "wheel"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    else:
        print(f"\n[CODEEP] Usando venv existente: {_VENV_DIR}")

    # ── Garante PyQt5 no venv ─────────────────────────────────────────────
    # Verifica sem importar (o venv ainda não está ativo no processo atual)
    check = subprocess.run(
        [str(_VENV_PYTHON), "-c", "import PyQt5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if check.returncode != 0:
        print("[CODEEP] Instalando PyQt5 no venv...")
        res = subprocess.run(
            [str(_VENV_PYTHON), "-m", "pip", "install", "--upgrade", "PyQt5"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if res.returncode != 0:
            print(f"[CODEEP] ❌ Falha ao instalar PyQt5:\n{res.stdout}")
            sys.exit(1)
        print("[CODEEP] ✅ PyQt5 instalado no venv.")
    else:
        print("[CODEEP] ✅ PyQt5 já disponível no venv.")

    # ── Reinicia o processo usando o Python do venv ───────────────────────
    _ensure_desktop() 
    print(f"\n[CODEEP] Reiniciando com o Python do venv...")
    print(f"         {_VENV_PYTHON}\n")
    env = os.environ.copy()
    env["CODEEP_VENV_ACTIVE"] = "1"    
    # os.execve substitui o processo atual — não retorna
    os.execve(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv, env)


_bootstrap_venv()   # ← executa ANTES de qualquer import Qt ou científico

# ==========================================================================================================================================
# ═══════════════════════════ IMPORTS PRINCIPAIS ═══════════════════════════
# ==========================================================================================================================================
import shutil, json, time, importlib, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── PyQt5 ─────────────────────────────────────────────────────────────────
from PyQt5.QtCore    import Qt, QThread, pyqtSignal, QTimer, QDate
from PyQt5.QtGui     import QPixmap, QGuiApplication, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QAction,
    QSplitter, QFrame, QScrollArea, QDialog,
    QTableWidget, QTableWidgetItem, QRadioButton, QButtonGroup,
    QDateEdit,
)
print("✅ PyQt5 importado com sucesso.")

# ── Módulo de requisitos ───────────────────────────────────────────────────
_REQ_MODULE_OK = False
try:
    # Garante que MODULES/ está no path
    _HERE = Path(__file__).parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from CODEEP_requirements import RequirementsInstaller, detect_hardware
    _REQ_MODULE_OK = True
    print("✅ CODEEP_requirements importado com sucesso.")
except ImportError as e:
    print(f"⚠️  CODEEP_requirements não encontrado: {e}")

# ── Bibliotecas opcionais ─────────────────────────────────────────────────
try:
    import psutil;      _PSUTIL_OK = True
except ImportError:     _PSUTIL_OK = False

try:
    import pandas as pd; _PANDAS_OK = True
except ImportError:      _PANDAS_OK = False

try:
    import matplotlib; matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import torch
    _TORCH_OK      = True
    _DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _TORCH_OK       = False
    _DEVICE_DEFAULT = "cpu"

def _has_nvrtc_runtime() -> bool:
    if find_library("nvrtc"):
        return True

    cuda_major = None
    if _TORCH_OK:
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            cuda_major = cuda_version.split(".", 1)[0]

    lib_names = ["libnvrtc.so"]
    if cuda_major:
        lib_names.insert(0, f"libnvrtc.so.{cuda_major}")

    search_dirs = []
    for env_var in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"):
        raw = os.environ.get(env_var, "")
        if not raw:
            continue
        for entry in raw.split(os.pathsep):
            if entry:
                search_dirs.append(entry)
                search_dirs.append(os.path.join(entry, "lib64"))
                search_dirs.append(os.path.join(entry, "lib"))

    search_dirs.extend([
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
    ])

    seen = set()
    for directory in search_dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        for lib_name in lib_names:
            if os.path.exists(os.path.join(directory, lib_name)):
                return True
    return False


def _detect_cupy_support() -> bool:
    if _DEVICE_DEFAULT != "cuda":
        return False
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return _has_nvrtc_runtime()


_CUPY_OK = _detect_cupy_support()

try:
    import cupy as cp
    _t = cp.zeros((2, 4), dtype=cp.float32)
    cp.einsum('ij->j', _t)
    del _t, cp
    _CUPY_OK = True
except Exception:
    _CUPY_OK = False


_BATCH_DEFAULT = 500 if _CUPY_OK else 100

# ==========================================================================================================================================
# ════════════════════════ CONSTANTES DE ESTILO — DARK MODE ════════════════
# Paleta inspirada na splash screen (#0D1B2A base, #2ECC71 acento verde).
#
# Cores base:
#   BG_DEEP   #0D1B2A  — fundo mais escuro (janela, painéis)
#   BG_MID    #14253A  — fundo médio (tabs, groupboxes)
#   BG_LIGHT  #1C3249  — fundo dos inputs, hover
#   BORDER    #2A4A6B  — bordas sutis
#   FG_MAIN   #C9D1D9  — texto principal
#   FG_DIM    #6E8CA8  — texto secundário / labels
#   ACCENT    #2ECC71  — verde principal (acento, seleção, títulos)
#   ACCENT2   #27AE60  — verde escuro (hover, barra de progresso)
#   DANGER    #E74C3C  — vermelho (botão Stop)
#   WARN      #E67E22  — laranja (avisos)
# ==========================================================================================================================================
_SS_MAIN = """
QMainWindow, QWidget {
    background-color: #0D1B2A;
    font-family: 'DejaVu Sans', 'Segoe UI', sans-serif;
    font-size: 10pt;
    color: #C9D1D9;
}
QTabWidget::pane {
    border: 1px solid #2A4A6B;
    border-radius: 6px;
    background: #14253A;
}
QTabBar::tab {
    background: #0D1B2A;
    color: #6E8CA8;
    padding: 7px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
    min-width: 110px;
}
QTabBar::tab:selected {
    background: #14253A;
    color: #F57C6F;
    border-bottom: 3px solid #F57C6F;
}
QTabBar::tab:hover:!selected {
    background: #1C3249;
    color: #C9D1D9;
}
QGroupBox {
    font-weight: bold;
    border: 1.5px solid #2A4A6B;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    background: #14253A;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #F57C6F;
}
QGroupBox QLabel {
    background: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    padding: 4px 8px;
    background: #1C3249;
    color: #C9D1D9;
    selection-background-color: #27AE60;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QTextEdit:focus {
    border: 1.5px solid #2ECC71;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #1C3249;
    color: #C9D1D9;
    border: 1px solid #2A4A6B;
    selection-background-color: #27AE60;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background: #0D1B2A;
    border: none;
}
QScrollBar:vertical {
    background: #0D1B2A;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2A4A6B;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #2ECC71; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0D1B2A;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #2A4A6B;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background: #2ECC71; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressBar {
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    text-align: center;
    background: #1C3249;
    color: #C9D1D9;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #145A32, stop:1 #1E8449);
    border-radius: 4px;
}
QCheckBox { color: #C9D1D9; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #2A4A6B;
    border-radius: 3px;
    background: #1C3249;
}
QCheckBox::indicator:checked {
    background: #2ECC71;
    border-color: #2ECC71;
}
QRadioButton { color: #C9D1D9; }
QFrame[frameShape="4"] { color: #2A4A6B; }
QMenuBar {
    background: #0D1B2A;
    color: #C9D1D9;
    border-bottom: 1px solid #2A4A6B;
}
QMenuBar::item:selected { background: #1C3249; color: #F57C6F; }
QMenu {
    background: #14253A;
    color: #C9D1D9;
    border: 1px solid #2A4A6B;
}
QMenu::item:selected { background: #1C3249; color: #F57C6F; }
QSplitter::handle { background: #2A4A6B; }
QTableWidget {
    background: #14253A;
    color: #C9D1D9;
    gridline-color: #2A4A6B;
    alternate-background-color: #1C3249;
}
QHeaderView::section {
    background: #2ECC71;
    color: #0D1B2A;
    font-weight: bold;
    padding: 4px;
    border: none;
}
QToolTip {
    background: #1C3249;
    color: #C9D1D9;
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    padding: 4px;
}
"""

_SS_BTN_PRIMARY = """
QPushButton {
    background: #1A4A30;
    color: #2ECC71;
    font-weight: bold;
    border: 1px solid #2ECC71;
    border-radius: 5px;
    padding: 6px 16px;
}
QPushButton:hover   { background: #27AE60; color: #0D1B2A; }
QPushButton:pressed { background: #1E8449; color: #FFF; }
QPushButton:disabled{ background: #1C2E20; color: #3D6B4A; border-color: #2A4A36; }
"""

_SS_BTN_DANGER = """
QPushButton {
    background: #4A1A1A;
    color: #E74C3C;
    font-weight: bold;
    border: 1px solid #E74C3C;
    border-radius: 5px;
    padding: 6px 16px;
}
QPushButton:hover   { background: #E74C3C; color: #FFF; }
QPushButton:pressed { background: #C0392B; color: #FFF; }
QPushButton:disabled{ background: #2A1A1A; color: #6B3030; border-color: #4A2A2A; }
"""

_SS_BTN_NAV = """
QPushButton {
    background: transparent;
    border: none;
    color: #2A4A6B;
    font-size: 12pt;
    font-weight: bold;
}
QPushButton:hover { color: #2ECC71; }
"""

_SS_LOG = """
QTextEdit {
    background: #060D14;
    color: #C9D1D9;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    border: 1px solid #2A4A6B;
    border-radius: 4px;
}
"""

_SS_LABEL_TITLE   = "font-size:14pt; font-weight:bold; color:#2ECC71;"
_SS_LABEL_SECTION = "font-size:11pt; font-weight:bold; color:#C9D1D9;"


# ==========================================================================================================================================
# ═══════════════════════════ WORKER GENÉRICO ══════════════════════════════
# Arquitectura de dois níveis:
#
#   PipelineWorker (QThread)
#     └─ _ChildProcess (multiprocessing.Process)
#           └─ fn(*args, **kwargs)   ← pipeline real (download, treino, etc.)
#
# O pipeline roda em um processo filho independente do CODEEP.
# Stop chama proc.terminate() + proc.kill() no filho — o CODEEP continua
# aberto normalmente. stdout/stderr do filho são lidos via multiprocessing.Queue
# e emitidos como sinais Qt linha a linha.
#
# Progresso: parseia linhas do tqdm ("NN%|" ou "N/M") para valor real.
# ETA:       calculado a partir do progresso real e tempo decorrido.
# ==========================================================================================================================================
import io, contextlib, traceback, time as _time, re as _re
import multiprocessing as _mp

# Função de topo de módulo necessária para multiprocessing no Linux (fork-safe)
def _child_run(fn, args, kwargs, queue):
    """Executada no processo filho. Redireciona stdout/stderr para a queue."""
    import io, sys, contextlib, traceback

    # Garante UTF-8 no processo filho (spawn não herda locale do pai)
    import locale, os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass

    class _QueueWriter(io.TextIOBase):
        """Escreve linhas na queue compartilhada entre filho e pai."""
        def __init__(self, q):
            super().__init__()
            self._q   = q
            self._buf = ""

        def write(self, s):
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._q.put(("line", line))
            return len(s)

        def flush(self):
            if self._buf:
                self._q.put(("line", self._buf))
                self._buf = ""

    writer = _QueueWriter(queue)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            fn(*args, **kwargs)
        queue.put(("done", True))
    except Exception:
        writer.write(traceback.format_exc())
        writer.flush()
        queue.put(("done", False))


class PipelineWorker(QThread):
    """
    QThread que gerencia um processo filho (multiprocessing.Process).
    O pipeline roda no filho; Stop() mata apenas o filho — CODEEP continua.
    Emite log_signal, progress_signal, eta_signal e finished_signal.
    """
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)   # 0-100, valor real do tqdm
    eta_signal      = pyqtSignal(str)   # ex: "2m 30s"
    finished_signal = pyqtSignal(bool)  # True = sucesso

    # Regex para extrair progresso de linhas do tqdm e similares
    _RE_PCT  = _re.compile(r'\b(\d{1,3})%\|')
    _RE_FRAC = _re.compile(r'\b(\d+)/(\d+)\b')

    def __init__(self, fn, args=(), kwargs=None, parent=None):
        super().__init__(parent)
        self.fn       = fn
        self.args     = args
        self.kwargs   = kwargs or {}
        self._proc    = None   # multiprocessing.Process filho
        self._t_start = 0.0

    def run(self):
        """Lança o processo filho e faz polling na queue até ele terminar."""
        self._t_start = _time.monotonic()
        # Queue compartilhada: filho → pai (linhas de log + sinal de fim)
        ctx = _mp.get_context("spawn")  # spawn obrigatório: CUDA não suporta fork
                                 # (~15s overhead amortizado sobre todo o dataset)
        queue = ctx.Queue()

        self._proc = ctx.Process(
            target = _child_run,
            args   = (self.fn, self.args, self.kwargs, queue),
            daemon = False,
        )
        self._proc.start()

        # Polling: lê fila até o filho sinalizar término
        success = False
        while True:
            try:
                msg_type, payload = queue.get(timeout=0.1)
            except Exception:
                # Timeout — verifica se o filho ainda está vivo
                if self._proc is not None and not self._proc.is_alive():
                    # Filho morreu sem enviar "done" (ex: killed pelo Stop)
                    break
                continue

            if msg_type == "line":
                self.log_signal.emit(payload)
                self._parse_progress(payload)
            elif msg_type == "done":
                success = payload
                break

        self.finished_signal.emit(success)

    def _parse_progress(self, line: str):
        """Extrai % real do tqdm e calcula ETA."""
        pct = None
        m = self._RE_PCT.search(line)
        if m:
            pct = min(int(m.group(1)), 99)
        else:
            m = self._RE_FRAC.search(line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    pct = min(int(cur / total * 100), 99)

        if pct is not None and 0 <= pct <= 100:
            self.progress_signal.emit(pct)
            elapsed = _time.monotonic() - self._t_start
            if pct > 0:
                remaining = max(0, elapsed / (pct / 100) - elapsed)
                self.eta_signal.emit(_fmt_eta(remaining))
            else:
                self.eta_signal.emit("—")

    def request_stop(self):
        """Para o download terminando apenas o processo filho.
        O CODEEP continua aberto normalmente."""
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()          # SIGTERM suave
            proc.join(timeout=3)      # aguarda até 3 s
            if proc.is_alive():
                proc.kill()           # SIGKILL se necessário
            self._proc = None
        # Emite sinal para restaurar os botões da UI
        self.finished_signal.emit(False)


def _fmt_eta(seconds: float) -> str:
    """Formata segundos restantes em string legível."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    else:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h}h {m:02d}m"


class _LineBuffer(io.TextIOBase):
    """Mantido por compatibilidade com etapas que não usam subprocess
    (ex: exploração, avaliação local). Emite cada linha como sinal Qt."""
    def __init__(self, signal, progress_cb=None):
        super().__init__()
        self._signal      = signal
        self._progress_cb = progress_cb
        self._buf         = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._signal.emit(line)
            if self._progress_cb:
                self._progress_cb(line)
        return len(s)

    def flush(self):
        if self._buf:
            self._signal.emit(self._buf)
            if self._progress_cb:
                self._progress_cb(self._buf)
            self._buf = ""


# ==========================================================================================================================================
# ═════════════════════════ HELPERS DE UI ══════════════════════════════════
# ==========================================================================================================================================

def _mk_title(text):
    lbl = QLabel(text); lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(_SS_LABEL_TITLE); return lbl

def _mk_section(text):
    lbl = QLabel(text); lbl.setStyleSheet(_SS_LABEL_SECTION); return lbl

def _mk_btn(text, style=_SS_BTN_PRIMARY, width=130):
    btn = QPushButton(text); btn.setFixedWidth(width)
    btn.setStyleSheet(style); return btn

def _mk_nav_row(back_cb, next_cb):
    lay = QHBoxLayout()
    for txt, cb, align in [(" << BACK ", back_cb, Qt.AlignLeft),
                            (" NEXT >> ", next_cb, Qt.AlignRight)]:
        b = QPushButton(txt); b.setFixedWidth(200)
        b.setStyleSheet(_SS_BTN_NAV); b.clicked.connect(cb)
        lay.addWidget(b, alignment=align)
    return lay

def _mk_separator():
    sep = QFrame(); sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    sep.setStyleSheet("color:#2A4A6B;"); return sep

def _mk_log_box(height=220):
    box = QTextEdit(); box.setReadOnly(True)
    box.setStyleSheet(_SS_LOG); box.setFixedHeight(height); return box

def _log(box, line):
    box.append(line)
    box.verticalScrollBar().setValue(box.verticalScrollBar().maximum())

def _mk_progress():
    """Cria QProgressBar em modo indeterminado até receber valor real do tqdm."""
    p = QProgressBar()
    p.setRange(0, 100); p.setValue(0)
    p.setFormat("Progress: %p%  —  ETA: —")
    p.setFixedHeight(22)
    return p

def _pulse(bar):
    """Incremento suave quando não há informação de progresso real na linha."""
    v = bar.value()
    if v < 30:          # avança livremente até 30%
        bar.setValue(v + 1)
    # acima de 30% só avança se tqdm emitir valor real

def _connect_worker(worker, log_box, prog_bar, btn_run, btn_stop, done_cb):
    """Conecta sinais do PipelineWorker aos widgets da aba.

    - log_signal   → appenda linha no log e faz pulse suave
    - progress_signal → atualiza a barra com valor real do tqdm
    - eta_signal   → atualiza o texto da barra com ETA calculado
    - finished_signal → restaura botões e chama done_cb
    """
    worker.log_signal.connect(
        lambda l: (_log(log_box, l), _pulse(prog_bar))
    )
    worker.progress_signal.connect(
        lambda pct: prog_bar.setValue(pct)
    )
    worker.eta_signal.connect(
        lambda eta: prog_bar.setFormat(f"Progress: %p%  —  ETA: {eta}")
    )
    worker.finished_signal.connect(lambda ok: (
        prog_bar.setValue(100),
        prog_bar.setFormat("Progress: 100%  —  Done"),
        btn_run.setEnabled(True),
        btn_stop.setEnabled(False),
        _log(log_box, "\n✅ Completed!" if ok else "\n❌ Interrupted or finished with errors."),
        done_cb(ok),
    ))
    btn_stop.clicked.connect(worker.request_stop)


def _pipeline_collect(dataset, moad_dir, phase1_max,
                      wants_phase1, wants_phase2, wants_phase3,
                      job_dir, org_terms, desc_terms, p3_ids):
    """Função de módulo (pickle-safe) que executa a coleta no processo filho."""
    import MODULES.pipeline_collect as m
    if wants_phase1 and dataset:
        m.collect_external_dataset(
            dataset=dataset, moad_dir=moad_dir, max_entries=phase1_max)
    if wants_phase2:
        m.collect_phase2(
            job_dir=job_dir,
            organism_terms=org_terms or None,
            description_terms=desc_terms or None)
    if wants_phase3:
        m.collect_phase3(
            job_dir=job_dir,
            extra_pdb_ids=p3_ids,
            target_list={})
    m.show_status(job_dir=job_dir)

def _pipeline_generate(root, job_dir, pdb, ckpt, n_samples, temperature,
                       cutoff, min_atoms, top_n, out, ligand_ref, site_pdb,
                       site_config_overrides=None):
    """Função de módulo (pickle-safe) que executa a geração no processo filho."""
    import MODULES.pipeline_generate as m
    from pathlib import Path as _P
    import numpy as np

    # PROJECT_DIR injetado aqui, no filho, após o re-import do módulo
    m.PROJECT_DIR = _P(root)

    pdb_path = _P(pdb)
    ckpt_dir = _P(ckpt)
    default_base = _P(job_dir) / "results" if job_dir else m.PROJECT_DIR / "results"
    out_dir  = _P(out) if out else default_base / pdb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    site_pdb_path = _P(site_pdb) if site_pdb else None

    # protonate_site/ph/site_qm_method — DEVEM bater com o que foi usado
    # para gerar os tensores de treino da fase (ver nota em
    # pipeline_generate.pdb_to_site_tensor sobre paridade treino/inferência).
    site_tensor, origin, voxel_size = m.pdb_to_site_tensor(
        pdb_path, cutoff, min_atoms, site_pdb=site_pdb_path,
        project_dir=_P(root),           # ver Problema 3
        config_overrides=site_config_overrides)
    np.savez_compressed(
        out_dir / "site_tensor.npz",
        tensor=site_tensor, origin=origin,
        voxel_size=np.array([voxel_size]))

    model, mod = m.load_model(
        ckpt_dir,
        project_dir=_P(root))           # ver Problema 3

    ref_mol = None
    if ligand_ref:
        ref_mol, ligand_ref_sdf = m.load_reference_ligand(_P(ligand_ref), out_dir)
        if ref_mol is None:
            raise RuntimeError("Não foi possível ler o ligante de referência selecionado.")

    candidates = m.generate_candidates(
        model, mod, site_tensor,
        tensor_dir=_P(root) / "data" / "tensors",
        n_samples=n_samples, temperature=temperature,
        voxel_size=voxel_size,
        ref_mol=ref_mol)              # ← voxel_size retornado por pdb_to_site_tensor
    if not candidates:
        raise RuntimeError("Nenhum candidato válido gerado.")

    df = m.rank_candidates(candidates, ref_mol=ref_mol)
    df.drop(columns=["_mol"], errors="ignore").to_csv(out_dir / "candidates.csv")
    m.save_sdf(df, out_dir / f"top{top_n}_candidates.sdf", top_n)
    m.save_grid_image(df, out_dir / f"top{top_n}_candidates.png", top_n)
    m.save_html_report(
        df, out_dir / "report.html",
        pdb_name=pdb_path.name, top_n=top_n,
        n_generated=n_samples, generation_time=0, ckpt_info={})
    print(f"\n✅ {len(df)} valid candidates saved to: {out_dir}")
    
# ==========================================================================================================================================
# ══════════════════════════ SPLASH SCREEN ═════════════════════════════════
# ==========================================================================================================================================
class SplashScreen(QDialog):
    ready_signal = pyqtSignal()

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(500, 300)
        self.setStyleSheet("background:#0D1B2A; border-radius:10px;")
        sc = QGuiApplication.primaryScreen().availableGeometry()
        self.move((sc.width()-500)//2, (sc.height()-300)//2)
        lay = QVBoxLayout(self); lay.setContentsMargins(40, 36, 40, 24); lay.setSpacing(10)

        _logo_path = Path(__file__).resolve().parent / "ICONS" / "CODEEP2.png"
        t = QLabel()
        t.setAlignment(Qt.AlignCenter)
        if _logo_path.exists():
            _pix = QPixmap(str(_logo_path))
            # Escala para largura máxima de 420px mantendo proporção
            # Ajuste o valor conforme o tamanho da sua imagem
            if not _pix.isNull():
                _pix = _pix.scaledToWidth(100, Qt.SmoothTransformation)
            t.setPixmap(_pix)
        else:
            # Fallback: texto original caso a imagem não seja encontrada
            t.setText("CODEEP")
            t.setStyleSheet(
                "font-size:42pt;font-weight:bold;color:#2ECC71;letter-spacing:8px;")
        lay.addWidget(t)
        for txt, ss in [
            ("Deep Learning · Ligand Generation Pipeline",
             "font-size:11pt;color:#95A5A6;"),
            ("CODOC  ·  CODYN  ·  CODEEP  ·  CODRUG",
             "font-size:8pt;color:#3D5A4A;margin-top:2px;"),
        ]:
            l = QLabel(txt); l.setAlignment(Qt.AlignCenter); l.setStyleSheet(ss)
            lay.addWidget(l)

        lay.addSpacing(14)
        self.lbl_status = QLabel("Initializing...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size:9pt;color:#BDC3C7;")
        lay.addWidget(self.lbl_status)

        self.bar = QProgressBar(); self.bar.setRange(0,100); self.bar.setValue(0)
        self.bar.setTextVisible(False); self.bar.setFixedHeight(8)
        self.bar.setStyleSheet(
            "QProgressBar{background:#1C2D3A;border-radius:4px;border:none;}"
            "QProgressBar::chunk{background:#2ECC71;border-radius:4px;}")
        lay.addWidget(self.bar)

        v = QLabel("v1.0 beta  ·  © 2024–2026 CEB / UFPR  ·  GPL-3.0")
        v.setAlignment(Qt.AlignCenter)
        v.setStyleSheet("font-size:8pt;color:#3D5A4A;margin-top:10px;")
        lay.addWidget(v)

    def start(self):
        self._steps = [(20,"Checking Python..."),(45,"Checking PyQt5..."),
                       (65,"Checking PyTorch..."),(85,"Checking environment..."),
                       (100,"Ready!")]
        self._i = 0
        self._t = QTimer(self); self._t.timeout.connect(self._tick); self._t.start(320)

    def _tick(self):
        if self._i >= len(self._steps):
            self._t.stop(); QTimer.singleShot(250, self._done); return
        v, msg = self._steps[self._i]
        self.bar.setValue(v); self.lbl_status.setText(msg); self._i += 1

    def _done(self):
        self.ready_signal.emit(); self.accept()


# ==========================================================================================================================================
# ══════════════════════ JANELA PRINCIPAL ══════════════════════════════════
# ==========================================================================================================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CODEEP – Deep Learning Ligand Generation Pipeline")
        self.setStyleSheet(_SS_MAIN)
        sc = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(sc.width()*0.60), int(sc.height()*0.80))
        self.move((sc.width()-self.width())//2, (sc.height()-self.height())//2)

        self.dp_dir: str = str(Path(__file__).parent)
        self.job_dir: str = ""
        self._workers: dict = {}          # tag → worker ativo

        self._build_menu()
        self._build_hw_monitor()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_tabs()

    # ── Menu ──────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = self.menuBar()
        menu = mb.addMenu("Menu")
        # Itens de menu sem ícones, em inglês
        labels = [
            "Home",        "Step 1 – Job",
            "Step 2 – Collection", "Step 3 – Curation",
            "Step 4 – Exploration","Step 5 – Voxelization",
            "Step 6 – Training",   "Step 7 – Evaluation",
            "Step 8 – Generation",
        ]
        for i, label in enumerate(labels):
            act = QAction(label, self)
            act.triggered.connect(lambda _, idx=i: self.tabs.setCurrentIndex(idx))
            menu.addAction(act)
        menu.addSeparator()
        menu.addAction(QAction("Exit", self, triggered=self.close))

        help_menu = mb.addMenu("Help")
        help_menu.addAction(QAction("Install Requirements", self,
                                    triggered=self._show_requirements))
        help_menu.addSeparator()
        help_menu.addAction(QAction("About", self, triggered=self._show_about))

    # ── Monitor CPU + GPU ─────────────────────────────────────────────────
    def _build_hw_monitor(self):
        self._mon_widget = QWidget()
        self._mon_widget.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self._mon_widget)
        lay.setContentsMargins(0,0,4,0); lay.setSpacing(6)

        self.lbl_cpu = QLabel("CPU: --")
        self.lbl_cpu.setAlignment(Qt.AlignCenter)
        self.lbl_cpu.setMinimumWidth(220)
        self.lbl_cpu.setStyleSheet(self._mon_ss("#27AE60"))
        lay.addWidget(self.lbl_cpu)

        self.lbl_gpu = QLabel("GPU: --")
        self.lbl_gpu.setAlignment(Qt.AlignCenter)
        self.lbl_gpu.setMinimumWidth(160)
        self.lbl_gpu.setStyleSheet(self._mon_ss("#2980B9"))
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self.lbl_gpu.setVisible(self._has_nvidia)
        lay.addWidget(self.lbl_gpu)

        self.menuBar().setCornerWidget(self._mon_widget, Qt.TopRightCorner)
        self._hw_timer = QTimer(self); self._hw_timer.setInterval(1500)
        self._hw_timer.timeout.connect(self._update_hw); self._hw_timer.start()
        self._update_hw()

    @staticmethod
    def _mon_ss(c):
        return (f"QLabel{{padding:2px 8px;border:2px solid {c};"
                f"border-radius:6px;background:#1C3249;color:{c};"
                f"font-weight:bold;font-size:9pt;}}")

    @staticmethod
    def _uc(p):     # uso → cor
        return "#F57C6F" if p>80 else "#E67E22" if p>50 else "#27AE60"

    @staticmethod
    def _tc(t):     # temp → cor
        return "#F57C6F" if t>80 else "#E67E22" if t>60 else "#27AE60"

    def _sync_cur_executor_checks(self):
        sender = self.sender()

        if sender is self.chk_cur_thread and self.chk_cur_thread.isChecked():
            self.chk_cur_process.blockSignals(True)
            self.chk_cur_process.setChecked(False)
            self.chk_cur_process.blockSignals(False)

        elif sender is self.chk_cur_process and self.chk_cur_process.isChecked():
            self.chk_cur_thread.blockSignals(True)
            self.chk_cur_thread.setChecked(False)
            self.chk_cur_thread.blockSignals(False)

        # Nunca deixa os dois desligados
        if not self.chk_cur_thread.isChecked() and not self.chk_cur_process.isChecked():
            self.chk_cur_thread.blockSignals(True)
            self.chk_cur_thread.setChecked(True)
            self.chk_cur_thread.blockSignals(False)

    def _update_hw(self):
        if _PSUTIL_OK:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            if per_core:
                p_avg = sum(per_core) / len(per_core)
                p_max = max(per_core)
            else:
                p_avg = 0.0
                p_max = 0.0

            ts = ""
            try:
                for k in ("coretemp", "k10temp", "acpitz", "cpu_thermal"):
                    d = psutil.sensors_temperatures().get(k)
                    if d:
                        tc = self._tc(d[0].current)
                        ts = f" <span style='color:{tc};'>{d[0].current:.0f}°C</span>"
                        break
            except Exception:
                pass

            cc = self._uc(p_max)
            self.lbl_cpu.setText(
                f"<span style='color:{cc};'>CPU avg {p_avg:.0f}%</span>"
                f"  <span style='color:{cc};'>max {p_max:.0f}%</span>{ts}"
            )
            self.lbl_cpu.setTextFormat(Qt.RichText)
            self.lbl_cpu.setStyleSheet(self._mon_ss(cc))

        if not self._has_nvidia:
            return
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,"
                "memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2
            ).strip().splitlines()[0]
            gp, gt, mu, mt = [x.strip() for x in raw.split(",")]
            gp, gt, mu, mt = float(gp), float(gt), int(mu), int(mt)
            gc = self._uc(gp)
            tc = self._tc(gt)
            self.lbl_gpu.setText(
                f"<span style='color:{gc};'>GPU {gp:.0f}%</span>"
                f"  <span style='color:{tc};'>{gt:.0f}°C</span>"
                f"  <span style='color:#5A6A7E;'>{mu}/{mt}MB</span>"
            )
            self.lbl_gpu.setTextFormat(Qt.RichText)
            self.lbl_gpu.setStyleSheet(self._mon_ss(gc))
            self.lbl_gpu.setMinimumWidth(230)
            self.lbl_gpu.setVisible(True)
        except Exception:
            self.lbl_gpu.setText("GPU: --")

    # ── Abas — sem ícones, apenas nome curto em inglês ────────────────────
    def _build_tabs(self):
        self.tabs.addTab(self._tab_home(),   "Home")
        self.tabs.addTab(self._tab_step1(),  "Step 1")
        self.tabs.addTab(self._tab_step2(),  "Step 2")
        self.tabs.addTab(self._tab_step3(),  "Step 3")
        self.tabs.addTab(self._tab_step4(),  "Step 4")
        self.tabs.addTab(self._tab_step5(),  "Step 5")
        self.tabs.addTab(self._tab_step6(),  "Step 6")
        self.tabs.addTab(self._tab_step7(),  "Step 7")
        self.tabs.addTab(self._tab_step8(),  "Step 8")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 0 — HOME
    # ══════════════════════════════════════════════════════════════════════

    # ── Coleta de informações de hardware ─────────────────────────────────
    @staticmethod
    def _hw_cpu_info() -> list[tuple[str, str]]:
        """Retorna lista de (label, valor) com informações da CPU.

        Tenta múltiplas fontes em ordem de confiabilidade:
          1. lscpu  — mais completo (Model name, MHz, cache)
          2. /proc/cpuinfo — fallback para model e clock em MHz
          3. psutil — fallback para contagem de núcleos e RAM
        """
        import platform, subprocess as _sp, re as _re
        rows = []

        # ── Valores a coletar (preenchidos pelas fontes abaixo) ───────────
        model      = ""
        base_mhz   = 0.0
        max_mhz    = 0.0
        cores_phys = 0
        cores_log  = 0
        sockets    = 1
        l2_cache   = ""
        l3_cache   = ""

        # ── Fonte 1: lscpu ────────────────────────────────────────────────
        try:
            out = _sp.check_output(["lscpu"], text=True, stderr=_sp.DEVNULL)
            for line in out.splitlines():
                k, _, v = line.partition(":")
                k = k.strip(); v = v.strip()
                # Model name — campo principal
                if k == "Model name":
                    model = v
                # Sockets e cores por socket → núcleos físicos
                elif k == "Socket(s)":
                    try: sockets = int(v)
                    except Exception: pass
                elif k == "Core(s) per socket":
                    try: cores_phys = int(v) * sockets
                    except Exception: pass
                # Threads lógicas
                elif k == "CPU(s)":
                    try: cores_log = int(v)
                    except Exception: pass
                # Clock base: "CPU MHz" ou "CPU base MHz" ou "BogoMIPS"
                elif k in ("CPU MHz", "CPU base MHz", "Frequency (MHz)"):
                    try: base_mhz = float(v)
                    except Exception: pass
                # Clock máximo
                elif k == "CPU max MHz":
                    try: max_mhz = float(v)
                    except Exception: pass
                # Cache
                elif k == "L2 cache":
                    l2_cache = v
                elif k == "L3 cache":
                    l3_cache = v
        except Exception:
            pass

        # ── Fonte 2: /proc/cpuinfo ────────────────────────────────────────
        # Usado quando lscpu não trouxe modelo ou clock
        try:
            cpuinfo = open("/proc/cpuinfo").read()
            if not model:
                m = _re.search(r"model name\s*:\s*(.+)", cpuinfo)
                if m:
                    model = m.group(1).strip()
            if base_mhz == 0.0:
                m = _re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
                if m:
                    base_mhz = float(m.group(1))
            if cores_phys == 0:
                # cpu cores é por socket; multiplica por physical id count
                cores_m = _re.findall(r"cpu cores\s*:\s*(\d+)", cpuinfo)
                phys_ids = set(_re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
                if cores_m:
                    n_sockets = len(phys_ids) or 1
                    cores_phys = int(cores_m[0]) * n_sockets
        except Exception:
            pass

        # ── Fonte 3: psutil ───────────────────────────────────────────────
        if _PSUTIL_OK:
            import psutil as _ps
            if cores_phys == 0:
                cores_phys = _ps.cpu_count(logical=False) or 0
            if cores_log == 0:
                cores_log  = _ps.cpu_count(logical=True)  or 0
            # psutil.cpu_freq() retorna base e max em MHz
            try:
                freq = _ps.cpu_freq()
                if freq:
                    if base_mhz == 0.0 and freq.current:
                        base_mhz = freq.current
                    if max_mhz == 0.0 and freq.max:
                        max_mhz = freq.max
            except Exception:
                pass

        # ── Fallback final de modelo ───────────────────────────────────────
        if not model:
            model = platform.processor() or "N/A"

        # ── Monta a lista de linhas ────────────────────────────────────────
        rows.append(("Architecture", platform.machine()))
        rows.append(("Model",        model if model else "N/A"))
        rows.append(("Base clock",
                      f"{base_mhz/1000:.2f} GHz" if base_mhz else "N/A"))
        rows.append(("Max clock",
                      f"{max_mhz/1000:.2f} GHz"  if max_mhz  else "N/A"))
        rows.append(("Physical cores",  str(cores_phys) if cores_phys else "N/A"))
        rows.append(("Logical threads", str(cores_log)  if cores_log  else "N/A"))
        if l2_cache:
            rows.append(("L2 cache", l2_cache))
        if l3_cache:
            rows.append(("L3 cache", l3_cache))

        # RAM via psutil
        if _PSUTIL_OK:
            import psutil as _ps
            try:
                mem = _ps.virtual_memory()
                rows.append(("RAM total",
                              f"{mem.total / 1073741824:.1f} GB"))
                rows.append(("RAM available",
                              f"{mem.available / 1073741824:.1f} GB"))
            except Exception:
                pass

        return rows

    @staticmethod
    def _hw_gpu_info() -> list[tuple[str, str]]:
        """Retorna lista de (label, valor) com informações da GPU via nvidia-smi.

        Nota: 'cuda_version' não é um campo válido de --query-gpu do nvidia-smi.
        A versão CUDA máxima suportada pelo driver vem de 'nvidia-smi' direto,
        e a versão CUDA em uso pelo PyTorch vem de torch.version.cuda.
        """
        import subprocess as _sp, shutil as _sh
        rows = []
        if not _sh.which("nvidia-smi"):
            rows.append(("GPU", "nvidia-smi not found"))
            return rows
        try:
            # Campos válidos de --query-gpu (cuda_version NÃO é aceito aqui)
            fields = (
                "name,memory.total,"
                "clocks.max.sm,"
                "clocks.max.memory,power.limit"
            )
            raw = _sp.check_output(
                ["nvidia-smi",
                 f"--query-gpu={fields}",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=_sp.PIPE, timeout=4
            ).strip().splitlines()[0]
            parts = [p.strip() for p in raw.split(",")]
            labels = [
                "Model", "VRAM total (MiB)",
                "Max SM clock (MHz)", "Max mem clock (MHz)",
                "TDP (W)",
            ]
            for lbl, val in zip(labels, parts):
                rows.append((lbl, val))

            # Versão CUDA máxima suportada pelo driver (linha do cabeçalho)
            try:
                header = _sp.check_output(
                    ["nvidia-smi"], text=True,
                    stderr=_sp.DEVNULL, timeout=4
                )
                import re as _re
                m = _re.search(r"CUDA Version:\s*([\d.]+)", header)
                if m:
                    rows.append(("CUDA max (driver)", m.group(1)))
            except Exception:
                pass

            # Versão CUDA em uso pelo PyTorch
            if _TORCH_OK:
                import torch as _t
                rows.append(("CUDA (PyTorch)",
                              _t.version.cuda or "N/A"))

            # CUDA cores, TFLOPs — derivados da arquitetura via torch
            if _TORCH_OK:
                import torch as _t
                if _t.cuda.is_available():
                    props = _t.cuda.get_device_properties(0)
                    mp    = props.multi_processor_count
                    rows.append(("Multiprocessors (SM)", str(mp)))
                    # Mapeamento compute capability → CUDA cores por SM
                    major = props.major
                    cores_per_sm = {
                        9: 128,   # Hopper (H100)
                        8: 128,   # Ampere RTX 30/40 (A100 usa 64, mas é raro)
                        7: 64,    # Volta / Turing
                        6: 128,   # Pascal (GP102/104; GP100 usa 64)
                        5: 128,   # Maxwell
                    }.get(major, 64)
                    total_cores = mp * cores_per_sm
                    rows.append(("CUDA cores (est.)", str(total_cores)))
                    # FP32 TFLOPs: cores × 2 operações/ciclo × clock_Hz / 1e12
                    try:
                        sm_vals = [v for l, v in rows if "Max SM clock" in l]
                        if sm_vals:
                            tflops = (total_cores * 2 * float(sm_vals[0])) / 1e6
                            rows.append(("FP32 perf. (TFLOPs, est.)",
                                          f"{tflops:.1f}"))
                    except Exception:
                        pass
                    rows.append(("Compute capability",
                                  f"{props.major}.{props.minor}"))
        except Exception as e:
            rows.append(("Error", str(e)))
        return rows

    @staticmethod
    def _sw_versions() -> list[tuple[str, str]]:
        """Retorna lista de (biblioteca, versão) para o quadro Software Specs."""
        import importlib, subprocess as _sp, shutil as _sh
        rows = []

        # Python
        rows.append(("Python", sys.version.split()[0]))

        # Bibliotecas — tenta importar e ler __version__
        libs = [
            ("PyQt5",           "PyQt5.QtCore",    "PYQT_VERSION_STR"),
            ("PyTorch",         "torch",            "__version__"),
            ("torch-geometric", "torch_geometric",  "__version__"),
            ("NumPy",           "numpy",            "__version__"),
            ("Pandas",          "pandas",           "__version__"),
            ("Matplotlib",      "matplotlib",       "__version__"),
            ("Seaborn",         "seaborn",          "__version__"),
            ("scikit-learn",    "sklearn",          "__version__"),
            ("SciPy",           "scipy",            "__version__"),
            ("RDKit",           "rdkit",            "__version__"),
            ("BioPython",       "Bio",              "__version__"),
            ("PySCF",           "pyscf",            "__version__"),
            ("CuPy",            "cupy",             "__version__"),
            ("psutil",          "psutil",           "__version__"),
            ("tqdm",            "tqdm",             "__version__"),
        ]
        for display, mod_name, attr in libs:
            try:
                mod = importlib.import_module(mod_name)
                ver = getattr(mod, attr, None) or "installed"
                rows.append((display, ver))
            except ImportError:
                rows.append((display, "not installed"))

        # CUDA via torch
        if _TORCH_OK:
            import torch as _t
            rows.append(("CUDA (torch)",
                          _t.version.cuda or "N/A"))

        # Driver NVIDIA
        if _sh.which("nvidia-smi"):
            try:
                drv = _sp.check_output(
                    ["nvidia-smi","--query-gpu=driver_version",
                     "--format=csv,noheader"],
                    text=True, stderr=_sp.DEVNULL, timeout=3
                ).strip().splitlines()[0].strip()
                rows.append(("NVIDIA driver", drv))
            except Exception:
                rows.append(("NVIDIA driver", "N/A"))
        else:
            rows.append(("NVIDIA driver", "nvidia-smi not found"))

        return rows

    # ── Construção da aba Home ─────────────────────────────────────────────
    def _tab_home(self):
        w = QWidget()
        # Scroll para acomodar os quadros de hardware
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(36,32,36,24); lay.setSpacing(16)
        scroll.setWidget(inner)

        # ── Cabeçalho ─────────────────────────────────────────────────────
        _logo_path = Path(__file__).resolve().parent / "ICONS" / "CODEEP2.png"
        hdr = QLabel()
        hdr.setAlignment(Qt.AlignCenter)
        if _logo_path.exists():
            _pix = QPixmap(str(_logo_path))
            # Escala para largura máxima de 150px mantendo proporção
            # Ajuste o valor conforme o tamanho da sua imagem
            if not _pix.isNull():
                _pix = _pix.scaledToWidth(150, Qt.SmoothTransformation)
            hdr.setPixmap(_pix)
        else:
            # Fallback: texto original caso a imagem não seja encontrada
            hdr.setText("CODEEP")
            hdr.setStyleSheet(
                "font-size:42pt;font-weight:bold;color:#2ECC71;letter-spacing:8px;")
        lay.addWidget(hdr)
        for txt, ss in [
            ("Deep learning applied to the Ligands Generation Pipeline",
             "font-size:13pt;color:#95A5A6;"),
        ]:
            l = QLabel(txt); l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet(ss); lay.addWidget(l)

        lay.addWidget(_mk_separator())

        # ── Linha dupla: Hardware Specs CPU | Hardware Specs GPU ───────────
        hw_row = QHBoxLayout(); hw_row.setSpacing(14)

        # Quadro CPU
        cpu_grp = QGroupBox("Hardware Specs — CPU")
        cpu_form = QFormLayout(cpu_grp)
        cpu_form.setHorizontalSpacing(16); cpu_form.setVerticalSpacing(4)
        for label, val in self._hw_cpu_info():
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            lbl_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            cpu_form.addRow(lbl_k, lbl_v)
        hw_row.addWidget(cpu_grp, 1)

        # Quadro GPU
        gpu_grp = QGroupBox("Hardware Specs — GPU")
        gpu_form = QFormLayout(gpu_grp)
        gpu_form.setHorizontalSpacing(16); gpu_form.setVerticalSpacing(4)
        for label, val in self._hw_gpu_info():
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            lbl_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            gpu_form.addRow(lbl_k, lbl_v)
        hw_row.addWidget(gpu_grp, 1)

        lay.addLayout(hw_row)

        # ── Software Specs ─────────────────────────────────────────────────
        sw_grp = QGroupBox("Software Specs")
        sw_grid = QGridLayout(sw_grp)
        sw_grid.setHorizontalSpacing(24); sw_grid.setVerticalSpacing(4)
        sw_items = self._sw_versions()
        # Distribui em 3 colunas para não ficar muito alto
        cols = 3
        for i, (lib, ver) in enumerate(sw_items):
            row, col = divmod(i, cols)
            ok = ver != "not installed"
            lbl_lib = QLabel(lib)
            lbl_lib.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_ver = QLabel(ver)
            lbl_ver.setStyleSheet(
                f"font-weight:bold;font-size:9pt;"
                f"color:{'#27AE60' if ok else '#E74C3C'};")
            lbl_ver.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sw_grid.addWidget(lbl_lib, row, col*2)
            sw_grid.addWidget(lbl_ver, row, col*2+1)
        lay.addWidget(sw_grp)

        # ── Pipeline — Steps ───────────────────────────────────────────────
        pipe_grp = QGroupBox("Pipeline — Steps")
        pipe_lay = QGridLayout(pipe_grp)
        pipe_lay.setHorizontalSpacing(20); pipe_lay.setVerticalSpacing(4)
        steps = [
            ("Step 1 – Job",          "Working directory setup, job naming and path validation"),
            ("Step 2 – Collection",   "PDB download (public datasets + user-defined targets)"),
            ("Step 3 – Curation",     "Structure validation, binding site extraction"),
            ("Step 4 – Exploration",  "Dataset statistics and visualization"),
            ("Step 5 – Voxelization", "3D tensors + molecular graphs (PyG)"),
            ("Step 6 – Training",     "CVAE: 3D-CNN encoder + GNN decoder"),
            ("Step 7 – Evaluation",   "Diagnostics, metrics and training curves"),
            ("Step 8 – Generation",   "Ligand candidates for a target protein"),
        ]
        for i, (name, desc) in enumerate(steps):
            name_lbl = QLabel(f"<b>{name}</b>")
            name_lbl.setStyleSheet("color:#2ECC71;font-size:9pt;")
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            pipe_lay.addWidget(name_lbl, i, 0)
            pipe_lay.addWidget(desc_lbl, i, 1)
        lay.addWidget(pipe_grp)

        lay.addWidget(_mk_separator())

        # ── Botões de ação — sem ícones ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_start = _mk_btn("Start", _SS_BTN_PRIMARY, 160)
        btn_start.setFixedHeight(38)
        btn_start.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        btn_req = _mk_btn("Install Requirements", _SS_BTN_PRIMARY, 190)
        btn_req.setFixedHeight(38)
        btn_req.clicked.connect(self._show_requirements)
        if not _REQ_MODULE_OK:
            btn_req.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(btn_start)
        btn_row.addSpacing(14)
        btn_row.addWidget(btn_req)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        ver = QLabel("v1.0 beta  ·  © 2024–2026 CEB/UFPR  ·  GPL-3.0")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("font-size:8pt;color:#3D5A4A;margin-top:6px;")
        lay.addWidget(ver)

        # Empacota scroll na aba
        outer = QVBoxLayout(w); outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — STEP 1: CONFIGURAÇÃO DO JOB
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step1(self):
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(30,16,30,16); lay.setSpacing(12)
        scroll.setWidget(inner)

        lay.addWidget(_mk_title("Step 1 — Job Configuration"))
        lay.addWidget(_mk_separator())

        # ── Diretório raiz do CODEEP ───────────────────────────────────────
        grp_dir = QGroupBox("CODEEP Root Directory")
        dl = QHBoxLayout(grp_dir)
        self.ed_root = QLineEdit(self.dp_dir); self.ed_root.setReadOnly(True)
        dl.addWidget(self.ed_root)
        btn_br = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_br.clicked.connect(self._browse_root)
        dl.addWidget(btn_br)
        lay.addWidget(grp_dir)

        # ── Identificação do Job ───────────────────────────────────────────
        grp_job = QGroupBox("Job Identification")
        jf = QFormLayout(grp_job); jf.setHorizontalSpacing(20)
        self.ed_job_name = QLineEdit()
        self.ed_job_name.setPlaceholderText("e.g.: KinaseInhibitors_run01")
        self.ed_job_name.setMaximumWidth(340)        
        self.de_job_date = QDateEdit()
        self.de_job_date.setCalendarPopup(True)
        self.de_job_date.setDate(QDate.currentDate())
        self.de_job_date.setDisplayFormat("yyyy-MM-dd")
        self.de_job_date.setMaximumWidth(160)
        self.de_job_date.setToolTip(
            "Date used to compose the Job ID.\n"
            "Change this to resume a job started on a previous day\n"
            "without renaming the folder manually."
        )
        self.lbl_job_id = QLabel("—")
        self.lbl_job_id.setStyleSheet("font-weight:bold;color:#2ECC71;")
        self.ed_job_name.textChanged.connect(self._update_job_id)
        self.de_job_date.dateChanged.connect(self._update_job_id)
        jf.addRow("Job Name:", self.ed_job_name)
        jf.addRow("Date:",     self.de_job_date)
        jf.addRow("Job ID:",   self.lbl_job_id)
        lay.addWidget(grp_job)

        # ── Termos de busca para Fases 2 e 3 ──────────────────────────────
        grp_search = QGroupBox("Phase 2 / 3 — Target Search Terms  (one per line)")
        sl = QVBoxLayout(grp_search); sl.setSpacing(6)

        sl.addWidget(_mk_section("Organism terms  (Phase 2 — RCSB taxonomy search)"))
        self.te_org_terms = QTextEdit()
        self.te_org_terms.setPlaceholderText(
            "e.g.:\nHomo sapiens\nRattus norvegicus\nMus musculus")
        self.te_org_terms.setFixedHeight(72)
        sl.addWidget(self.te_org_terms)

        sl.addWidget(_mk_section("Entity description terms  (Phase 2 — RCSB keyword search)"))
        self.te_desc_terms = QTextEdit()
        self.te_desc_terms.setPlaceholderText(
            "e.g.:\nprotein kinase\ntyrosine kinase\nATP-binding")
        self.te_desc_terms.setFixedHeight(72)
        sl.addWidget(self.te_desc_terms)

        sl.addWidget(_mk_section("Phase 3 — Specific PDB IDs  (comma or newline separated)"))
        self.te_phase3_ids = QTextEdit()
        self.te_phase3_ids.setPlaceholderText(
            "e.g.:\n1ATP, 2SRC, 3EKL\n4HJB\n5J5X")
        self.te_phase3_ids.setFixedHeight(60)
        sl.addWidget(self.te_phase3_ids)

        lay.addWidget(grp_search)

        # ── Detected Paths ─────────────────────────────────────────────────
        grp_paths = QGroupBox("Detected Paths")
        pf = QFormLayout(grp_paths); pf.setHorizontalSpacing(16)
        pf.setVerticalSpacing(3)

        # Caminhos globais (Fase 1 — compartilhados entre todos os jobs)
        self._path_labels = {}
        global_paths = [
            ("global_phase1_raw", "data/raw/pdb/ (Phase 1)", os.path.join("data", "raw", "pdb")),
            ("global_phase1_curated", "data/curated/ (Phase 1)", os.path.join("data", "curated")),
            ("global_phase1_tensors", "data/tensors/ (Phase 1)", os.path.join("data", "tensors")),
            ("global_phase1_ckpt", "checkpoints/phase1/", os.path.join("checkpoints", "phase1")),
        ]
        _sep = QLabel("<b>Global (shared — Phase 1)</b>")
        _sep.setStyleSheet("color:#2ECC71;font-size:9pt;margin-top:4px;")
        pf.addRow(_sep)
        for key, label, sub in global_paths:
            lbl = QLabel("—"); lbl.setStyleSheet("font-size:9pt;color:#C9D1D9;")
            lbl.setTextFormat(Qt.RichText)
            self._path_labels[key] = (lbl, sub, "global")
            pf.addRow(label, lbl)

        # Caminhos do job (Fases 2/3 — por job_id)
        _sep2 = QLabel("<b>Job-specific (Phase 2 / 3)</b>")
        _sep2.setStyleSheet("color:#5DADE2;font-size:9pt;margin-top:6px;")
        pf.addRow(_sep2)
        job_paths = [
            ("job_phase2_raw", "data/raw/pdb/phase2/", os.path.join("data", "raw", "pdb", "phase2")),
            ("job_phase3_raw", "data/raw/pdb/phase3/", os.path.join("data", "raw", "pdb", "phase3")),
            ("job_phase2_curated", "data/curated/phase2/", os.path.join("data", "curated", "phase2")),
            ("job_phase3_curated", "data/curated/phase3/", os.path.join("data", "curated", "phase3")),
            ("job_phase2_tensors", "data/tensors/phase2/", os.path.join("data", "tensors", "phase2")),
            ("job_phase3_tensors", "data/tensors/phase3/", os.path.join("data", "tensors", "phase3")),
            ("job_phase2_ckpt", "checkpoints/phase2/", os.path.join("checkpoints", "phase2")),
            ("job_phase3_ckpt", "checkpoints/phase3/", os.path.join("checkpoints", "phase3")),
        ]
        for key, label, sub in job_paths:
            lbl = QLabel("—"); lbl.setStyleSheet("font-size:9pt;color:#C9D1D9;")
            lbl.setTextFormat(Qt.RichText)
            self._path_labels[key] = (lbl, sub, "job")
            pf.addRow(label, lbl)

        lay.addWidget(grp_paths)

        # ── Botão e status ─────────────────────────────────────────────────
        self.btn_job_run = _mk_btn("Create / Load Job", _SS_BTN_PRIMARY, 230)
        self.btn_job_run.setFixedHeight(38)
        self.btn_job_run.clicked.connect(self._job_run)
        lay.addWidget(self.btn_job_run, alignment=Qt.AlignCenter)

        self.lbl_job_status = QLabel("")
        self.lbl_job_status.setAlignment(Qt.AlignCenter)
        self.lbl_job_status.setStyleSheet("font-size:9pt;color:#2ECC71;")
        lay.addWidget(self.lbl_job_status)

        lay.addStretch()

        # Empacota scroll
        outer = QVBoxLayout(w); outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)
        # Nav row fora do scroll
        nav = QWidget(); nl = QVBoxLayout(nav); nl.setContentsMargins(8,4,8,8)
        nl.addLayout(_mk_nav_row(lambda: self.tabs.setCurrentIndex(0),
                                  lambda: self.tabs.setCurrentIndex(2)))
        outer.addWidget(nav)
        return w

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select the CODEEP root folder", self.dp_dir)
        if d:
            self.dp_dir = d; self.ed_root.setText(d)
            self._refresh_paths()

    def _update_job_id(self):
        name = self.ed_job_name.text().strip()
        date = self.de_job_date.date().toString("yyyy-MM-dd")
        self.lbl_job_id.setText(f"{date}_{name}" if name else "—")

    def _refresh_paths(self):
        """Atualiza os labels de Detected Paths para global e job-specific."""
        root = self.ed_root.text().strip()
        if not root:
            return

        def _fmt(p: str) -> str:
            """Formata o caminho com ícone e contagem de arquivos relevantes."""
            from pathlib import Path as _P

            path = _P(p)
            exists = path.exists()
            color = "#27AE60" if exists else "#E74C3C"
            icon = "✔" if exists else "✘"

            count = ""
            if exists and path.is_dir():
                pdbs = len(list(path.rglob("*.pdb")))
                sdfs = len(list(path.rglob("*.sdf")))
                pts = len(list(path.rglob("*.pt")))
                npzs = len(list(path.rglob("*.npz")))

                parts = []
                if pdbs:
                    parts.append(f"{pdbs} PDB")
                if sdfs:
                    parts.append(f"{sdfs} SDF")
                if pts:
                    parts.append(f"{pts} .pt")
                if npzs:
                    parts.append(f"{npzs} .npz")

                # Para Phase 1 raw PDB, mostra também o detalhamento por subpasta
                if path.parts[-3:] == ("data", "raw", "pdb"):
                    sub_parts = []
                    for child in sorted([d for d in path.iterdir() if d.is_dir()]):
                        n_child = len(list(child.glob("*.pdb")))
                        if n_child:
                            sub_parts.append(f"{child.name}: {n_child}")
                    if sub_parts:
                        parts.append(" | ".join(sub_parts))

                if parts:
                    count = f"  <span style='color:#6E8CA8;font-size:8pt;'>({'; '.join(parts)})</span>"

            return f'<span style="color:{color};">{icon}  {p}</span>{count}'

        job_id  = self.lbl_job_id.text().strip()
        has_job = bool(job_id and job_id != "—")

        for key, (lbl, sub, kind) in self._path_labels.items():
            if kind == "global":
                full = os.path.join(root, sub)
            else:
                # Caminhos job-specific ficam em JOBS/<job_id>/
                if has_job:
                    full = os.path.join(root, "JOBS", job_id, sub)
                else:
                    lbl.setText('<span style="color:#BDC3C7;">— (set job name first) —</span>')
                    continue
            lbl.setText(_fmt(full))

    def _job_run(self):
        root = self.ed_root.text().strip()
        name = self.ed_job_name.text().strip()
        if not root or not name:
            QMessageBox.warning(
                self, "Warning",
                "Please fill in the CODEEP root directory and a job name."); return

        date     = self.de_job_date.date().toString("yyyy-MM-dd")
        job_id   = f"{date}_{name}"
        job_dir  = os.path.join(root, "JOBS", job_id)

        # Estrutura global da Fase 1
        for sub in [
            os.path.join("data", "raw", "pdb"),
            os.path.join("data", "raw", "sdf"),
            os.path.join("data", "curated"),
            os.path.join("data", "tensors"),
            os.path.join("data", "figures"),
            os.path.join("data", "logs"),
            os.path.join("checkpoints", "phase1"),
        ]:
            os.makedirs(os.path.join(root, sub), exist_ok=True)

        # Estrutura do job para Fases 2 e 3
        for sub in [
            os.path.join("data", "raw", "pdb", "phase2"),
            os.path.join("data", "raw", "pdb", "phase3"),
            os.path.join("data", "raw", "sdf", "phase2"),
            os.path.join("data", "raw", "sdf", "phase3"),
            os.path.join("data", "curated", "phase2"),
            os.path.join("data", "curated", "phase3"),
            os.path.join("data", "tensors", "phase2"),
            os.path.join("data", "tensors", "phase3"),
            os.path.join("data", "figures", "phase2"),
            os.path.join("data", "figures", "phase3"),
            os.path.join("data", "logs", "phase2"),
            os.path.join("data", "logs", "phase3"),
            os.path.join("checkpoints", "phase2"),
            os.path.join("checkpoints", "phase3"),
            "results",
        ]:
            os.makedirs(os.path.join(job_dir, sub), exist_ok=True)

        # Salva configuração do job em JSON (termos de busca + metadados)
        job_config = {
            "job_id":          job_id,
            "job_name":        name,
            "created":         datetime.now().isoformat(),
            "root_dir":        root,
            "job_dir":         job_dir,
            "organism_terms":  self._parse_text_list(self.te_org_terms.toPlainText()),
            "desc_terms":      self._parse_text_list(self.te_desc_terms.toPlainText()),
            "phase3_pdb_ids":  self._parse_csv_ids(self.te_phase3_ids.toPlainText()),
        }
        cfg_path = os.path.join(job_dir, "job_config.json")
        with open(cfg_path, "w") as f:
            json.dump(job_config, f, indent=2)

        self.job_dir = job_dir
        self.lbl_job_id.setText(job_id)
        self.lbl_job_status.setText(f"✅  Job ready:  {job_dir}")
        self._refresh_paths()
        self._inject_root(root, job_dir)

    @staticmethod
    def _parse_text_list(text: str) -> list:
        """Converte texto multilinha em lista de strings não-vazias."""
        return [l.strip() for l in text.splitlines() if l.strip()]

    @staticmethod
    def _parse_csv_ids(text: str) -> list:
        """Converte texto com IDs separados por vírgulas ou quebras em lista."""
        import re as _re
        tokens = _re.split(r"[,\n\r\s]+", text)
        return [t.strip().upper() for t in tokens if t.strip()]

    def _inject_root(self, root: str, job_dir: str = ""):
        """Injeta caminhos corretos nos módulos do pipeline.

        Fase 1 (coleta geral, curadoria, exploração, voxelização):
            BASE_DIR, RAW_DIR, CUR_DIR, TENSOR_DIR, FIG_DIR → <root>/data/...

        Fases 2/3 (coleta específica, curadoria job, treino, avaliação, geração):
            Os módulos de coleta recebem job_dir diretamente como argumento,
            não via variável de módulo, para manter a separação limpa.
            checkpoints e logs → <job_dir>/...
        """
        from pathlib import Path as _P
        import importlib

        # Garante que MODULES/ e raiz estão no sys.path
        for p in [os.path.join(os.path.dirname(__file__), "MODULES"),
                  os.path.dirname(__file__)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        _root_p = _P(root)
        _job_p  = _P(job_dir) if job_dir else _root_p

        for mod_name in ["pipeline_collect","pipeline_curate","pipeline_explore",
                         "pipeline_voxelize","pipeline_train",
                         "pipeline_evaluate","pipeline_generate"]:
            try:
                mod = importlib.import_module(f"MODULES.{mod_name}")
                mod._CODEEP_ROOT    = root
                mod._CODEEP_JOB_DIR = str(_job_p)

                if hasattr(mod, "PROJECT_DIR"):
                    mod.PROJECT_DIR = _job_p
                if hasattr(mod, "BASE_DIR"):
                    mod.BASE_DIR    = _root_p / "data"
                if hasattr(mod, "RAW_DIR"):
                    mod.RAW_DIR     = _root_p / "data" / "raw"
                if hasattr(mod, "CUR_DIR"):
                    mod.CUR_DIR     = _root_p / "data" / "curated"
                if hasattr(mod, "TENSOR_DIR"):
                    mod.TENSOR_DIR  = _root_p / "data" / "tensors"
                if hasattr(mod, "FIG_DIR"):
                    mod.FIG_DIR     = _root_p / "data" / "figures"
                if hasattr(mod, "LOG_DIR"):
                    mod.LOG_DIR     = _root_p / "data" / "logs"

            except Exception as e:
                print(f"[CODEEP] warning injecting paths into {mod_name}: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # HELPER GENÉRICO: constrói aba de etapa com log + progresso + botões
    # ══════════════════════════════════════════════════════════════════════
    def _build_pipeline_tab(self, title, params_widget, run_fn_getter,
                            worker_key, prev_idx, next_idx,
                            log_height=280):
        """Constrói o layout padrão de uma aba de pipeline.

        params_widget : QWidget com os controles específicos da etapa
        run_fn_getter : callable() → (fn, args, kwargs) a executar
        """
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(30,20,30,20); lay.setSpacing(12)
        lay.addWidget(_mk_title(title))
        lay.addWidget(_mk_separator())

        # Painel de parâmetros (scroll)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(params_widget); scroll.setMaximumHeight(420)
        lay.addWidget(scroll)

        # Log de execução em tempo real
        lay.addWidget(_mk_section("Execution Log"))
        log_box = _mk_log_box(log_height)
        lay.addWidget(log_box)

        # Barra de progresso
        prog = _mk_progress(); lay.addWidget(prog)

        # Botões de execução e parada
        btn_row = QHBoxLayout()
        btn_run  = _mk_btn("▶  Run",  _SS_BTN_PRIMARY, 150)
        btn_stop = _mk_btn("⏹  Stop", _SS_BTN_DANGER,  100)
        btn_stop.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(btn_run); btn_row.addWidget(btn_stop)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addLayout(_mk_nav_row(
            lambda: self.tabs.setCurrentIndex(prev_idx),
            lambda: self.tabs.setCurrentIndex(next_idx),
        ))

        def _run():
            if not self.ed_root.text().strip():
                QMessageBox.warning(w,"Warning","Please configure and start a Job (Step 1) first.")
                return
            try:
                fn, args, kwargs = run_fn_getter()
            except Exception as e:
                QMessageBox.critical(w, "Configuration Error", str(e)); return
            log_box.clear(); prog.setValue(0)
            btn_run.setEnabled(False); btn_stop.setEnabled(True)
            worker = PipelineWorker(fn, args, kwargs)
            _connect_worker(worker, log_box, prog, btn_run, btn_stop,
                            lambda ok: None)
            self._workers[worker_key] = worker
            worker.start()

        btn_run.clicked.connect(_run)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — STEP 2: COLETA
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step2(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(["lp-pdbbind", "rcsb-wide", "plinder", "moad"])

        self.cb_phase = QComboBox()
        self.cb_phase.addItems(["Phase 1", "Phase 2", "Phase 3", "All Phases"])
        self.cb_phase.setToolTip(
            "Phase 1 — Download the external dataset. (lp-PDBbind / RCSB / PLINDER / MOAD)\n"
            "Phase 2 — Search RCSB by organism/description (terms defined in Step 1)\n"
            "Phase 3 — Download specific PDBs by ID (IDs defined in Step 1)\n"
            "All Phases — Execute all three phases in sequence"
        )

        self.sp_phase1_max = QSpinBox()
        self.sp_phase1_max.setRange(1000, 500000)
        self.sp_phase1_max.setValue(50000)

        self.ed_moad_dir = QLineEdit()
        self.ed_moad_dir.setPlaceholderText("Required only for MOAD dataset")
        btn_moad = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_moad.clicked.connect(self._browse_moad_dir)
        moad_row = QWidget(); ml = QHBoxLayout(moad_row); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.ed_moad_dir); ml.addWidget(btn_moad)

        self.chk_status = QCheckBox("Only show collected data status")

        pf.addRow("Phase 1 Dataset:", self.cb_dataset)
        pf.addRow("Collection Phase:", self.cb_phase)
        pf.addRow("Max Phase 1 entries:", self.sp_phase1_max)
        pf.addRow("MOAD Dir:",        moad_row)
        pf.addRow("",                 self.chk_status)

        def run_fn():
            import MODULES.pipeline_collect as m
            if self.chk_status.isChecked():
                return m.show_status, (), {
                    "job_dir": self.job_dir or None
                }

            root        = self.ed_root.text().strip()
            job_dir     = self.job_dir or root
            phase       = self.cb_phase.currentText()
            dataset     = self.cb_dataset.currentText()
            moad_dir    = self.ed_moad_dir.text().strip()
            phase1_max = self.sp_phase1_max.value()
            org_terms   = self._parse_text_list(self.te_org_terms.toPlainText())
            desc_terms  = self._parse_text_list(self.te_desc_terms.toPlainText())
            p3_ids      = self._parse_csv_ids(self.te_phase3_ids.toPlainText())

            wants_phase1 = phase in ("Phase 1", "All Phases")
            wants_phase2 = phase in ("Phase 2", "All Phases")
            wants_phase3 = phase in ("Phase 3", "All Phases")

            # Validação Phase 2 — termos obrigatórios
            if wants_phase2 and not org_terms and not desc_terms:
                resp = QMessageBox.question(
                    params,
                    "Phase 2 — Search terms not configured",
                    "Phase 2 requires organism or description search terms "
                    "(configured in Step 1), but both fields are empty.\n\n"
                    "Without search terms, Phase 2 cannot retrieve target-specific "
                    "structures from RCSB.\n\n"
                    "Would you like to skip Phase 2 and continue with the other "
                    "selected phases?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    raise ValueError(
                        "Cancelled. Fill in organism or description terms in Step 1 "
                        "and try again.")
                wants_phase2 = False

            # Validação Phase 3 — IDs obrigatórios
            if wants_phase3 and not p3_ids:
                resp = QMessageBox.question(
                    params,
                    "Phase 3 — PDB IDs not configured",
                    "Phase 3 requires a list of specific PDB IDs "
                    "(configured in Step 1), but the field is empty.\n\n"
                    "Without PDB IDs, Phase 3 cannot download the target structures.\n\n"
                    "Would you like to skip Phase 3 and continue with the other "
                    "selected phases?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    raise ValueError(
                        "Cancelled. Fill in the PDB IDs in Step 1 and try again.")
                wants_phase3 = False

            # Verificação final — pelo menos uma fase ativa após validações
            if not any([wants_phase1, wants_phase2, wants_phase3]):
                raise ValueError(
                    "No collection phase active after validation. "
                    "Please check your configuration in Step 1 and try again.")

            return _pipeline_collect, (), dict(
                dataset      = dataset,
                moad_dir     = moad_dir,
                phase1_max   = phase1_max,
                wants_phase1 = wants_phase1,
                wants_phase2 = wants_phase2,
                wants_phase3 = wants_phase3,
                job_dir      = job_dir,
                org_terms    = org_terms,
                desc_terms   = desc_terms,
                p3_ids       = p3_ids,
            )

        return self._build_pipeline_tab(
            "Step 2 — Data Collection (PDB / Datasets)",
            params, run_fn, "collect", 1, 3
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — STEP 3: CURADORIA
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step3(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.sp_cur_workers = QSpinBox()
        self.sp_cur_workers.setRange(0,64); self.sp_cur_workers.setValue(0)
        self.sp_cur_workers.setToolTip("0 = detect automatically")
        self.sp_cutoff = QDoubleSpinBox()
        self.sp_cutoff.setRange(3.0,12.0); self.sp_cutoff.setValue(6.0)
        self.sp_cutoff.setSingleStep(0.5)
        self.sp_min_site = QSpinBox()
        self.sp_min_site.setRange(1,50); self.sp_min_site.setValue(5)
        self.sp_max_mw = QDoubleSpinBox()
        self.sp_max_mw.setRange(100,2000); self.sp_max_mw.setValue(800)
        self.sp_min_heavy = QSpinBox()
        self.sp_min_heavy.setRange(1,30); self.sp_min_heavy.setValue(7)
        self.cb_cur_phase = QComboBox()
        self.cb_cur_phase.addItems(["1", "2", "3"])
        self.chk_cur_thread = QCheckBox("ThreadPoolExecutor")
        self.chk_cur_process = QCheckBox("ProcessPoolExecutor")

        self.chk_cur_thread.setChecked(True)
        self.chk_cur_process.setChecked(False)

        self.chk_cur_thread.toggled.connect(self._sync_cur_executor_checks)
        self.chk_cur_process.toggled.connect(self._sync_cur_executor_checks)
        
        pf.addRow("Phase:", self.cb_cur_phase)
        pf.addRow("Workers (0=auto):",      self.sp_cur_workers)
        pf.addRow("Site cutoff (Å):",       self.sp_cutoff)
        pf.addRow("Min. site residues:",     self.sp_min_site)
        pf.addRow("Max. ligand MW (Da):",    self.sp_max_mw)
        pf.addRow("Min. heavy atoms:",       self.sp_min_heavy)
        pf.addRow("", self.chk_cur_thread)
        pf.addRow("", self.chk_cur_process)          

        def run_fn():
            import MODULES.pipeline_curate as m
            config = dict(m.CURATE_CONFIG)
            config["binding_site_cutoff_A"] = self.sp_cutoff.value()
            config["min_site_residues"] = self.sp_min_site.value()
            config["max_ligand_mw"] = self.sp_max_mw.value()
            config["min_heavy_atoms"] = self.sp_min_heavy.value()

            root = Path(self.ed_root.text().strip())
            job = Path(self.job_dir)
            phase = self.cb_cur_phase.currentText()
            workers = self.sp_cur_workers.value() or None

            if phase == "1":
                input_pdb_dir = root / "data" / "raw" / "pdb"
                output_curated_dir = root / "data" / "curated"
                output_metadata_csv = root / "data" / "curated_metadata.csv"
                output_report_csv = root / "data" / "quality_report.csv"
            elif phase == "2":
                input_pdb_dir = job / "data" / "raw" / "pdb" / "phase2"
                output_curated_dir = job / "data" / "curated" / "phase2"
                output_metadata_csv = job / "data" / "curated" / "phase2_metadata.csv"
                output_report_csv = job / "data" / "curated" / "phase2_quality_report.csv"
            else:
                input_pdb_dir = job / "data" / "raw" / "pdb" / "phase3"
                output_curated_dir = job / "data" / "curated" / "phase3"
                output_metadata_csv = job / "data" / "curated" / "phase3_metadata.csv"
                output_report_csv = job / "data" / "curated" / "phase3_quality_report.csv"
            executor_mode = "process" if self.chk_cur_process.isChecked() else "thread"
            return m.run_curation, (), {
                "config": config,
                "input_pdb_dir": input_pdb_dir,
                "output_curated_dir": output_curated_dir,
                "output_metadata_csv": output_metadata_csv,
                "output_report_csv": output_report_csv,
                "workers": workers,
                "executor_mode": executor_mode,
            }
        
        return self._build_pipeline_tab(
            "Step 3 — Structure Validation and Curation",
            params, run_fn, "curate", 2, 4
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — STEP 4: EXPLORAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step4(self):
        params = QWidget(); pf = QVBoxLayout(params)
        lbl = QLabel("Generates statistics and the exploration dashboard from the curated dataset.\n"
                      "Saves summary_dashboard.png and .svg to data/figures/.")
        lbl.setStyleSheet("color:#6E8CA8;font-size:9pt;"); lbl.setWordWrap(True)
        pf.addWidget(lbl)

        # Botão para visualizar o dashboard gerado
        self.btn_view_dash = _mk_btn("View Dashboard", _SS_BTN_PRIMARY, 160)
        self.btn_view_dash.clicked.connect(self._view_explore_dashboard)
        
        pf.addWidget(QLabel("Phase:"))
        self.cb_explore_phase = QComboBox()
        self.cb_explore_phase.addItems(["1", "2", "3"])        
        pf.addWidget(self.cb_explore_phase)
        
        pf.addWidget(self.btn_view_dash, Qt.AlignCenter)
        pf.addStretch()        

        def run_fn():
            import MODULES.pipeline_explore as m
            root  = Path(self.ed_root.text().strip())
            job   = Path(self.job_dir) if self.job_dir else root
            phase = self.cb_explore_phase.currentText()

            if phase == "1":
                curated_dir  = root / "data" / "curated"
                metadata_csv = root / "data" / "curated_metadata.csv"
                fig_dir      = root / "data" / "figures"
            elif phase == "2":
                curated_dir  = job / "data" / "curated" / "phase2"
                metadata_csv = job / "data" / "curated" / "phase2_metadata.csv"
                fig_dir      = job / "data" / "figures" / "phase2"
            else:
                curated_dir  = job / "data" / "curated" / "phase3"
                metadata_csv = job / "data" / "curated" / "phase3_metadata.csv"
                fig_dir      = job / "data" / "figures" / "phase3"

            return m.main, (), dict(
                curated_dir  = curated_dir,
                metadata_csv = metadata_csv,
                fig_dir      = fig_dir,
            )

        return self._build_pipeline_tab(
            "Step 4 — Database Exploration",
            params, run_fn, "explore", 3, 5
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — STEP 5: VOXELIZAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step5(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.ed_vox_pair = QLineEdit()
        self.ed_vox_pair.setPlaceholderText("empty = process all pairs")
        self.sp_vox_workers = QSpinBox()
        self.sp_vox_workers.setRange(0,16); self.sp_vox_workers.setValue(0)
        self.sp_vox_workers.setToolTip("0 = automatic (recommended = 1 for CUDA)")
        self.sp_grid = QSpinBox()
        self.sp_grid.setRange(16,64); self.sp_grid.setValue(32)
        self.sp_vox_pad = QDoubleSpinBox()
        self.sp_vox_pad.setRange(0.5,8.0); self.sp_vox_pad.setValue(3.0)
        self.sp_max_qm = QSpinBox()
        self.sp_max_qm.setRange(10,200); self.sp_max_qm.setValue(60)
        self.sp_max_qm.setToolTip(
            "Limite de átomos pesados para QM completo (RHF/STO-3G).\n"
            "Usado tanto para o ligante (não conectado ao tensor hoje) quanto\n"
            "para CADA RESÍDUO do sítio no método 'RHF/STO-3G per residue'\n"
            "(um resíduo protonado raramente passa de ~25 átomos, então o\n"
            "padrão de 60 quase sempre é suficiente)."
        )
        self.sp_vox_batch = QSpinBox()
        self.sp_vox_batch.setRange(10, 2000)
        self.sp_vox_batch.setValue(_BATCH_DEFAULT)
        self.sp_vox_batch.setSingleStep(50)
        self.sp_vox_batch.setToolTip(
            "Átomos por lote na voxelização gaussiana.\n"
            "CuPy (GPU): 500 recomendado (RTX 8–16 GB).\n"
            "NumPy (CPU): 100 recomendado.\n"
            "Valores maiores reduzem overhead de kernel mas aumentam uso de VRAM.\n"
            f"Preset detectado: {_BATCH_DEFAULT} ({'CuPy disponível' if _CUPY_OK else 'CuPy indisponível'})"
        )
        self.cb_qm_basis = QComboBox()
        self.cb_qm_basis.addItems(["sto-3g","3-21g","6-31g"])
        self.chk_ecfp4 = QCheckBox("Use ECFP4 in graph (reduces diversity)")
        self.cb_vox_phase = QComboBox()
        self.cb_vox_phase.addItems(["1", "2", "3"])

        # ── Protonação do sítio + método QM dos canais 3-4 (doador/aceptor) ──
        self.chk_protonate_site = QCheckBox("Protonate site residues (pH-based)")
        self.chk_protonate_site.setChecked(True)
        self.chk_protonate_site.setToolTip(
            "Adiciona H explícitos aos resíduos do sítio conforme pKa padrão\n"
            "e o pH abaixo (Asp/Glu/His/Cys/Tyr/Lys/Arg). Também troca a carga\n"
            "aproximada por elemento por cargas Gasteiger reais por átomo.\n"
            "Desligar volta ao comportamento legado (sem H, carga aproximada)."
        )
        self.dsb_site_ph = QDoubleSpinBox()
        self.dsb_site_ph.setDecimals(1); self.dsb_site_ph.setRange(0.0, 14.0)
        self.dsb_site_ph.setValue(7.4); self.dsb_site_ph.setSingleStep(0.5)
        self.dsb_site_ph.setToolTip(
            "pH usado para decidir o estado de protonação dos resíduos\n"
            "ionizáveis do sítio (regra de limiar sobre pKa de tabela)."
        )
        self.cb_site_qm_method = QComboBox()
        self.cb_site_qm_method.addItems([
            "RHF/STO-3G per residue (fragmented)",
            "GFN-xTB (whole site, requires xtb binary)",
            "Heuristic (electronegativity, no QM)",
        ])
        self.cb_site_qm_method.setToolTip(
            "Nível de teoria para os canais 3-4 (campo doador/aceptor do sítio):\n"
            "  RHF/STO-3G por resíduo — QM real, fragmentado tipo FMO (mais lento,\n"
            "    ~1s/resíduo; usa PySCF/GPU4PySCF já instalado).\n"
            "  GFN-xTB no sítio inteiro — semi-empírico, mais rápido, requer o\n"
            "    binário `xtb` no PATH (não instalado por padrão).\n"
            "  Heurística — eletronegatividade + carga, sem QM (mais rápido,\n"
            "    também o fallback automático quando os outros dois falham)."
        )
        self._SITE_QM_METHOD_VALUES = [
            "rhf_sto3g_fragments", "gfn_xtb_full", "heuristic",
        ]

        pf.addRow("Phase:", self.cb_vox_phase)
        pf.addRow("Specific pair:",    self.ed_vox_pair)
        pf.addRow("Workers (0=auto):", self.sp_vox_workers)
        pf.addRow("Grid size (N³):",   self.sp_grid)
        pf.addRow("Padding (Å):",      self.sp_vox_pad)
        pf.addRow("Max QM atoms:",     self.sp_max_qm)
        pf.addRow("Vox batch size:",   self.sp_vox_batch)
        pf.addRow("QM basis:",         self.cb_qm_basis)
        pf.addRow("",                  self.chk_ecfp4)
        pf.addRow("",                  self.chk_protonate_site)
        pf.addRow("Site pH:",          self.dsb_site_ph)
        pf.addRow("Site channels 3-4 method:", self.cb_site_qm_method)

        def run_fn():
            import MODULES.pipeline_voxelize as m
            import types

            config = dict(m.VOXEL_CONFIG)
            config["grid_size"] = self.sp_grid.value()
            config["padding_A"] = self.sp_vox_pad.value()
            config["max_qm_atoms"] = self.sp_max_qm.value()
            config["vox_batch"]    = self.sp_vox_batch.value()
            config["qm_basis"] = self.cb_qm_basis.currentText()
            config["use_ecfp4"] = self.chk_ecfp4.isChecked()
            config["protonate_site"] = self.chk_protonate_site.isChecked()
            config["ph"] = self.dsb_site_ph.value()
            config["site_qm_method"] = self._SITE_QM_METHOD_VALUES[
                self.cb_site_qm_method.currentIndex()
            ]

            root = Path(self.ed_root.text().strip())
            job = Path(self.job_dir)
            phase = self.cb_vox_phase.currentText()

            if phase == "1":
                curated_dir = root / "data" / "curated"
                tensor_dir  = root / "data" / "tensors"
                log_dir     = root / "data" / "logs"
            elif phase == "2":
                curated_dir = job / "data" / "curated" / "phase2"
                tensor_dir  = job / "data" / "tensors" / "phase2"
                log_dir     = job / "data" / "logs" / "phase2"
            else:
                curated_dir = job / "data" / "curated" / "phase3"
                tensor_dir  = job / "data" / "tensors" / "phase3"
                log_dir     = job / "data" / "logs" / "phase3"

            config["log_dir"] = str(log_dir)

            args_ns = types.SimpleNamespace(
                pair=self.ed_vox_pair.text().strip() or None,
                workers=self.sp_vox_workers.value() or None,
            )

            return m.run, (args_ns,), {
                "curated_dir": curated_dir,
                "tensor_dir": tensor_dir,
                "config": config,
            }
        return self._build_pipeline_tab(
            "Step 5 — 3D Voxelization + Molecular Graphs",
            params, run_fn, "voxelize", 4, 6
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — STEP 6: TREINAMENTO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step6(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(8)

        self.cb_train_phase = QComboBox()
        self.cb_train_phase.addItems(["1", "2", "3"])
        self.sp_epochs     = QSpinBox(); self.sp_epochs.setRange(1,5000); self.sp_epochs.setValue(100)
        self.sp_batch      = QSpinBox(); self.sp_batch.setRange(1,512);   self.sp_batch.setValue(8)
        self.sp_latent     = QSpinBox(); self.sp_latent.setRange(2,1024); self.sp_latent.setValue(128)
        self.sp_hidden     = QSpinBox(); self.sp_hidden.setRange(16,4096);self.sp_hidden.setValue(128)
        self.sp_gnn_layers = QSpinBox(); self.sp_gnn_layers.setRange(1,12);self.sp_gnn_layers.setValue(3)
        self.dsb_lr        = QDoubleSpinBox()
        self.dsb_lr.setDecimals(6); self.dsb_lr.setRange(1e-6,1.0); self.dsb_lr.setValue(3e-4)
        self.dsb_beta      = QDoubleSpinBox()
        self.dsb_beta.setDecimals(3); self.dsb_beta.setRange(0,10); self.dsb_beta.setValue(0.5)
        self.sp_warmup     = QSpinBox(); self.sp_warmup.setRange(1,500); self.sp_warmup.setValue(30)
        self.dsb_dropout   = QDoubleSpinBox()
        self.dsb_dropout.setDecimals(2); self.dsb_dropout.setRange(0,0.9); self.dsb_dropout.setValue(0.35)
        self.sp_patience   = QSpinBox(); self.sp_patience.setRange(1,500); self.sp_patience.setValue(15)
        self.cb_train_device = QComboBox()
        self.cb_train_device.addItems(["cuda","cpu","cuda:0","cuda:1"])
        self.cb_train_device.setCurrentText(_DEVICE_DEFAULT)

        # Campo para retomada de treino / fine-tuning
        self.ed_resume = QLineEdit(); self.ed_resume.setPlaceholderText("optional — best_model.pt from previous phase")
        self.ed_resume.setReadOnly(True)
        btn_resume = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        resume_row = QWidget(); rl = QHBoxLayout(resume_row); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self.ed_resume); rl.addWidget(btn_resume)
        btn_resume.clicked.connect(self._browse_resume)
        self.chk_freeze_enc = QCheckBox("Freeze encoder (Phase 3 fine-tuning)")

        self.ed_ckpt_out = QLineEdit(); self.ed_ckpt_out.setPlaceholderText("checkpoints/")
        self.ed_ckpt_out.setReadOnly(True)
        btn_ckpt = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        ckpt_row = QWidget(); cl = QHBoxLayout(ckpt_row); cl.setContentsMargins(0,0,0,0)
        cl.addWidget(self.ed_ckpt_out); cl.addWidget(btn_ckpt)
        btn_ckpt.clicked.connect(self._browse_ckpt_out)

        for label, widget in [
            ("Phase:",          self.cb_train_phase),
            ("Epochs:",         self.sp_epochs),
            ("Batch size:",     self.sp_batch),
            ("Latent dim:",     self.sp_latent),
            ("Hidden dim:",     self.sp_hidden),
            ("GNN layers:",     self.sp_gnn_layers),
            ("Learning rate:",  self.dsb_lr),
            ("Beta KL (max):",  self.dsb_beta),
            ("Beta warmup:",    self.sp_warmup),
            ("Dropout:",        self.dsb_dropout),
            ("Patience:",       self.sp_patience),
            ("Device:",         self.cb_train_device),
            ("Resume from:",    resume_row),
            ("",                self.chk_freeze_enc),
            ("Save ckpt to:",   ckpt_row),
        ]:
            pf.addRow(label, widget)  

        def run_fn():
            import MODULES.pipeline_train as m

            root = Path(self.ed_root.text().strip())
            job = Path(self.job_dir)
            phase = self.cb_train_phase.currentText()

            if phase == "1":
                tensor_dir = root / "data" / "tensors"
                ckpt_out   = Path(self.ed_ckpt_out.text().strip() or root / "checkpoints" / "phase1")
                resume     = self.ed_resume.text().strip() or None

            elif phase == "2":
                tensor_dir    = job / "data" / "tensors" / "phase2"
                ckpt_out      = Path(self.ed_ckpt_out.text().strip() or job / "checkpoints" / "phase2")
                # Usa campo Browse se preenchido; caso contrário sugere fase 1 se existir
                _resume_field = self.ed_resume.text().strip()
                _resume_auto  = root / "checkpoints" / "phase1" / "best_model.pt"
                if _resume_field:
                    resume = _resume_field
                elif _resume_auto.exists():
                    resume = str(_resume_auto)
                    log.info(f"Phase 2: fine-tuning a partir de {_resume_auto}")
                else:
                    resume = None   # treina do zero com dados de fase 2

            else:  # phase == "3"
                tensor_dir    = job / "data" / "tensors" / "phase3"
                ckpt_out      = Path(self.ed_ckpt_out.text().strip() or job / "checkpoints" / "phase3")
                # Usa campo Browse se preenchido; caso contrário sugere fase 2 se existir,
                # depois fase 1, depois treina do zero
                _resume_field = self.ed_resume.text().strip()
                _resume_p2    = job  / "checkpoints" / "phase2" / "best_model.pt"
                _resume_p1    = root / "checkpoints" / "phase1" / "best_model.pt"
                if _resume_field:
                    resume = _resume_field
                elif _resume_p2.exists():
                    resume = str(_resume_p2)
                    log.info(f"Phase 3: fine-tuning a partir de {_resume_p2}")
                elif _resume_p1.exists():
                    resume = str(_resume_p1)
                    log.info(f"Phase 3: nenhum checkpoint de fase 2 encontrado, "
                            f"fine-tuning a partir de fase 1 ({_resume_p1})")
                else:
                    resume = None   # treina do zero com dados de fase 3

            if not (tensor_dir / "tensor_index.csv").exists():
                raise ValueError(f"Tensor dataset not found for Phase {phase}: {tensor_dir}")

            return m.train, (), dict(
                epochs=self.sp_epochs.value(),
                batch_size=self.sp_batch.value(),
                latent_dim=self.sp_latent.value(),
                hidden_dim=self.sp_hidden.value(),
                n_gnn_layers=self.sp_gnn_layers.value(),
                lr=self.dsb_lr.value(),
                beta_max=self.dsb_beta.value(),
                beta_warmup=self.sp_warmup.value(),
                dropout=self.dsb_dropout.value(),
                patience=self.sp_patience.value(),
                checkpoint_dir=str(ckpt_out),
                tensor_dir=str(tensor_dir),
                resume=resume,
                freeze_encoder=self.chk_freeze_enc.isChecked(),
                device=self.cb_train_device.currentText(),
            )

        return self._build_pipeline_tab(
            "Step 6 — CVAE / GNN Training",
            params, run_fn, "train", 5, 7, log_height=320
        )

    def _browse_resume(self):
        p, _ = QFileDialog.getOpenFileName(self,"Select best_model.pt",
                                           self.ed_root.text(),"PyTorch (*.pt)")
        if p: self.ed_resume.setText(p)

    def _browse_ckpt_out(self):
        d = QFileDialog.getExistingDirectory(self,"Checkpoints folder", self.ed_root.text())
        if d: self.ed_ckpt_out.setText(d)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 7 — STEP 7: AVALIAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step7(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.ed_eval_ckpt = QLineEdit(); self.ed_eval_ckpt.setReadOnly(True)
        self.ed_eval_ckpt.setPlaceholderText("checkpoints/ (default)")
        btn_eval_ckpt = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_eval_ckpt.clicked.connect(self._browse_eval_ckpt)
        eck_row = QWidget(); el = QHBoxLayout(eck_row); el.setContentsMargins(0,0,0,0)
        el.addWidget(self.ed_eval_ckpt); el.addWidget(btn_eval_ckpt)
        self.chk_do_generate = QCheckBox("Generate ligands during evaluation")

        pf.addRow("Checkpoints:", eck_row)
        pf.addRow("",             self.chk_do_generate)

        # Botões de visualização de figuras
        vis_row = QWidget(); vl = QHBoxLayout(vis_row); vl.setContentsMargins(0,0,0,0)
        for name, stem in [("Training Curves","training_curves"),
                            ("Performance Report","performance_report")]:
            b = _mk_btn(name, _SS_BTN_PRIMARY, 180)
            b.clicked.connect(
                lambda _, s=stem, t=name: self._view_eval_fig(s, t)
            )
            vl.addWidget(b)
        pf.addRow("Figures:", vis_row)

        self.cb_eval_phase = QComboBox()
        self.cb_eval_phase.addItems(["1", "2", "3"])
        pf.insertRow(0, "Phase:", self.cb_eval_phase)
        
        def run_fn():
            import MODULES.pipeline_evaluate as m

            root = Path(self.ed_root.text().strip())
            job = Path(self.job_dir)
            phase = self.cb_eval_phase.currentText()

            if phase == "1":
                ckpt_dir = Path(self.ed_eval_ckpt.text().strip() or root / "checkpoints" / "phase1")
                tensor_dir = root / "data" / "tensors"
                fig_dir = root / "data" / "figures"
            elif phase == "2":
                ckpt_dir = Path(self.ed_eval_ckpt.text().strip() or job / "checkpoints" / "phase2")
                tensor_dir = job / "data" / "tensors" / "phase2"
                fig_dir = job / "data" / "figures" / "phase2"
            else:
                ckpt_dir = Path(self.ed_eval_ckpt.text().strip() or job / "checkpoints" / "phase3")
                tensor_dir = job / "data" / "tensors" / "phase3"
                fig_dir = job / "data" / "figures" / "phase3"

            return m.main, (), {
                "ckpt_dir": ckpt_dir,
                "do_generate": self.chk_do_generate.isChecked(),
                "tensor_dir": tensor_dir,
                "fig_dir": fig_dir,
            }

        return self._build_pipeline_tab(
            "Step 7 — Trained Model Evaluation",
            params, run_fn, "evaluate", 6, 8
        )

    def _browse_eval_ckpt(self):
        d = QFileDialog.getExistingDirectory(self,"Checkpoints folder", self.ed_root.text())
        if d: self.ed_eval_ckpt.setText(d)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 8 — STEP 8: GERAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step8(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)
        self.cb_gen_phase = QComboBox()
        self.cb_gen_phase.addItems(["1", "2", "3"])
        self.ed_gen_pdb = QLineEdit(); self.ed_gen_pdb.setReadOnly(True)
        self.ed_gen_pdb.setPlaceholderText("Target protein PDB file (required)")
        btn_pdb = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_pdb.clicked.connect(self._browse_gen_pdb)
        pdb_row = QWidget(); pl = QHBoxLayout(pdb_row); pl.setContentsMargins(0,0,0,0)
        pl.addWidget(self.ed_gen_pdb); pl.addWidget(btn_pdb)

        self.ed_gen_site_pdb = QLineEdit(); self.ed_gen_site_pdb.setReadOnly(True)
        self.ed_gen_site_pdb.setPlaceholderText("optional — PDB with only binding-site residues")
        btn_site_pdb = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_site_pdb.clicked.connect(self._browse_gen_site_pdb)
        site_pdb_row = QWidget(); spl = QHBoxLayout(site_pdb_row); spl.setContentsMargins(0,0,0,0)
        spl.addWidget(self.ed_gen_site_pdb); spl.addWidget(btn_site_pdb)

        self.ed_gen_ckpt = QLineEdit(); self.ed_gen_ckpt.setReadOnly(True)
        self.ed_gen_ckpt.setPlaceholderText("checkpoints/ (default)")
        btn_gc = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_gc.clicked.connect(self._browse_gen_ckpt)
        gc_row = QWidget(); gl2 = QHBoxLayout(gc_row); gl2.setContentsMargins(0,0,0,0)
        gl2.addWidget(self.ed_gen_ckpt); gl2.addWidget(btn_gc)

        self.ed_gen_out = QLineEdit(); self.ed_gen_out.setReadOnly(True)
        self.ed_gen_out.setPlaceholderText("JOBS/<job_id>/results/{pdb}/ (default)")
        btn_gout = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_gout.clicked.connect(self._browse_gen_out)
        gout_row = QWidget(); gl3 = QHBoxLayout(gout_row); gl3.setContentsMargins(0,0,0,0)
        gl3.addWidget(self.ed_gen_out); gl3.addWidget(btn_gout)

        self.ed_ligand_ref = QLineEdit(); self.ed_ligand_ref.setReadOnly(True)
        self.ed_ligand_ref.setPlaceholderText("optional — reference ligand SDF")
        btn_ref = _mk_btn("Browse...", _SS_BTN_PRIMARY, 100)
        btn_ref.clicked.connect(self._browse_ligand_ref)
        ref_row = QWidget(); rl2 = QHBoxLayout(ref_row); rl2.setContentsMargins(0,0,0,0)
        rl2.addWidget(self.ed_ligand_ref); rl2.addWidget(btn_ref)

        self.sp_n_samples  = QSpinBox(); self.sp_n_samples.setRange(1,10000); self.sp_n_samples.setValue(200)
        self.dsb_temp      = QDoubleSpinBox()
        self.dsb_temp.setDecimals(2); self.dsb_temp.setRange(0.1,5.0); self.dsb_temp.setValue(1.0)
        self.sp_cutoff_gen = QDoubleSpinBox()
        self.sp_cutoff_gen.setDecimals(1); self.sp_cutoff_gen.setRange(3.0,15.0); self.sp_cutoff_gen.setValue(6.0)
        self.sp_top_n      = QSpinBox(); self.sp_top_n.setRange(1,200); self.sp_top_n.setValue(20)
        self.sp_min_atoms_gen = QSpinBox(); self.sp_min_atoms_gen.setRange(1,30); self.sp_min_atoms_gen.setValue(5)

        # ── Protonação do sítio + método QM — DEVE bater com o que foi usado
        # para gerar os tensores de treino da fase escolhida, senão o tensor
        # de inferência não tem a mesma distribuição do treino (ver nota em
        # pipeline_generate.pdb_to_site_tensor). Defaults iguais aos da
        # Step 5 por esse motivo — mude só se souber que o treino usou algo
        # diferente.
        self.chk_gen_protonate_site = QCheckBox("Protonate site residues (pH-based)")
        self.chk_gen_protonate_site.setChecked(True)
        self.dsb_gen_ph = QDoubleSpinBox()
        self.dsb_gen_ph.setDecimals(1); self.dsb_gen_ph.setRange(0.0, 14.0)
        self.dsb_gen_ph.setValue(7.4); self.dsb_gen_ph.setSingleStep(0.5)
        self.cb_gen_site_qm_method = QComboBox()
        self.cb_gen_site_qm_method.addItems([
            "RHF/STO-3G per residue (fragmented)",
            "GFN-xTB (whole site, requires xtb binary)",
            "Heuristic (electronegativity, no QM)",
        ])
        self.cb_gen_site_qm_method.setToolTip(
            "DEVE ser igual ao método usado para gerar os tensores de treino\n"
            "desta fase (Step 5) — um mismatch aqui reintroduz a diferença\n"
            "treino↔inferência que os canais 3-4 foram desenhados para evitar."
        )
        self._GEN_SITE_QM_METHOD_VALUES = [
            "rhf_sto3g_fragments", "gfn_xtb_full", "heuristic",
        ]

        for label, widget in [
            ("Model phase:",          self.cb_gen_phase),
            ("Protein (PDB):",       pdb_row),
            ("Site residues (PDB):", site_pdb_row),
            ("Checkpoints:",     gc_row),
            ("Output:",              gout_row),
            ("Ref. ligand:",         ref_row),
            ("No. samples:",         self.sp_n_samples),
            ("Temperature:",         self.dsb_temp),
            ("Site cutoff (Å):",     self.sp_cutoff_gen),
            ("Top N report:",        self.sp_top_n),
            ("Min. site atoms:",     self.sp_min_atoms_gen),
            ("",                     self.chk_gen_protonate_site),
            ("Site pH:",             self.dsb_gen_ph),
            ("Site channels 3-4 method:", self.cb_gen_site_qm_method),
        ]:
            pf.addRow(label, widget)

        def run_fn():
            pdb = self.ed_gen_pdb.text().strip()
            if not pdb or not os.path.isfile(pdb):
                raise ValueError("Please select a valid PDB file for the target protein.")
            site_pdb = self.ed_gen_site_pdb.text().strip() or None
            if site_pdb and not os.path.isfile(site_pdb):
                raise ValueError("Please select a valid PDB file containing only the binding-site residues.")
            root   = self.ed_root.text().strip()
            phase = self.cb_gen_phase.currentText()
            if phase == "1":
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(root) / "checkpoints" / "phase1")
            elif phase == "2":
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(self.job_dir) / "checkpoints" / "phase2")
            else:
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(self.job_dir) / "checkpoints" / "phase3")

            if not (Path(ckpt) / "best_model.pt").exists():
                raise ValueError(f"No trained checkpoint found for Phase {phase}.")
            out    = self.ed_gen_out.text().strip() or None
            ref    = self.ed_ligand_ref.text().strip() or None
            from pathlib import Path as _P

            return _pipeline_generate, (), dict(
                root        = root,
                job_dir     = self.job_dir or None,
                pdb         = pdb,
                ckpt        = ckpt,
                n_samples   = self.sp_n_samples.value(),
                temperature = self.dsb_temp.value(),
                cutoff      = self.sp_cutoff_gen.value(),
                min_atoms   = self.sp_min_atoms_gen.value(),
                top_n       = self.sp_top_n.value(),
                out         = self.ed_gen_out.text().strip() or None,
                ligand_ref  = self.ed_ligand_ref.text().strip() or None,
                site_pdb    = site_pdb,
                site_config_overrides = dict(
                    protonate_site = self.chk_gen_protonate_site.isChecked(),
                    ph             = self.dsb_gen_ph.value(),
                    site_qm_method = self._GEN_SITE_QM_METHOD_VALUES[
                        self.cb_gen_site_qm_method.currentIndex()
                    ],
                ),
            )

        return self._build_pipeline_tab(
            "Step 8 — Ligand Candidate Generation",
            params, run_fn, "generate", 7, 0, log_height=260
        )

    def _browse_gen_pdb(self):
        p, _ = QFileDialog.getOpenFileName(self,"Select PDB file", self.ed_root.text(),"PDB (*.pdb)")
        if p: self.ed_gen_pdb.setText(p)

    def _browse_gen_site_pdb(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select binding-site residues PDB",
            self.ed_root.text(),
            "PDB (*.pdb)",
        )
        if p: self.ed_gen_site_pdb.setText(p)

    def _browse_gen_ckpt(self):
        d = QFileDialog.getExistingDirectory(self,"Checkpoints folder", self.ed_root.text())
        if d: self.ed_gen_ckpt.setText(d)

    def _browse_moad_dir(self):
        d = QFileDialog.getExistingDirectory(self, "MOAD folder", self.ed_root.text())
        if d: self.ed_moad_dir.setText(d)

    def _browse_gen_out(self):
        d = QFileDialog.getExistingDirectory(self,"Output folder", self.ed_root.text())
        if d: self.ed_gen_out.setText(d)

    def _browse_ligand_ref(self):
        p, _ = QFileDialog.getOpenFileName(self,"Reference ligand",
                                           self.ed_root.text(),"Ligand files (*.sdf *.pdb *.mol2)")
        
        
        if p: self.ed_ligand_ref.setText(p)

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS GERAIS
    # ══════════════════════════════════════════════════════════════════════
    def _view_fig(self, stem, title):
        """Exibe figura a partir do fig_dir padrão (root/data/figures)."""
        root    = self.ed_root.text().strip()
        fig_dir = os.path.join(root, "data", "figures") if root else ""
        for ext in ("png", "svg"):
            path = os.path.join(fig_dir, f"{stem}.{ext}")
            if os.path.isfile(path):
                self._view_fig_path(path, title)
                return
        QMessageBox.information(self, "Figure not found",
                                f"Run the corresponding step first.\n\nExpected at:\n{fig_dir}")

    def _view_fig_path(self, path: str, title: str):
        """Exibe uma figura dado o caminho absoluto."""
        dlg = QDialog(self); dlg.setWindowTitle(title); dlg.resize(920, 640)
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea(); lbl = QLabel()
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Error", f"Could not load:\n{path}"); return
        lbl.setPixmap(pix.scaled(880, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl); scroll.setWidgetResizable(True)
        lay.addWidget(scroll)
        br = QHBoxLayout()
        b_open = _mk_btn("Open in system", _SS_BTN_PRIMARY, 160)
        b_open.clicked.connect(lambda: subprocess.Popen(["xdg-open", path]))
        b_close = _mk_btn("Close", _SS_BTN_PRIMARY, 100)
        b_close.clicked.connect(dlg.accept)
        br.addStretch(); br.addWidget(b_open); br.addWidget(b_close)
        lay.addLayout(br)
        dlg.exec_()

    def _view_explore_dashboard(self):
        """Abre o dashboard da fase selecionada no Step 4."""
        phase = self.cb_explore_phase.currentText()
        root  = self.ed_root.text().strip()
        job   = self.job_dir or root

        if phase == "1":
            fig_dir = os.path.join(root, "data", "figures")
        elif phase == "2":
            fig_dir = os.path.join(job, "data", "figures", "phase2")
        else:
            fig_dir = os.path.join(job, "data", "figures", "phase3")

        for ext in ("png", "svg"):
            path = os.path.join(fig_dir, f"summary_dashboard.{ext}")
            if os.path.isfile(path):
                self._view_fig_path(path, f"Summary Dashboard — Phase {phase}")
                return

        QMessageBox.information(
            self, "Figure not found",
            f"Run Step 4 with Phase {phase} first.\n\nExpected at:\n{fig_dir}"
        )

    def _view_eval_fig(self, stem: str, title: str):
        """Abre figura de avaliação da fase selecionada no Step 7."""
        phase = self.cb_eval_phase.currentText()
        root  = Path(self.ed_root.text().strip())
        job   = Path(self.job_dir) if self.job_dir else root

        if phase == "1":
            fig_dir = root / "data" / "figures"
        elif phase == "2":
            fig_dir = job / "data" / "figures" / "phase2"
        else:
            fig_dir = job / "data" / "figures" / "phase3"

        for ext in ("png", "svg"):
            path = fig_dir / f"{stem}.{ext}"
            if path.exists():
                self._view_fig_path(str(path), f"{title} — Phase {phase}")
                return

        QMessageBox.information(
            self, "Figure not found",
            f"Run Step 7 with Phase {phase} first.\n\nExpected at:\n{fig_dir}"
        )

    def _show_requirements(self):
        if not _REQ_MODULE_OK:
            QMessageBox.warning(self,"Module not found",
                                "CODEEP_requirements.py was not found in the CODEEP folder."); return
        self._req_win = RequirementsInstaller(None)
        self._req_win.setWindowModality(Qt.ApplicationModal)
        self._req_win.show(); self._req_win.activateWindow(); self._req_win.raise_()

    def _show_about(self):
        QMessageBox.about(self,"About CODEEP",
            "CODEEP\nDeep Learning Ligand Generation Pipeline\n\n"
            "Part of the suite:\n   CODOC · CODYN · CODEEP · CODRUG\n\n"
            "Developed by:\n   Moisés Maia Neto\n"
            "   CEB / Universidade Federal do Paraná - UFPR\n\n"
            "   Brasil\n\n"
            "Contact: moimaian@gmail.com\n\nVersion 1.0 (beta)   © 2024–2026")


# ==========================================================================================================================================
# ═══════════════════════ PONTO DE ENTRADA ═════════════════════════════════
# ==========================================================================================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("CODEEP")

    splash = SplashScreen()
    splash.show()

    def _on_ready():
        window = MainWindow()
        window.show()

    splash.ready_signal.connect(_on_ready)
    splash.start()
    sys.exit(app.exec_())
