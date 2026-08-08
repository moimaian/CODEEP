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

from pathlib import Path
from versao import VERSAO, LICENCA_CURTA


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
        f"Version={VERSAO}\n"
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
import json, time, importlib, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── PyQt5 ─────────────────────────────────────────────────────────────────
from PyQt5.QtCore    import Qt, QThread, pyqtSignal, QTimer, QDate, QSettings, QSize
from PyQt5.QtGui     import QPixmap, QGuiApplication, QFont, QIcon
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

# ── Internacionalização (PT-BR/EN, bandeiras BR/UK — mesmo padrão do CODRUG,
#    ver MODULES/i18n.py) ───────────────────────────────────────────────────
import MODULES.i18n as i18n

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

# CuPy: NumPy acelerado por CUDA, usado pela voxelização (pipeline_voxelize.py).
# NOTA: CuPy 14 tem um bug nos headers CCCL que causa NVRTC_ERROR_COMPILATION
# em kernels de reduction via CUB (.sum(axis=0), matmul, tensordot). Porém
# cp.einsum('ij->j', x) usa cuBLAS e funciona corretamente — por isso a
# checagem abaixo executa um einsum de verdade em vez de só testar se a lib
# importa ou se um arquivo .so existe (mesma lógica de pipeline_voxelize.py,
# a fonte da verdade de onde CuPy é de fato usado). Só tenta se houver GPU
# CUDA — em CPU-only, `import cupy` já falharia, mas evita o overhead.
_CUPY_OK = False
if _DEVICE_DEFAULT == "cuda":
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
    background: #F57C6F;
    border-color: #F57C6F;
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

_SS_BTN_SECONDARY = """
QPushButton {
    background: #243746;
    color: #C9D1D9;
    font-weight: bold;
    border: 1px solid #6E8CA8;
    border-radius: 5px;
    padding: 6px 16px;
}
QPushButton:hover   { background: #37536C; color: #FFFFFF; }
QPushButton:pressed { background: #1C3249; color: #FFF; }
QPushButton:disabled{ background: #1B2730; color: #4F6577; border-color: #344654; }
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
QPushButton:hover { color: #F57C6F; }
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

_SS_LABEL_TITLE   = "font-size:14pt; font-weight:bold; color:#FFFFFF;"
_SS_LABEL_SECTION = "font-size:11pt; font-weight:bold; color:#C9D1D9;"

# QGroupBox sem cor de fundo própria (só a borda) — sobrepõe localmente o "background:
# #14253A" do QGroupBox global (_SS_MAIN acima) com o mesmo tom da janela principal
# (BG_DEEP #0D1B2A), mantendo borda/título/margens herdados do estilo global (o Qt mescla
# propriedade a propriedade entre o stylesheet local do widget e o da aplicação — só a
# propriedade especificada aqui é sobrescrita). Usado nos grupos da aba Home.
_SS_GROUPBOX_NO_BG = "QGroupBox { background: #0D1B2A; }"


class CollapsibleGroupBox(QGroupBox):
    """QGroupBox colapsável ao clicar no título ou na borda."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._collapsed = False

    def _collapsed_height(self):
        return max(28, self.fontMetrics().height() + 14)

    def _set_layout_widgets_visible(self, layout, visible):
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(visible)
            child_layout = item.layout()
            if child_layout is not None:
                self._set_layout_widgets_visible(child_layout, visible)

    def set_collapsed(self, collapsed):
        self._collapsed = bool(collapsed)
        self._set_layout_widgets_visible(self.layout(), not self._collapsed)
        if self._collapsed:
            h = self._collapsed_height()
            self.setMinimumHeight(h)
            self.setMaximumHeight(h)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.adjustSize()

    def _is_toggle_hit(self, pos):
        title_h = self._collapsed_height()
        border = 6
        if pos.y() <= title_h:
            return True
        return (
            pos.x() <= border
            or pos.x() >= self.width() - border
            or pos.y() >= self.height() - border
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_toggle_hit(event.pos()):
            self.set_collapsed(not self._collapsed)
            event.accept()
            return
        super().mousePressEvent(event)


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

def _mk_progress(idioma_getter=None):
    """Cria QProgressBar em modo indeterminado até receber valor real do tqdm.

    idioma_getter: callable() -> "en"/"pt" (ex.: lambda: self._idioma) — função de módulo, sem
    acesso a self, então precisa de um getter para saber o idioma ATUAL tanto na criação quanto
    (via _connect_worker) nas atualizações subsequentes de formato."""
    idioma = idioma_getter() if idioma_getter else i18n.IDIOMA_PADRAO
    p = QProgressBar()
    p.setRange(0, 100); p.setValue(0)
    p.setFormat(f"{i18n.t('progress_label', idioma)}: %p%  —  {i18n.t('progress_eta_label', idioma)}: —")
    p.setFixedHeight(22)
    return p

def _pulse(bar):
    """Incremento suave quando não há informação de progresso real na linha."""
    v = bar.value()
    if v < 30:          # avança livremente até 30%
        bar.setValue(v + 1)
    # acima de 30% só avança se tqdm emitir valor real

def _connect_worker(worker, log_box, prog_bar, btn_run, btn_stop, done_cb, idioma_getter=None):
    """Conecta sinais do PipelineWorker aos widgets da aba.

    - log_signal   → appenda linha no log e faz pulse suave
    - progress_signal → atualiza a barra com valor real do tqdm
    - eta_signal   → atualiza o texto da barra com ETA calculado
    - finished_signal → restaura botões e chama done_cb

    idioma_getter: callable() -> "en"/"pt" (ex.: lambda: self._idioma) — reavaliado a cada
    sinal (não capturado uma vez), para refletir o idioma ATUAL mesmo que o usuário troque de
    idioma enquanto um worker está rodando."""
    def _idioma():
        return idioma_getter() if idioma_getter else i18n.IDIOMA_PADRAO

    worker.log_signal.connect(
        lambda l: (_log(log_box, l), _pulse(prog_bar))
    )
    worker.progress_signal.connect(
        lambda pct: prog_bar.setValue(pct)
    )
    worker.eta_signal.connect(
        lambda eta: prog_bar.setFormat(
            f"{i18n.t('progress_label', _idioma())}: %p%  —  "
            f"{i18n.t('progress_eta_label', _idioma())}: {eta}")
    )
    worker.finished_signal.connect(lambda ok: (
        prog_bar.setValue(100),
        prog_bar.setFormat(
            f"{i18n.t('progress_label', _idioma())}: 100%  —  "
            f"{i18n.t('progress_done', _idioma())}"),
        btn_run.setEnabled(True),
        btn_stop.setEnabled(False),
        _log(log_box, i18n.t("log_completed", _idioma()) if ok
             else i18n.t("log_interrupted", _idioma())),
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
                       site_config_overrides=None, final_ff="mmff_uff",
                       induced_fit=False):
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

    # Modelo carregado ANTES do sítio — precisa da voxel_config gravada no
    # checkpoint (ver pipeline_generate.load_model) para resolver com que
    # config REVOXELIZAR esse sítio novo. protonate_site/ph/site_qm_method
    # (site_config_overrides, vindos da Etapa 8 da interface) só valem
    # como FALLBACK — para as chaves ausentes do checkpoint, ou quando ele
    # foi treinado antes desta funcionalidade existir; quando o checkpoint
    # tem um valor gravado, ele sempre vence (ver
    # resolve_site_config_overrides — evita o mismatch treino↔inferência
    # que motivou esses parâmetros existirem como config em primeiro lugar).
    model, mod, ckpt_voxel_config = m.load_model(
        ckpt_dir,
        project_dir=_P(root))           # ver Problema 3

    (site_tensor, origin, voxel_size, site_coords, site_elements,
     site_mol, site_is_backbone) = m.pdb_to_site_tensor(
        pdb_path, cutoff, min_atoms, site_pdb=site_pdb_path,
        project_dir=_P(root),           # ver Problema 3
        config_overrides=m.resolve_site_config_overrides(
            ckpt_voxel_config, site_config_overrides))
    np.savez_compressed(
        out_dir / "site_tensor.npz",
        tensor=site_tensor, origin=origin,
        voxel_size=np.array([voxel_size]))

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
        ref_mol=ref_mol,               # ← voxel_size retornado por pdb_to_site_tensor
        site_coords=site_coords, site_elements=site_elements,
        site_mol=site_mol, site_is_backbone=site_is_backbone,
        final_ff=final_ff, induced_fit=induced_fit)
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
# ══════════════════════ JANELA PRINCIPAL ══════════════════════════════════
# ==========================================================================================================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # Idioma da interface (PT-BR/EN, ver MODULES/i18n.py): lido de QSettings para persistir
        # entre sessões, mesmo padrão de bandeiras BR/UK do CODRUG/AgendaLab (i18n.py + widget de
        # bandeiras ao lado do CPU/GPU Monitor). self._i18n_registry guarda (chave, apply_fn,
        # kwargs) para cada texto já traduzido, permitindo reaplicar tudo numa troca de idioma
        # sem precisar reconstruir nenhum widget.
        self._idioma = QSettings("CODEEP", "CODEEP").value("idioma", i18n.IDIOMA_PADRAO)
        if self._idioma not in ("en", "pt"):
            self._idioma = i18n.IDIOMA_PADRAO
        self._i18n_registry = []

        self._tr("app_titulo", self.setWindowTitle)
        self.setStyleSheet(_SS_MAIN)
        sc = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(sc.width()*0.50), int(sc.height()*0.60))
        self.move((sc.width()-self.width())//2, (sc.height()-self.height())//2)

        self.dp_dir: str = str(Path(__file__).parent)
        self.job_dir: str = ""
        self._workers: dict = {}          # tag → worker ativo

        self._build_menu()
        self._build_hw_monitor()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_tabs()

    def _tr(self, chave, apply_fn, **kwargs):
        """Registra apply_fn (ex.: label.setText, btn.setToolTip, lambda txt:
        self.tabs.setTabText(0, txt)) para ser reexecutado sempre que o idioma da interface
        mudar, e já aplica o texto do idioma atual imediatamente. chave/kwargs são repassados
        para i18n.t() a cada reaplicação, então strings com placeholders dinâmicos (ex.:
        contagens, caminhos) continuam corretas após trocar de idioma."""
        self._i18n_registry.append((chave, apply_fn, kwargs))
        apply_fn(i18n.t(chave, self._idioma, **kwargs))

    def _trL(self, chave, **kwargs):
        """Cria um QLabel já registrado em i18n (ver _tr) — atalho para os muitos labels
        estáticos espalhados pelos formulários das abas Step 1-8."""
        lbl = QLabel()
        self._tr(chave, lbl.setText, **kwargs)
        return lbl

    def _mk_title_tr(self, chave):
        lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(_SS_LABEL_TITLE)
        self._tr(chave, lbl.setText)
        return lbl

    def _mk_section_tr(self, chave):
        lbl = QLabel(); lbl.setStyleSheet(_SS_LABEL_SECTION)
        self._tr(chave, lbl.setText)
        return lbl

    def _mk_btn_tr(self, chave, style=_SS_BTN_PRIMARY, width=130, **kwargs):
        btn = QPushButton(); btn.setFixedWidth(width); btn.setStyleSheet(style)
        self._tr(chave, btn.setText, **kwargs)
        return btn

    def _mk_combo_tr(self, item_keys):
        """QComboBox cujos itens são traduzidos e reaplicados numa troca de idioma (ver _tr) —
        usa setItemText em vez de clear()+addItems() para preservar currentIndex(). Chamadores
        que precisam do VALOR (não do texto exibido) de cada item devem indexar por
        currentIndex() contra sua própria lista paralela de valores internos (mesmo padrão já
        usado por _SITE_QM_METHOD_VALUES/_FINAL_FF_VALUES) — nunca por currentText(), que muda
        com o idioma."""
        combo = QComboBox()
        combo.addItems([""] * len(item_keys))
        for i, chave in enumerate(item_keys):
            self._tr(chave, lambda txt, c=combo, idx=i: c.setItemText(idx, txt))
        return combo

    def _mk_nav_row_tr(self, back_cb, next_cb):
        lay = QHBoxLayout()
        for chave, cb, align in [("nav_back", back_cb, Qt.AlignLeft),
                                  ("nav_next", next_cb, Qt.AlignRight)]:
            b = QPushButton(); b.setFixedWidth(200); b.setStyleSheet(_SS_BTN_NAV)
            self._tr(chave, b.setText)
            b.clicked.connect(cb)
            lay.addWidget(b, alignment=align)
        return lay

    def _definir_idioma(self, idioma):
        if idioma == self._idioma:
            return
        self._idioma = idioma
        QSettings("CODEEP", "CODEEP").setValue("idioma", idioma)
        for chave, apply_fn, kwargs in self._i18n_registry:
            apply_fn(i18n.t(chave, self._idioma, **kwargs))
            # A tradução para PT costuma ser mais longa que o texto original em EN — botões com
            # largura fixa foram dimensionados uma única vez na construção da interface, para o
            # texto do idioma inicial. Sem isso, o texto do idioma mais longo ficaria cortado.
            # _ensure_button_text_fits só AUMENTA a largura quando necessário (nunca reduz), então
            # o layout permanece estável ao alternar os idiomas.
            target = getattr(apply_fn, "__self__", None)
            if isinstance(target, QPushButton):
                self._ensure_button_text_fits(target)

    @staticmethod
    def _ensure_button_text_fits(button):
        text = button.text() or ""
        if not text.strip():
            return
        button.ensurePolished()
        required_width = button.sizeHint().width() + 8
        if button.icon() and not button.icon().isNull():
            required_width += button.iconSize().width()

        current_min = button.minimumWidth()
        current_max = button.maximumWidth()
        has_fixed_width = current_min > 0 and current_min == current_max

        if has_fixed_width:
            if current_min < required_width:
                button.setFixedWidth(required_width)
            return

        if current_min < required_width:
            button.setMinimumWidth(required_width)

    # ── Menu ──────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = self.menuBar()
        menu = mb.addMenu("")
        self._tr("menu_title", menu.setTitle)
        # Itens de menu sem ícones — chaves i18n na ordem das abas
        menu_keys = [
            "menu_home",  "menu_step1",
            "menu_step2", "menu_step3",
            "menu_step4", "menu_step5",
            "menu_step6", "menu_step7",
            "menu_step8",
        ]
        for i, chave in enumerate(menu_keys):
            act = QAction(self)
            self._tr(chave, act.setText)
            act.triggered.connect(lambda _, idx=i: self.tabs.setCurrentIndex(idx))
            menu.addAction(act)
        menu.addSeparator()
        act_exit = QAction(self, triggered=self.close)
        self._tr("menu_exit", act_exit.setText)
        menu.addAction(act_exit)

        help_menu = mb.addMenu("")
        self._tr("menu_help", help_menu.setTitle)
        act_req = QAction(self, triggered=self._show_requirements)
        self._tr("menu_install_requirements", act_req.setText)
        help_menu.addAction(act_req)
        help_menu.addSeparator()
        act_about = QAction(self, triggered=self._show_about)
        self._tr("menu_about", act_about.setText)
        help_menu.addAction(act_about)

    # ── Monitor CPU + GPU ─────────────────────────────────────────────────
    def _build_hw_monitor(self):
        self._mon_widget = QWidget()
        self._mon_widget.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(self._mon_widget)
        lay.setContentsMargins(0,0,4,0); lay.setSpacing(6)

        # Bandeiras BR/UK de seleção de idioma (mesmo padrão do CODRUG/AgendaLab,
        # MODULES/i18n.py) — ao lado do CPU/GPU Monitor, no canto superior direito da janela
        # principal. Usa ícones SVG próprios (ICONS/flag_br.svg, ICONS/flag_gb.svg) em vez de
        # emoji de bandeira: em muitos ambientes Linux/Qt o emoji de bandeira (par de "Regional
        # Indicator Symbols") não tem glifo colorido combinado disponível na fonte do sistema e
        # cai no fallback textual (as duas letras do código do país), em vez de desenhar a
        # bandeira.
        flag_style = (
            "QPushButton{border:none;background:transparent;padding:0;}"
            "QPushButton:hover{background:#16283B;border-radius:4px;}"
        )
        flag_icon_size = QSize(24, 16)

        self.btn_bandeira_pt = QPushButton()
        self.btn_bandeira_pt.setIcon(QIcon(os.path.join(self.dp_dir, "ICONS", "flag_br.svg")))
        self.btn_bandeira_pt.setIconSize(flag_icon_size)
        self.btn_bandeira_pt.setFlat(True)
        self.btn_bandeira_pt.setFixedSize(30, 22)
        self.btn_bandeira_pt.setStyleSheet(flag_style)
        self._tr("tooltip_bandeira_pt", self.btn_bandeira_pt.setToolTip)
        self.btn_bandeira_pt.clicked.connect(lambda: self._definir_idioma("pt"))
        lay.addWidget(self.btn_bandeira_pt)

        self.btn_bandeira_en = QPushButton()
        self.btn_bandeira_en.setIcon(QIcon(os.path.join(self.dp_dir, "ICONS", "flag_gb.svg")))
        self.btn_bandeira_en.setIconSize(flag_icon_size)
        self.btn_bandeira_en.setFlat(True)
        self.btn_bandeira_en.setFixedSize(30, 22)
        self.btn_bandeira_en.setStyleSheet(flag_style)
        self._tr("tooltip_bandeira_en", self.btn_bandeira_en.setToolTip)
        self.btn_bandeira_en.clicked.connect(lambda: self._definir_idioma("en"))
        lay.addWidget(self.btn_bandeira_en)

        # Monitor de CPU/GPU — mesmo padrão do CODRUG (MODULES/hw_monitor_window.py): em vez de
        # QLabels embutidos aqui atualizados por um QTimer da janela principal (que dependem do
        # event loop desta janela estar livre — e por isso "congelam" se algo bloquear esse event
        # loop), só um botão que abre uma janela de monitor num PROCESSO totalmente separado (seu
        # próprio QApplication/event loop), imune ao que a janela principal estiver fazendo.
        self.btn_open_hw_monitor = QPushButton()
        self._tr("btn_cpu_gpu_monitor", self.btn_open_hw_monitor.setText)
        self._tr(
            "tooltip_cpu_gpu_monitor",
            lambda txt: self.btn_open_hw_monitor.setToolTip(
                f"<html><p style='font-size:10pt;'>{txt}</p></html>"),
        )
        self.btn_open_hw_monitor.setStyleSheet(
            "QPushButton{padding:4px 12px;border:2px solid #6E8CA8;border-radius:6px;"
            "background:#0D1B2A;color:#6E8CA8;font-weight:bold;font-size:9pt;}"
            "QPushButton:hover{background:#16283B;}"
        )
        self.btn_open_hw_monitor.clicked.connect(self.open_hw_monitor_window)
        lay.addWidget(self.btn_open_hw_monitor)

        self.menuBar().setCornerWidget(self._mon_widget, Qt.TopRightCorner)

    def open_hw_monitor_window(self):
        """Abre MODULES/hw_monitor_window.py como um processo separado (não uma QThread): assim
        ele mantém seu próprio QApplication/event loop e continua atualizando a cada 1,5s mesmo
        que a janela principal esteja ocupada."""
        script_path = os.path.join(self.dp_dir, "MODULES", "hw_monitor_window.py")
        if not os.path.isfile(script_path):
            QMessageBox.warning(
                self, i18n.t("msg_title_monitor", self._idioma),
                i18n.t("msg_monitor_script_not_found", self._idioma, path=script_path))
            return
        try:
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            QMessageBox.critical(
                self, i18n.t("msg_title_monitor_error", self._idioma),
                i18n.t("msg_monitor_open_error", self._idioma, error=e))

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

    # ── Abas — sem ícones, apenas nome curto, traduzido via i18n ───────────
    def _build_tabs(self):
        pages_keys = [
            (self._tab_home(),  "tab_home"),
            (self._tab_step1(), "tab_step1"),
            (self._tab_step2(), "tab_step2"),
            (self._tab_step3(), "tab_step3"),
            (self._tab_step4(), "tab_step4"),
            (self._tab_step5(), "tab_step5"),
            (self._tab_step6(), "tab_step6"),
            (self._tab_step7(), "tab_step7"),
            (self._tab_step8(), "tab_step8"),
        ]
        for idx, (page, chave) in enumerate(pages_keys):
            self.tabs.addTab(page, "")
            self._tr(chave, lambda txt, i=idx: self.tabs.setTabText(i, txt))

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
        tagline = self._trL("home_tagline")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet("font-size:13pt;color:#95A5A6;")
        lay.addWidget(tagline)

        lay.addWidget(_mk_separator())

        # ── Linha dupla: Hardware Specs CPU | Hardware Specs GPU ───────────
        hw_row = QHBoxLayout(); hw_row.setSpacing(14)

        # Quadro CPU
        cpu_grp = CollapsibleGroupBox()
        cpu_grp.setStyleSheet(_SS_GROUPBOX_NO_BG)
        self._tr("home_grp_cpu", cpu_grp.setTitle)
        cpu_form = QFormLayout(cpu_grp)
        cpu_form.setHorizontalSpacing(16); cpu_form.setVerticalSpacing(4)
        for label, val in self._hw_cpu_info():
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            lbl_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            cpu_form.addRow(lbl_k, lbl_v)
        cpu_grp.set_collapsed(False)
        hw_row.addWidget(cpu_grp, 1)

        # Quadro GPU
        gpu_grp = CollapsibleGroupBox()
        gpu_grp.setStyleSheet(_SS_GROUPBOX_NO_BG)
        self._tr("home_grp_gpu", gpu_grp.setTitle)
        gpu_form = QFormLayout(gpu_grp)
        gpu_form.setHorizontalSpacing(16); gpu_form.setVerticalSpacing(4)
        for label, val in self._hw_gpu_info():
            lbl_k = QLabel(label)
            lbl_k.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            lbl_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            gpu_form.addRow(lbl_k, lbl_v)
        gpu_grp.set_collapsed(False)
        hw_row.addWidget(gpu_grp, 1)

        lay.addLayout(hw_row)

        # ── Software Specs ─────────────────────────────────────────────────
        sw_grp = CollapsibleGroupBox()
        sw_grp.setStyleSheet(_SS_GROUPBOX_NO_BG)
        self._tr("home_grp_sw", sw_grp.setTitle)
        sw_grid = QGridLayout(sw_grp)
        sw_grid.setHorizontalSpacing(24); sw_grid.setVerticalSpacing(4)
        sw_items = self._sw_versions()
        # Distribui em 3 colunas para não ficar muito alto
        cols = 3
        for i, (lib, ver) in enumerate(sw_items):
            row, col = divmod(i, cols)
            lbl_lib = QLabel(lib)
            lbl_lib.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            lbl_ver = QLabel(ver)
            # Mesma fonte branca em negrito usada nos valores de Hardware Specs — CPU/GPU.
            lbl_ver.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            lbl_ver.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sw_grid.addWidget(lbl_lib, row, col*2)
            sw_grid.addWidget(lbl_ver, row, col*2+1)
        sw_grp.set_collapsed(True)
        lay.addWidget(sw_grp)

        # ── Pipeline — Steps ───────────────────────────────────────────────
        pipe_grp = CollapsibleGroupBox()
        pipe_grp.setStyleSheet(_SS_GROUPBOX_NO_BG)
        self._tr("home_grp_pipeline", pipe_grp.setTitle)
        pipe_lay = QGridLayout(pipe_grp)
        pipe_lay.setHorizontalSpacing(20); pipe_lay.setVerticalSpacing(4)
        step_keys = [
            ("home_step1_name", "home_step1_desc"),
            ("home_step2_name", "home_step2_desc"),
            ("home_step3_name", "home_step3_desc"),
            ("home_step4_name", "home_step4_desc"),
            ("home_step5_name", "home_step5_desc"),
            ("home_step6_name", "home_step6_desc"),
            ("home_step7_name", "home_step7_desc"),
            ("home_step8_name", "home_step8_desc"),
        ]
        for i, (name_key, desc_key) in enumerate(step_keys):
            name_lbl = QLabel()
            # Mesma fonte branca em negrito usada nos valores de Hardware Specs — CPU/GPU.
            name_lbl.setStyleSheet("font-weight:bold;font-size:9pt;color:#C9D1D9;")
            self._tr(name_key, lambda txt, lbl=name_lbl: lbl.setText(f"<b>{txt}</b>"))
            desc_lbl = self._trL(desc_key)
            desc_lbl.setStyleSheet("color:#6E8CA8;font-size:9pt;")
            pipe_lay.addWidget(name_lbl, i, 0)
            pipe_lay.addWidget(desc_lbl, i, 1)
        pipe_grp.set_collapsed(True)
        lay.addWidget(pipe_grp)

        lay.addWidget(_mk_separator())

        # ── Botões de ação — sem ícones ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_start = self._mk_btn_tr("home_btn_start", _SS_BTN_PRIMARY, 160)
        btn_start.setFixedHeight(38)
        btn_start.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        btn_req = self._mk_btn_tr("home_btn_install", _SS_BTN_SECONDARY, 190)
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

        ver = QLabel(f"v{VERSAO}  ·  © 2024–2026 CEB/UFPR  ·  {LICENCA_CURTA}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("font-size:8pt;color:#6E8CA8;margin-top:6px;")
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

        lay.addWidget(self._mk_title_tr("title_step1"))
        lay.addWidget(_mk_separator())

        # ── Diretório raiz do CODEEP ───────────────────────────────────────
        grp_dir = QGroupBox(); self._tr("s1_grp_root", grp_dir.setTitle)
        grp_dir.setStyleSheet(_SS_GROUPBOX_NO_BG)
        dl = QVBoxLayout(grp_dir)
        self.ed_root = QLineEdit(self.dp_dir); self.ed_root.setReadOnly(True)
        dl.addWidget(self.ed_root)
        btn_br = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_br.clicked.connect(self._browse_root)
        dl.addWidget(btn_br, alignment=Qt.AlignLeft)

        # ── Identificação do Job ───────────────────────────────────────────
        grp_job = QGroupBox(); self._tr("s1_grp_job_id", grp_job.setTitle)
        grp_job.setStyleSheet(_SS_GROUPBOX_NO_BG)
        jf = QFormLayout(grp_job); jf.setHorizontalSpacing(20)
        self.ed_job_name = QLineEdit()
        self._tr("s1_ph_job_name", self.ed_job_name.setPlaceholderText)
        self.ed_job_name.setMaximumWidth(340)
        self.de_job_date = QDateEdit()
        self.de_job_date.setCalendarPopup(True)
        self.de_job_date.setDate(QDate.currentDate())
        self.de_job_date.setDisplayFormat("yyyy-MM-dd")
        self.de_job_date.setMaximumWidth(160)
        self._tr("s1_tooltip_date", self.de_job_date.setToolTip)
        self.lbl_job_id = QLabel("—")
        self.lbl_job_id.setStyleSheet("font-weight:bold;color:#2ECC71;")
        self.ed_job_name.textChanged.connect(self._update_job_id)
        self.de_job_date.dateChanged.connect(self._update_job_id)
        jf.addRow(self._trL("s1_lbl_job_name"), self.ed_job_name)
        jf.addRow(self._trL("s1_lbl_date"),     self.de_job_date)
        jf.addRow(self._trL("s1_lbl_job_id"),   self.lbl_job_id)

        # ── Termos de busca para Fases 2 e 3 ──────────────────────────────
        grp_search = QGroupBox(); self._tr("s1_grp_search", grp_search.setTitle)
        grp_search.setStyleSheet(_SS_GROUPBOX_NO_BG)
        sl = QVBoxLayout(grp_search); sl.setSpacing(6)

        sl.addWidget(self._mk_section_tr("s1_sec_org_terms"))
        self.te_org_terms = QTextEdit()
        self._tr("s1_ph_org_terms", self.te_org_terms.setPlaceholderText)
        self.te_org_terms.setFixedHeight(72)
        sl.addWidget(self.te_org_terms)

        sl.addWidget(self._mk_section_tr("s1_sec_desc_terms"))
        self.te_desc_terms = QTextEdit()
        self._tr("s1_ph_desc_terms", self.te_desc_terms.setPlaceholderText)
        self.te_desc_terms.setFixedHeight(72)
        sl.addWidget(self.te_desc_terms)

        sl.addWidget(self._mk_section_tr("s1_sec_p3_ids"))
        self.te_phase3_ids = QTextEdit()
        self._tr("s1_ph_p3_ids", self.te_phase3_ids.setPlaceholderText)
        self.te_phase3_ids.setFixedHeight(60)
        sl.addWidget(self.te_phase3_ids)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(grp_dir, 1)
        top_row.addWidget(grp_job, 1)
        lay.addLayout(top_row)

        # ── Detected Paths ─────────────────────────────────────────────────
        grp_paths = QGroupBox(); self._tr("s1_grp_paths", grp_paths.setTitle)
        grp_paths.setStyleSheet(_SS_GROUPBOX_NO_BG)
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
        _sep = QLabel()
        self._tr("s1_sep_global", lambda txt: _sep.setText(f"<b>{txt}</b>"))
        _sep.setStyleSheet("color:#2ECC71;font-size:9pt;margin-top:4px;")
        pf.addRow(_sep)
        for key, label, sub in global_paths:
            lbl = QLabel("—"); lbl.setStyleSheet("font-size:9pt;color:#C9D1D9;")
            lbl.setTextFormat(Qt.RichText)
            self._path_labels[key] = (lbl, sub, "global")
            pf.addRow(label, lbl)

        # Caminhos do job (Fases 2/3 — por job_id)
        _sep2 = QLabel()
        self._tr("s1_sep_job", lambda txt: _sep2.setText(f"<b>{txt}</b>"))
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

        second_row = QHBoxLayout()
        second_row.setSpacing(12)
        second_row.addWidget(grp_search, 1)
        second_row.addWidget(grp_paths, 1)
        lay.addLayout(second_row)

        # ── Botão e status ─────────────────────────────────────────────────
        self.btn_job_run = self._mk_btn_tr("s1_btn_create_job", _SS_BTN_PRIMARY, 230)
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
        nl.addLayout(self._mk_nav_row_tr(lambda: self.tabs.setCurrentIndex(0),
                                          lambda: self.tabs.setCurrentIndex(2)))
        outer.addWidget(nav)
        return w

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("dlg_select_root_title", self._idioma), self.dp_dir)
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
                    _ph = i18n.t("s1_placeholder_no_job", self._idioma)
                    lbl.setText(f'<span style="color:#BDC3C7;">{_ph}</span>')
                    continue
            lbl.setText(_fmt(full))

    def _job_run(self):
        root = self.ed_root.text().strip()
        name = self.ed_job_name.text().strip()
        if not root or not name:
            QMessageBox.warning(
                self, i18n.t("msg_warning_title", self._idioma),
                i18n.t("s1_msg_fill_root_name", self._idioma)); return

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
        self.lbl_job_status.setText(i18n.t("s1_msg_job_ready", self._idioma, job_dir=job_dir))
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
    def _build_pipeline_tab(self, title_key, params_widget, run_fn_getter,
                            worker_key, prev_idx, next_idx,
                            log_height=280):
        """Constrói o layout padrão de uma aba de pipeline.

        title_key     : chave i18n (ver MODULES/i18n.py) do título da aba
        params_widget : QWidget com os controles específicos da etapa
        run_fn_getter : callable() → (fn, args, kwargs) a executar
        """
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(30,20,30,20); lay.setSpacing(12)
        lay.addWidget(self._mk_title_tr(title_key))
        lay.addWidget(_mk_separator())

        # Painel de parâmetros (scroll)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(params_widget); scroll.setMaximumHeight(420)
        lay.addWidget(scroll)

        # Log de execução em tempo real
        lay.addWidget(self._mk_section_tr("sec_execution_log"))
        log_box = _mk_log_box(log_height)
        lay.addWidget(log_box)

        # Barra de progresso
        prog = _mk_progress(lambda: self._idioma); lay.addWidget(prog)
        # Reaplica o texto ocioso ("Progress: 0% — ETA: —") no idioma atual numa troca de
        # idioma — só quando ainda não rodou nada (value==0); depois disso o texto já é mantido
        # atualizado ao vivo por _connect_worker (eta_signal/finished_signal), então não deve ser
        # sobrescrito aqui.
        self._tr("progress_label", lambda _t: prog.setFormat(
            f"{i18n.t('progress_label', self._idioma)}: %p%  —  "
            f"{i18n.t('progress_eta_label', self._idioma)}: —"
        ) if prog.value() == 0 else None)

        # Botões de execução e parada
        btn_row = QHBoxLayout()
        btn_run  = self._mk_btn_tr("btn_run",  _SS_BTN_PRIMARY, 150)
        btn_stop = self._mk_btn_tr("btn_stop", _SS_BTN_DANGER,  100)
        btn_stop.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(btn_run); btn_row.addWidget(btn_stop)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addLayout(self._mk_nav_row_tr(
            lambda: self.tabs.setCurrentIndex(prev_idx),
            lambda: self.tabs.setCurrentIndex(next_idx),
        ))

        def _run():
            if not self.ed_root.text().strip():
                QMessageBox.warning(
                    w, i18n.t("msg_warning_title", self._idioma),
                    i18n.t("msg_please_start_job", self._idioma))
                return
            try:
                fn, args, kwargs = run_fn_getter()
            except Exception as e:
                QMessageBox.critical(w, i18n.t("msg_config_error_title", self._idioma), str(e)); return
            log_box.clear(); prog.setValue(0)
            btn_run.setEnabled(False); btn_stop.setEnabled(True)
            worker = PipelineWorker(fn, args, kwargs)
            _connect_worker(worker, log_box, prog, btn_run, btn_stop,
                            lambda ok: None, idioma_getter=lambda: self._idioma)
            self._workers[worker_key] = worker
            worker.start()

        btn_run.clicked.connect(_run)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — STEP 2: COLETA
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step2(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        # cb_dataset: nomes técnicos de dataset (identificadores usados literalmente pelo
        # pipeline_collect) — NÃO traduzidos de propósito, ver MODULES/i18n.py.
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(["lp-pdbbind", "rcsb-wide", "plinder", "moad"])

        # cb_phase: texto exibido traduzido via _mk_combo_tr; o VALOR usado pela lógica abaixo
        # vem de _COLLECT_PHASE_VALUES[currentIndex()] (dígitos "1"/"2"/"3"/"all", nunca
        # currentText() — que mudaria com o idioma).
        self.cb_phase = self._mk_combo_tr(["s2_phase1", "s2_phase2", "s2_phase3", "s2_phase_all"])
        self._COLLECT_PHASE_VALUES = ["1", "2", "3", "all"]
        self._tr("s2_tooltip_phase", self.cb_phase.setToolTip)

        self.sp_phase1_max = QSpinBox()
        self.sp_phase1_max.setRange(1000, 500000)
        self.sp_phase1_max.setValue(50000)

        self.ed_moad_dir = QLineEdit()
        self._tr("s2_ph_moad_dir", self.ed_moad_dir.setPlaceholderText)
        btn_moad = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_moad.clicked.connect(self._browse_moad_dir)
        moad_row = QWidget(); ml = QHBoxLayout(moad_row); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(self.ed_moad_dir); ml.addWidget(btn_moad)

        self.chk_status = QCheckBox(); self._tr("s2_chk_status", self.chk_status.setText)

        pf.addRow(self._trL("s2_lbl_dataset"), self.cb_dataset)
        pf.addRow(self._trL("s2_lbl_phase"), self.cb_phase)
        pf.addRow(self._trL("s2_lbl_max_entries"), self.sp_phase1_max)
        pf.addRow(self._trL("s2_lbl_moad_dir"), moad_row)
        pf.addRow("",                 self.chk_status)

        def run_fn():
            import MODULES.pipeline_collect as m
            if self.chk_status.isChecked():
                return m.show_status, (), {
                    "job_dir": self.job_dir or None
                }

            root        = self.ed_root.text().strip()
            job_dir     = self.job_dir or root
            phase       = self._COLLECT_PHASE_VALUES[self.cb_phase.currentIndex()]
            dataset     = self.cb_dataset.currentText()
            moad_dir    = self.ed_moad_dir.text().strip()
            phase1_max = self.sp_phase1_max.value()
            org_terms   = self._parse_text_list(self.te_org_terms.toPlainText())
            desc_terms  = self._parse_text_list(self.te_desc_terms.toPlainText())
            p3_ids      = self._parse_csv_ids(self.te_phase3_ids.toPlainText())

            wants_phase1 = phase in ("1", "all")
            wants_phase2 = phase in ("2", "all")
            wants_phase3 = phase in ("3", "all")

            # Validação Phase 2 — termos obrigatórios
            if wants_phase2 and not org_terms and not desc_terms:
                resp = QMessageBox.question(
                    params,
                    i18n.t("s2_dlg_phase2_title", self._idioma),
                    i18n.t("s2_dlg_phase2_body", self._idioma),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    raise ValueError(i18n.t("s2_err_phase2_cancelled", self._idioma))
                wants_phase2 = False

            # Validação Phase 3 — IDs obrigatórios
            if wants_phase3 and not p3_ids:
                resp = QMessageBox.question(
                    params,
                    i18n.t("s2_dlg_phase3_title", self._idioma),
                    i18n.t("s2_dlg_phase3_body", self._idioma),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    raise ValueError(i18n.t("s2_err_phase3_cancelled", self._idioma))
                wants_phase3 = False

            # Verificação final — pelo menos uma fase ativa após validações
            if not any([wants_phase1, wants_phase2, wants_phase3]):
                raise ValueError(i18n.t("s2_err_no_phase_active", self._idioma))

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
            "title_step2",
            params, run_fn, "collect", 1, 3
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — STEP 3: CURADORIA
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step3(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.sp_cur_workers = QSpinBox()
        self.sp_cur_workers.setRange(0,64); self.sp_cur_workers.setValue(0)
        self._tr("s3_tooltip_workers", self.sp_cur_workers.setToolTip)
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
        
        pf.addRow(self._trL("lbl_phase"), self.cb_cur_phase)
        pf.addRow(self._trL("lbl_workers_auto"),  self.sp_cur_workers)
        pf.addRow(self._trL("s3_lbl_cutoff"),     self.sp_cutoff)
        pf.addRow(self._trL("s3_lbl_min_site"),   self.sp_min_site)
        pf.addRow(self._trL("s3_lbl_max_mw"),     self.sp_max_mw)
        pf.addRow(self._trL("s3_lbl_min_heavy"),  self.sp_min_heavy)
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
            "title_step3",
            params, run_fn, "curate", 2, 4
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — STEP 4: EXPLORAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step4(self):
        params = QWidget(); pf = QVBoxLayout(params)
        lbl = self._trL("s4_lbl_desc")
        lbl.setStyleSheet("color:#6E8CA8;font-size:9pt;"); lbl.setWordWrap(True)
        pf.addWidget(lbl)

        # Botão para visualizar o dashboard gerado
        self.btn_view_dash = self._mk_btn_tr("s4_btn_view_dash", _SS_BTN_SECONDARY, 160)
        self.btn_view_dash.clicked.connect(self._view_explore_dashboard)

        pf.addWidget(self._trL("lbl_phase"))
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
            "title_step4",
            params, run_fn, "explore", 3, 5
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — STEP 5: VOXELIZAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step5(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.ed_vox_pair = QLineEdit()
        self._tr("s5_ph_pair", self.ed_vox_pair.setPlaceholderText)
        self.sp_vox_workers = QSpinBox()
        self.sp_vox_workers.setRange(0,16); self.sp_vox_workers.setValue(0)
        self._tr("s5_tooltip_vox_workers", self.sp_vox_workers.setToolTip)
        self.sp_grid = QSpinBox()
        self.sp_grid.setRange(16,64); self.sp_grid.setValue(32)
        self.sp_vox_pad = QDoubleSpinBox()
        self.sp_vox_pad.setRange(0.5,8.0); self.sp_vox_pad.setValue(3.0)
        self.sp_max_qm = QSpinBox()
        self.sp_max_qm.setRange(10,200); self.sp_max_qm.setValue(60)
        self._tr("s5_tooltip_max_qm", self.sp_max_qm.setToolTip)
        self.sp_vox_batch = QSpinBox()
        self.sp_vox_batch.setRange(10, 2000)
        self.sp_vox_batch.setValue(_BATCH_DEFAULT)
        self.sp_vox_batch.setSingleStep(50)
        # Tooltip com conteúdo dinâmico (preset detectado + status do CuPy) — o texto-modelo
        # (chave) é traduzido normalmente, mas "cupy_status" É ELE MESMO uma string traduzida,
        # então não pode ser um kwarg estático capturado uma vez no registro (ficaria preso no
        # idioma da 1ª chamada); recalculado a cada reaplicação dentro do apply_fn.
        def _apply_vox_batch_tooltip(_txt):
            cupy_key = "s5_cupy_available" if _CUPY_OK else "s5_cupy_unavailable"
            self.sp_vox_batch.setToolTip(i18n.t(
                "s5_tooltip_vox_batch", self._idioma,
                batch=_BATCH_DEFAULT, cupy_status=i18n.t(cupy_key, self._idioma),
            ))
        self._tr("s5_tooltip_vox_batch", _apply_vox_batch_tooltip)
        self.cb_qm_basis = QComboBox()
        self.cb_qm_basis.addItems(["sto-3g","3-21g","6-31g"])
        self._tr("s5_tooltip_qm_basis", self.cb_qm_basis.setToolTip)
        self.chk_ecfp4 = QCheckBox(); self._tr("s5_chk_ecfp4", self.chk_ecfp4.setText)
        self.cb_vox_phase = QComboBox()
        self.cb_vox_phase.addItems(["1", "2", "3"])

        # ── Protonação do sítio + método QM dos canais 3-4 (doador/aceptor) ──
        self.chk_protonate_site = QCheckBox()
        self._tr("s5_chk_protonate_site", self.chk_protonate_site.setText)
        self.chk_protonate_site.setChecked(True)
        self._tr("s5_tooltip_protonate_site", self.chk_protonate_site.setToolTip)
        self.dsb_site_ph = QDoubleSpinBox()
        self.dsb_site_ph.setDecimals(1); self.dsb_site_ph.setRange(0.0, 14.0)
        self.dsb_site_ph.setValue(7.4); self.dsb_site_ph.setSingleStep(0.5)
        self._tr("s5_tooltip_site_ph", self.dsb_site_ph.setToolTip)
        self.cb_site_qm_method = self._mk_combo_tr(
            ["qm_method_rhf_fragments", "qm_method_gfn_xtb", "qm_method_heuristic"]
        )
        self._tr("s5_tooltip_site_qm_method", self.cb_site_qm_method.setToolTip)
        self._SITE_QM_METHOD_VALUES = [
            "rhf_sto3g_fragments", "gfn_xtb_full", "heuristic",
        ]

        pf.addRow(self._trL("lbl_phase"), self.cb_vox_phase)
        pf.addRow(self._trL("s5_lbl_pair"),    self.ed_vox_pair)
        pf.addRow(self._trL("lbl_workers_auto"), self.sp_vox_workers)
        pf.addRow(self._trL("s5_lbl_grid"),   self.sp_grid)
        pf.addRow(self._trL("s5_lbl_padding"),      self.sp_vox_pad)
        pf.addRow(self._trL("s5_lbl_max_qm"),     self.sp_max_qm)
        pf.addRow(self._trL("s5_lbl_batch"),   self.sp_vox_batch)
        pf.addRow(self._trL("s5_lbl_qm_basis"),         self.cb_qm_basis)
        pf.addRow("",                  self.chk_ecfp4)
        pf.addRow("",                  self.chk_protonate_site)
        pf.addRow(self._trL("s5_lbl_site_ph"),          self.dsb_site_ph)
        pf.addRow(self._trL("s5_lbl_site_qm_method"), self.cb_site_qm_method)

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
            "title_step5",
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
        self.ed_resume = QLineEdit(); self._tr("s6_ph_resume", self.ed_resume.setPlaceholderText)
        self.ed_resume.setReadOnly(True)
        btn_resume = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        resume_row = QWidget(); rl = QHBoxLayout(resume_row); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self.ed_resume); rl.addWidget(btn_resume)
        btn_resume.clicked.connect(self._browse_resume)
        self.chk_freeze_enc = QCheckBox(); self._tr("s6_chk_freeze_enc", self.chk_freeze_enc.setText)

        self.ed_ckpt_out = QLineEdit(); self.ed_ckpt_out.setPlaceholderText("checkpoints/")
        self.ed_ckpt_out.setReadOnly(True)
        btn_ckpt = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        ckpt_row = QWidget(); cl = QHBoxLayout(ckpt_row); cl.setContentsMargins(0,0,0,0)
        cl.addWidget(self.ed_ckpt_out); cl.addWidget(btn_ckpt)
        btn_ckpt.clicked.connect(self._browse_ckpt_out)

        for label_key, widget in [
            ("lbl_phase",          self.cb_train_phase),
            ("s6_lbl_epochs",      self.sp_epochs),
            ("s6_lbl_batch",       self.sp_batch),
            ("s6_lbl_latent",      self.sp_latent),
            ("s6_lbl_hidden",      self.sp_hidden),
            ("s6_lbl_gnn_layers",  self.sp_gnn_layers),
            ("s6_lbl_lr",          self.dsb_lr),
            ("s6_lbl_beta_max",    self.dsb_beta),
            ("s6_lbl_beta_warmup", self.sp_warmup),
            ("s6_lbl_dropout",     self.dsb_dropout),
            ("s6_lbl_patience",    self.sp_patience),
            ("s6_lbl_device",      self.cb_train_device),
            ("s6_lbl_resume",      resume_row),
            (None,                 self.chk_freeze_enc),
            ("s6_lbl_ckpt_out",    ckpt_row),
        ]:
            pf.addRow(self._trL(label_key) if label_key else "", widget)

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
                raise ValueError(i18n.t(
                    "s6_err_tensor_not_found", self._idioma,
                    phase=phase, tensor_dir=tensor_dir))

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
            "title_step6",
            params, run_fn, "train", 5, 7, log_height=320
        )

    def _browse_resume(self):
        p, _ = QFileDialog.getOpenFileName(
            self, i18n.t("s6_dlg_select_best_model", self._idioma),
            self.ed_root.text(), "PyTorch (*.pt)")
        if p: self.ed_resume.setText(p)

    def _browse_ckpt_out(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("dlg_checkpoints_folder", self._idioma), self.ed_root.text())
        if d: self.ed_ckpt_out.setText(d)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 7 — STEP 7: AVALIAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step7(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)

        self.ed_eval_ckpt = QLineEdit(); self.ed_eval_ckpt.setReadOnly(True)
        self._tr("ph_checkpoints_default", self.ed_eval_ckpt.setPlaceholderText)
        btn_eval_ckpt = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_eval_ckpt.clicked.connect(self._browse_eval_ckpt)
        eck_row = QWidget(); el = QHBoxLayout(eck_row); el.setContentsMargins(0,0,0,0)
        el.addWidget(self.ed_eval_ckpt); el.addWidget(btn_eval_ckpt)
        self.chk_do_generate = QCheckBox()
        self._tr("s7_chk_do_generate", self.chk_do_generate.setText)

        pf.addRow(self._trL("s7_lbl_checkpoints"), eck_row)
        pf.addRow("",             self.chk_do_generate)

        # Botões de visualização de figuras — a chave (não o texto) é capturada na lambda, para
        # que o título repassado a _view_eval_fig reflita o idioma ATUAL no momento do clique,
        # não o idioma vigente quando o botão foi criado.
        vis_row = QWidget(); vl = QHBoxLayout(vis_row); vl.setContentsMargins(0,0,0,0)
        for name_key, stem in [("s7_btn_training_curves", "training_curves"),
                                ("s7_btn_performance_report", "performance_report")]:
            b = self._mk_btn_tr(name_key, _SS_BTN_SECONDARY, 180)
            b.clicked.connect(
                lambda _, s=stem, k=name_key: self._view_eval_fig(s, i18n.t(k, self._idioma))
            )
            vl.addWidget(b)
        pf.addRow(self._trL("s7_lbl_figures"), vis_row)

        self.cb_eval_phase = QComboBox()
        self.cb_eval_phase.addItems(["1", "2", "3"])
        pf.insertRow(0, self._trL("lbl_phase"), self.cb_eval_phase)
        
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
            "title_step7",
            params, run_fn, "evaluate", 6, 8
        )

    def _browse_eval_ckpt(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("dlg_checkpoints_folder", self._idioma), self.ed_root.text())
        if d: self.ed_eval_ckpt.setText(d)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 8 — STEP 8: GERAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    def _tab_step8(self):
        params = QWidget(); pf = QFormLayout(params); pf.setSpacing(10)
        self.cb_gen_phase = QComboBox()
        self.cb_gen_phase.addItems(["1", "2", "3"])
        self.ed_gen_pdb = QLineEdit(); self.ed_gen_pdb.setReadOnly(True)
        self._tr("s8_ph_pdb", self.ed_gen_pdb.setPlaceholderText)
        btn_pdb = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_pdb.clicked.connect(self._browse_gen_pdb)
        pdb_row = QWidget(); pl = QHBoxLayout(pdb_row); pl.setContentsMargins(0,0,0,0)
        pl.addWidget(self.ed_gen_pdb); pl.addWidget(btn_pdb)

        self.ed_gen_site_pdb = QLineEdit(); self.ed_gen_site_pdb.setReadOnly(True)
        self._tr("s8_ph_site_pdb", self.ed_gen_site_pdb.setPlaceholderText)
        btn_site_pdb = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_site_pdb.clicked.connect(self._browse_gen_site_pdb)
        site_pdb_row = QWidget(); spl = QHBoxLayout(site_pdb_row); spl.setContentsMargins(0,0,0,0)
        spl.addWidget(self.ed_gen_site_pdb); spl.addWidget(btn_site_pdb)

        self.ed_gen_ckpt = QLineEdit(); self.ed_gen_ckpt.setReadOnly(True)
        self._tr("ph_checkpoints_default", self.ed_gen_ckpt.setPlaceholderText)
        btn_gc = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_gc.clicked.connect(self._browse_gen_ckpt)
        gc_row = QWidget(); gl2 = QHBoxLayout(gc_row); gl2.setContentsMargins(0,0,0,0)
        gl2.addWidget(self.ed_gen_ckpt); gl2.addWidget(btn_gc)

        self.ed_gen_out = QLineEdit(); self.ed_gen_out.setReadOnly(True)
        self._tr("s8_ph_out", self.ed_gen_out.setPlaceholderText)
        btn_gout = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
        btn_gout.clicked.connect(self._browse_gen_out)
        gout_row = QWidget(); gl3 = QHBoxLayout(gout_row); gl3.setContentsMargins(0,0,0,0)
        gl3.addWidget(self.ed_gen_out); gl3.addWidget(btn_gout)

        self.ed_ligand_ref = QLineEdit(); self.ed_ligand_ref.setReadOnly(True)
        self._tr("s8_ph_ligand_ref", self.ed_ligand_ref.setPlaceholderText)
        btn_ref = self._mk_btn_tr("btn_browse", _SS_BTN_SECONDARY, 100)
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
        # NOTA (ver resolve_site_config_overrides em pipeline_generate.py):
        # desde que o checkpoint passou a gravar a config de voxelização usada
        # no treino (best_model.pt["config"]["voxel_config"]), estes campos
        # só valem como FALLBACK — usados de fato apenas (a) para checkpoints
        # treinados antes desta funcionalidade existir, ou (b) para qualquer
        # chave que o checkpoint não tenha gravado. Quando o checkpoint tem
        # um valor gravado, ele sempre vence sobre a escolha manual aqui (com
        # aviso no log se os dois divergirem) — não é mais possível gerar
        # com um mismatch treino↔inferência por engano nestes 4 campos.
        self.chk_gen_protonate_site = QCheckBox()
        self._tr("s8_chk_protonate_site", self.chk_gen_protonate_site.setText)
        self.chk_gen_protonate_site.setChecked(True)
        self.dsb_gen_ph = QDoubleSpinBox()
        self.dsb_gen_ph.setDecimals(1); self.dsb_gen_ph.setRange(0.0, 14.0)
        self.dsb_gen_ph.setValue(7.4); self.dsb_gen_ph.setSingleStep(0.5)
        self.cb_gen_site_qm_method = self._mk_combo_tr(
            ["qm_method_rhf_fragments", "qm_method_gfn_xtb", "qm_method_heuristic"]
        )
        self._tr("s8_tooltip_gen_qm_method", self.cb_gen_site_qm_method.setToolTip)
        self._GEN_SITE_QM_METHOD_VALUES = [
            "rhf_sto3g_fragments", "gfn_xtb_full", "heuristic",
        ]
        # qm_basis (base do RHF/STO-3G por resíduo) — antes desta correção
        # não existia NENHUM controle equivalente na Etapa 8: a geração
        # sempre usava o default "sto-3g" do módulo, mesmo que o treino
        # tivesse usado 3-21g/6-31g, sem nenhum aviso. Continua valendo só
        # como fallback (mesma regra acima).
        self.cb_gen_qm_basis = QComboBox()
        self.cb_gen_qm_basis.addItems(["sto-3g", "3-21g", "6-31g"])
        self._tr("s8_tooltip_gen_qm_basis", self.cb_gen_qm_basis.setToolTip)

        # ── FF de geometria final (minimização conjunta ligante+sítio) ───────
        self.cb_final_ff = self._mk_combo_tr(["ff_mmff_uff", "ff_amber_ff14sb", "ff_charmm36"])
        self.cb_final_ff.setCurrentIndex(0)   # MMFF/UFF — mais simples/rápido, pré-selecionado
        self._tr("s8_tooltip_final_ff", self.cb_final_ff.setToolTip)
        self._FINAL_FF_VALUES = ["mmff_uff", "amber_ff14sb", "charmm36"]

        # ── Induced-fit: backbone travado, cadeia lateral livre ──────────────
        self.chk_induced_fit = QCheckBox()
        self._tr("s8_chk_induced_fit", self.chk_induced_fit.setText)
        self.chk_induced_fit.setChecked(False)   # desmarcado por padrão — sítio rígido
        self._tr("s8_tooltip_induced_fit", self.chk_induced_fit.setToolTip)

        for label_key, widget in [
            ("s8_lbl_model_phase",   self.cb_gen_phase),
            ("s8_lbl_protein_pdb",   pdb_row),
            ("s8_lbl_site_pdb",      site_pdb_row),
            ("s7_lbl_checkpoints",   gc_row),
            ("s8_lbl_output",        gout_row),
            ("s8_lbl_ref_ligand",    ref_row),
            ("s8_lbl_n_samples",     self.sp_n_samples),
            ("s8_lbl_temperature",   self.dsb_temp),
            ("s8_lbl_cutoff",        self.sp_cutoff_gen),
            ("s8_lbl_top_n",         self.sp_top_n),
            ("s8_lbl_min_atoms",     self.sp_min_atoms_gen),
            (None,                   self.chk_gen_protonate_site),
            ("s5_lbl_site_ph",       self.dsb_gen_ph),
            ("s5_lbl_site_qm_method", self.cb_gen_site_qm_method),
            ("s5_lbl_qm_basis",      self.cb_gen_qm_basis),
            ("s8_lbl_final_ff",      self.cb_final_ff),
            (None,                   self.chk_induced_fit),
        ]:
            pf.addRow(self._trL(label_key) if label_key else "", widget)

        def run_fn():
            pdb = self.ed_gen_pdb.text().strip()
            if not pdb or not os.path.isfile(pdb):
                raise ValueError(i18n.t("s8_err_invalid_pdb", self._idioma))
            site_pdb = self.ed_gen_site_pdb.text().strip() or None
            if site_pdb and not os.path.isfile(site_pdb):
                raise ValueError(i18n.t("s8_err_invalid_site_pdb", self._idioma))
            root   = self.ed_root.text().strip()
            phase = self.cb_gen_phase.currentText()
            if phase == "1":
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(root) / "checkpoints" / "phase1")
            elif phase == "2":
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(self.job_dir) / "checkpoints" / "phase2")
            else:
                ckpt = self.ed_gen_ckpt.text().strip() or str(Path(self.job_dir) / "checkpoints" / "phase3")

            if not (Path(ckpt) / "best_model.pt").exists():
                raise ValueError(i18n.t("s8_err_no_checkpoint", self._idioma, phase=phase))
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
                    qm_basis       = self.cb_gen_qm_basis.currentText(),
                ),
                final_ff = self._FINAL_FF_VALUES[self.cb_final_ff.currentIndex()],
                induced_fit = self.chk_induced_fit.isChecked(),
            )

        return self._build_pipeline_tab(
            "title_step8",
            params, run_fn, "generate", 7, 0, log_height=260
        )

    def _browse_gen_pdb(self):
        p, _ = QFileDialog.getOpenFileName(
            self, i18n.t("s8_dlg_select_pdb", self._idioma), self.ed_root.text(), "PDB (*.pdb)")
        if p: self.ed_gen_pdb.setText(p)

    def _browse_gen_site_pdb(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            i18n.t("s8_dlg_select_site_pdb", self._idioma),
            self.ed_root.text(),
            "PDB (*.pdb)",
        )
        if p: self.ed_gen_site_pdb.setText(p)

    def _browse_gen_ckpt(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("dlg_checkpoints_folder", self._idioma), self.ed_root.text())
        if d: self.ed_gen_ckpt.setText(d)

    def _browse_moad_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("dlg_moad_folder", self._idioma), self.ed_root.text())
        if d: self.ed_moad_dir.setText(d)

    def _browse_gen_out(self):
        d = QFileDialog.getExistingDirectory(
            self, i18n.t("s8_dlg_output_folder", self._idioma), self.ed_root.text())
        if d: self.ed_gen_out.setText(d)

    def _browse_ligand_ref(self):
        p, _ = QFileDialog.getOpenFileName(
            self, i18n.t("s8_dlg_reference_ligand", self._idioma),
            self.ed_root.text(), "Ligand files (*.sdf *.pdb *.mol2)")
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
        QMessageBox.information(
            self, i18n.t("msg_figure_not_found", self._idioma),
            i18n.t("msg_run_step_first", self._idioma, fig_dir=fig_dir))

    def _view_fig_path(self, path: str, title: str):
        """Exibe uma figura dado o caminho absoluto."""
        dlg = QDialog(self); dlg.setWindowTitle(title); dlg.resize(920, 640)
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea(); lbl = QLabel()
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(
                self, i18n.t("msg_error_title", self._idioma),
                i18n.t("msg_could_not_load", self._idioma, path=path)); return
        lbl.setPixmap(pix.scaled(880, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl); scroll.setWidgetResizable(True)
        lay.addWidget(scroll)
        br = QHBoxLayout()
        b_open = _mk_btn(i18n.t("btn_open_in_system", self._idioma), _SS_BTN_PRIMARY, 160)
        b_open.clicked.connect(lambda: subprocess.Popen(["xdg-open", path]))
        b_close = _mk_btn(i18n.t("btn_close", self._idioma), _SS_BTN_PRIMARY, 100)
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
                self._view_fig_path(
                    path, i18n.t("msg_summary_dashboard_title", self._idioma, phase=phase))
                return

        QMessageBox.information(
            self, i18n.t("msg_figure_not_found", self._idioma),
            i18n.t("msg_run_step4_first", self._idioma, phase=phase, fig_dir=fig_dir)
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
                self._view_fig_path(
                    str(path),
                    i18n.t("msg_fig_title_with_phase", self._idioma, title=title, phase=phase))
                return

        QMessageBox.information(
            self, i18n.t("msg_figure_not_found", self._idioma),
            i18n.t("msg_run_step7_first", self._idioma, phase=phase, fig_dir=fig_dir)
        )

    def _show_requirements(self):
        if not _REQ_MODULE_OK:
            QMessageBox.warning(
                self, i18n.t("msg_module_not_found_title", self._idioma),
                i18n.t("msg_module_not_found_body", self._idioma)); return
        self._req_win = RequirementsInstaller(None)
        self._req_win.setWindowModality(Qt.ApplicationModal)
        self._req_win.show(); self._req_win.activateWindow(); self._req_win.raise_()

    def _show_about(self):
        QMessageBox.about(
            self,
            i18n.t("about_title", self._idioma),
            f"{i18n.t('about_body', self._idioma)}\n\nVersion: {VERSAO}")


# ==========================================================================================================================================
# ═══════════════════════ PONTO DE ENTRADA ═════════════════════════════════
# ==========================================================================================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("CODEEP")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
