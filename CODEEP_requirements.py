# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# CODEEP – Deep Learning Pipeline for Ligand Generation
# Módulo: Gerenciamento de Requisitos e Instalação
# Copyright (C) 2024–2026 Moisés Maia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
CODEEP_requirements.py
======================
Módulo responsável por:
  1. Verificar e instalar PyQt5 antes da UI iniciar (bootstrap pré-Qt).
  2. Detectar hardware (CPU, GPU/CUDA, RAM, Disco).
  3. Selecionar versões corretas de pacotes com base no hardware.
  4. Fornecer a classe RequirementsInstaller (janela PyQt5) com:
     - Verificação de cada pacote (✅ instalado / ⚠️ ausente).
     - Campos de versão editáveis.
     - Instalação via pip no venv ~/.venv/CODEEP.
     - Log em tempo real + barra de progresso.
"""

from __future__ import annotations
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, List


# ==========================================================================
# ─── HELPERS GENÉRICOS ────────────────────────────────────────────────────
# ==========================================================================

def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs
    )


def _which(p: str) -> Optional[str]:
    return shutil.which(p)


def _pkg_installed(import_name: str) -> bool:
    """Verifica se um pacote é importável no Python atual."""
    return importlib.util.find_spec(import_name) is not None


def _pkg_version(import_name: str) -> Optional[str]:
    """Retorna a versão instalada de um pacote, ou None.

    Suprime UserWarnings emitidos durante o import (ex: aviso do gpu4pyscf
    sobre cuTENSOR), pois são irrelevantes no contexto de verificação de versão.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


# ==========================================================================
# ─── BOOTSTRAP PRÉ-QT: GARANTIR PYQT5 ────────────────────────────────────
# ==========================================================================

def bootstrap_pyqt5(interactive: bool = True) -> bool:
    """
    Versão standalone exportável do bootstrap.
    Garante que PyQt5 esteja disponível NO VENV ~/.venv/CODEEP.
    Usado quando CODEEP_requirements.py é executado diretamente.
    """
    venv_python = Path.home() / ".venv" / "CODEEP" / "bin" / "python"

    # Já está rodando no venv — PyQt5 deve estar lá
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return True
    except Exception:
        pass

    # Verifica se PyQt5 já está no venv
    check = subprocess.run(
        [str(venv_python), "-c", "import PyQt5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) if venv_python.is_file() else None

    if check and check.returncode == 0:
        return True  # já instalado no venv

    msg = (
        "\n[CODEEP] PyQt5 não encontrado no venv ~/.venv/CODEEP.\n"
        f"  Python atual: {sys.executable}\n"
    )
    print(msg)

    if interactive:
        try:
            resp = input("  Deseja instalar PyQt5 no venv agora? [S/n]: ").strip().lower()
            if resp in ("n", "nao", "não", "no"):
                print("[CODEEP] Instalação cancelada.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n[CODEEP] Cancelado pelo usuário.")
            return False

    # Cria o venv se necessário
    p = ensure_venv("CODEEP")

    print("[CODEEP] Instalando PyQt5 no venv...")
    res = _run([p["python"], "-m", "pip", "install", "--upgrade", "PyQt5"])
    if res.returncode == 0:
        print("[CODEEP] ✅ PyQt5 instalado com sucesso no venv!\n")
        print(f"[CODEEP] Execute novamente com: {p['python']} {' '.join(sys.argv)}")
        return True
    else:
        lines = [l for l in res.stdout.splitlines()
                 if any(k in l for k in ("ERROR","error","Failed"))]
        print(f"[CODEEP] ❌ Falha ao instalar PyQt5:")
        for l in lines[:5]: print(f"   {l}")
        return False


# ==========================================================================
# ─── VENV ─────────────────────────────────────────────────────────────────
# ==========================================================================

def venv_paths(env_name: str = "CODEEP") -> dict:
    """Retorna caminhos do venv dedicado ao CODEEP."""
    venv_dir = Path.home() / ".venv" / env_name
    return {
        "venv_dir": str(venv_dir),
        "python":   str(venv_dir / "bin" / "python"),
        "pip":      str(venv_dir / "bin" / "pip"),
    }


def ensure_venv(env_name: str = "CODEEP") -> dict:
    """Cria o venv ~/.venv/CODEEP se não existir. Retorna venv_paths.

    Usa sempre o Python do sistema (não sys.executable) para criar o venv,
    evitando recursão caso o script já esteja rodando de dentro de outro venv.
    """
    p = venv_paths(env_name)
    venv_dir = Path(p["venv_dir"])

    if not venv_dir.is_dir() or not Path(p["python"]).is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)

        # Usa o Python do sistema para criar o venv, não o atual (pode ser venv)
        system_python = _find_system_python()
        res = _run([system_python, "-m", "venv", str(venv_dir)])
        if res.returncode != 0:
            raise RuntimeError(
                f"Falha ao criar venv com {system_python}:\n{res.stdout}\n"
                "Verifique se 'python3-venv' está instalado:\n"
                "  sudo apt install python3-venv"
            )
        # Atualiza pip/setuptools/wheel no venv
        _run([p["python"], "-m", "pip", "install", "--upgrade",
              "pip", "setuptools", "wheel"])

    return p


def _find_system_python() -> str:
    """Localiza o Python do sistema (fora de qualquer venv)."""
    import shutil as _shutil
    # Se o Python atual NÃO está num venv, usa ele diretamente
    if not (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
        return sys.executable
    # Caso contrário, procura python3 no PATH do sistema
    for candidate in ("python3", "python3.12", "python3.11",
                      "python3.10", "python3.9", "python"):
        found = _shutil.which(candidate)
        if found:
            return found
    return sys.executable  # fallback


def _pip_install(python_bin: str, pkgs: List[str]) -> subprocess.CompletedProcess:
    return _run([python_bin, "-m", "pip", "install"] + pkgs)


# ==========================================================================
# ─── DETECÇÃO DE HARDWARE ─────────────────────────────────────────────────
# ==========================================================================

def detect_python() -> dict:
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    ok = ver >= (3, 9)
    return {
        "version": version_str,
        "ok": ok,
        "msg": version_str if ok else f"Python {version_str} — necessário >= 3.9",
    }


def detect_cpu() -> dict:
    info = {
        "brand":          platform.processor() or "Desconhecido",
        "arch":           platform.machine(),
        "cores_logical":  os.cpu_count() or 1,
        "cores_physical": 1,
    }
    try:
        if platform.system() == "Linux":
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "Core(s) per socket" in line:
                    cores = int(line.split(":")[1].strip())
                    sockets_lines = [l for l in out.splitlines() if "Socket(s)" in l]
                    sockets = int(sockets_lines[0].split(":")[1].strip()) if sockets_lines else 1
                    info["cores_physical"] = cores * sockets
                if "Model name" in line:
                    info["brand"] = line.split(":")[1].strip()
    except Exception:
        info["cores_physical"] = max(1, info["cores_logical"] // 2)
    return info


def detect_gpu() -> dict:
    result = {
        "available": False, "name": None, "vram_gb": 0,
        "driver_version": None, "cuda_version": None,
        "cuda_major": 0, "cuda_minor": 0,
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        result.update({
            "available":      True,
            "name":           parts[0],
            "vram_gb":        round(int(parts[1]) / 1024, 1),
            "driver_version": parts[2],
        })
    except Exception:
        return result

    # CUDA via nvcc
    try:
        nvcc_out = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        m = re.search(r"release (\d+)\.(\d+)", nvcc_out)
        if m:
            result["cuda_major"] = int(m.group(1))
            result["cuda_minor"] = int(m.group(2))
            result["cuda_version"] = f"{result['cuda_major']}.{result['cuda_minor']}"
    except Exception:
        # Inferência pelo driver
        try:
            major = int(result["driver_version"].split(".")[0])
            if   major >= 570: result["cuda_major"], result["cuda_minor"] = 12, 8
            elif major >= 545: result["cuda_major"], result["cuda_minor"] = 12, 3
            elif major >= 530: result["cuda_major"], result["cuda_minor"] = 12, 1
            elif major >= 520: result["cuda_major"], result["cuda_minor"] = 11, 8
            result["cuda_version"] = f"{result['cuda_major']}.{result['cuda_minor']}"
            result["cuda_inferred"] = True
        except Exception:
            pass
    return result


def detect_ram() -> dict:
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1])
            total_gb = round(mem.get("MemTotal",     0) / 1e6, 1)
            free_gb  = round(mem.get("MemAvailable", 0) / 1e6, 1)
            return {"total_gb": total_gb, "free_gb": free_gb, "ok": total_gb >= 8}
    except Exception:
        pass
    return {"total_gb": 0, "free_gb": 0, "ok": True}


def detect_disk(path: str = None) -> dict:
    path = path or str(Path.home())
    try:
        stat = shutil.disk_usage(path)
        return {
            "total_gb": round(stat.total / 1e9, 1),
            "free_gb":  round(stat.free  / 1e9, 1),
            "ok":       stat.free / 1e9 >= 10,
        }
    except Exception:
        return {"total_gb": 0, "free_gb": 0, "ok": False}


def detect_hardware() -> dict:
    """Roda todas as detecções e retorna dict consolidado."""
    return {
        "python": detect_python(),
        "cpu":    detect_cpu(),
        "gpu":    detect_gpu(),
        "ram":    detect_ram(),
        "disk":   detect_disk(),
    }


def select_torch_variant(gpu: dict) -> dict:
    """Seleciona index_url e tag corretos para PyTorch baseado na GPU."""
    cuda = gpu.get("cuda_major", 0)
    minor = gpu.get("cuda_minor", 0)
    if gpu["available"] and cuda >= 12:
        if minor >= 6:
            return {"index_url": "https://download.pytorch.org/whl/cu128", "tag": "cu128",
                    "version": "torch==2.7.0"}
        return {"index_url": "https://download.pytorch.org/whl/cu121", "tag": "cu121",
                "version": "torch==2.5.1"}
    elif gpu["available"] and cuda == 11:
        return {"index_url": "https://download.pytorch.org/whl/cu118", "tag": "cu118",
                "version": "torch==2.4.0"}
    return {"index_url": "https://download.pytorch.org/whl/cpu", "tag": "cpu",
            "version": "torch==2.7.0"}


# ==========================================================================
# ─── CATÁLOGO DE PACOTES DO CODEEP ────────────────────────────────────────
# ==========================================================================

# Cada entrada: (display_name, import_name, default_version_spec, description, is_critical)
PACKAGES_CATALOG: list[tuple] = [
    # ── Required (core pipeline) ──────────────────────────────────────────
    ("PyQt5",                "PyQt5",           "PyQt5==5.15.11",
     "CODEEP graphical user interface.",                              True),
    ("PyTorch",              "torch",            "torch==2.7.0",
     "Deep learning framework. Version adjusted to hardware.",       True),
    ("torch-geometric",      "torch_geometric",  "torch-geometric",
     "Graph Neural Networks on top of PyTorch.",                     True),
    ("NumPy",                "numpy",            "numpy>=1.24",
     "Numerical operations and tensor handling.",                    True),
    ("Pandas",               "pandas",           "pandas>=2.0",
     "DataFrame and CSV manipulation.",                              True),
    ("SciPy",                "scipy",            "scipy>=1.10",
     "Scientific algorithms and optimization.",                      True),
    ("Matplotlib",           "matplotlib",       "matplotlib>=3.7",
     "Plotting and figure generation.",                              True),
    ("Seaborn",              "seaborn",          "seaborn>=0.13",
     "Statistical data visualization.",                              False),
    ("scikit-learn",         "sklearn",          "scikit-learn>=1.4",
     "ML metrics and auxiliary models.",                             True),
    ("Optuna",               "optuna",           "optuna>=3.6",
     "Hyperparameter optimization (Step 6). Optional: only Step 6 "
     "needs it.",                                                   False),
    ("RDKit",                "rdkit",            "rdkit",
     "Cheminformatics: SMILES, descriptors, filters.",               True),
    ("BioPython",            "Bio",              "biopython>=1.80",
     "PDB structure and FASTA sequence handling.",                   True),
    # ── Optional (GPU acceleration / QM) ─────────────────────────────────
    ("PySCF",                "pyscf",            "pyscf",
     "QM / quantum chemistry calculations (CPU). Fallback for GPU4PySCF.", False),
    ("GPU4PySCF",            "gpu4pyscf",        "gpu4pyscf-cuda12x",
     "GPU-accelerated QM calculations (CUDA 12).",                   False),
    ("CuPy",                 "cupy",             "cupy-cuda12x==13.3.0",
     "NumPy-compatible GPU array operations (CUDA 12).",             False),
    ("cuTENSOR (NVIDIA)",    "cutensor",         "cutensor-cu12",
     "Native NVIDIA tensor contraction engine. Improves performance "
     "in QM calculations with large tensors.",                       False),
    # ── Utilities ─────────────────────────────────────────────────────────
    ("tqdm",                 "tqdm",             "tqdm>=4.65",
     "Progress bars for CLI scripts.",                               False),
    ("psutil",               "psutil",           "psutil",
     "CPU and RAM monitoring.",                                      False),
    ("requests",             "requests",         "requests>=2.28",
     "HTTP requests and file downloads.",                            False),
    ("openpyxl",             "openpyxl",         "openpyxl",
     "Excel file read/write support.",                               False),
]


# ==========================================================================
# ─── JANELA DE INSTALAÇÃO (RequirementsInstaller) ─────────────────────────
# ==========================================================================

# Importação PyQt5 — só alcançável após bootstrap
from PyQt5.QtCore    import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QCheckBox, QPushButton, QTextEdit, QApplication,
    QMessageBox, QProgressBar, QLineEdit, QScrollArea,
    QGroupBox, QFrame, QSizePolicy,
)
from PyQt5.QtGui import QFont, QColor


# ── Estilos — dark mode (mesma paleta do CODEEP.py) ───────────────────────
_SS_WIN = """
QWidget {
    background: #0D1B2A;
    font-family: 'DejaVu Sans', 'Segoe UI', sans-serif;
    font-size: 10pt;
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
QLineEdit {
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    padding: 3px 6px;
    background: #1C3249;
    color: #C9D1D9;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
}
QLineEdit:focus { border: 1.5px solid #2ECC71; }
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
QScrollArea, QScrollArea > QWidget > QWidget { background: #0D1B2A; border: none; }
QScrollBar:vertical {
    background: #0D1B2A; width: 10px; border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2A4A6B; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #2ECC71; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar {
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    text-align: center;
    background: #1C3249;
    color: #C9D1D9;
    height: 20px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #27AE60, stop:1 #2ECC71);
    border-radius: 4px;
}
QFrame[frameShape="4"] { color: #2A4A6B; }
QToolTip {
    background: #1C3249;
    color: #C9D1D9;
    border: 1px solid #2A4A6B;
    border-radius: 4px;
    padding: 4px;
}
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

_SS_BTN_INSTALL = """
QPushButton {
    background: #1A4A30;
    color: #2ECC71;
    font-weight: bold;
    border: 1px solid #2ECC71;
    border-radius: 5px;
    padding: 7px 20px;
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
    padding: 7px 20px;
}
QPushButton:hover   { background: #37536C; color: #FFFFFF; }
QPushButton:pressed { background: #1C3249; color: #FFF; }
QPushButton:disabled{ background: #1B2730; color: #4F6577; border-color: #344654; }
"""

_SS_BTN_CLOSE = """
QPushButton {
    background: #4A1A1A;
    color: #E74C3C;
    font-weight: bold;
    border: 1px solid #E74C3C;
    border-radius: 5px;
    padding: 7px 20px;
}
QPushButton:hover   { background: #E74C3C; color: #FFF; }
QPushButton:pressed { background: #C0392B; color: #FFF; }
"""


# ==========================================================================
# ─── FERRAMENTAS EXTERNAS (binários, fora do pip) ─────────────────────────
# ==========================================================================
#
# Nem toda dependência do CODEEP é um pacote Python. A busca heurística de
# sítio da Etapa 9 usa o P2Rank, que é um programa Java distribuído como
# tarball. Aqui ele é tratado como cidadão de primeira classe do instalador,
# em vez de exigir que o usuário siga instruções de linha de comando — que é
# exatamente a barreira que o CODEEP existe para remover.
#
# O P2Rank é instalado dentro de <projeto>/tools/, sem privilégio de root.
# O Java NÃO é instalado automaticamente: exige apt/sudo, e um instalador
# gráfico que pede senha de root para mexer em pacotes do sistema é um risco
# desproporcional. Ele é detectado e, se ausente ou anterior ao 17, a linha
# de comando é exibida para o usuário executar.

P2RANK_VERSION = "2.5.1"
P2RANK_URL_TMPL = ("https://github.com/rdk/p2rank/releases/download/"
                   "{v}/p2rank_{v}.tar.gz")
JAVA_MIN_MAJOR = 17
JAVA_INSTALL_HINT = "sudo apt install openjdk-17-jre-headless"

# xtb: binário ESTÁTICO (`ldd` confirma: "não é um executável dinâmico"),
# sem nenhuma dependência de sistema — mais simples que o P2Rank, que ao
# menos precisa de Java. Usado pela voxelização (Etapa 5) como nível de
# teoria semi-empírico (GFN-xTB) para os canais doador/aceptor de elétrons
# do sítio; sem ele, cai automaticamente na aproximação heurística.
XTB_VERSION = "6.7.1"
XTB_URL_TMPL = ("https://github.com/grimme-lab/xtb/releases/download/"
                "v{v}/xtb-{v}-linux-x86_64.tar.xz")

_PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = _PROJECT_ROOT / "tools"


def detect_java() -> dict:
    """Versão do Java disponível e se atende ao mínimo exigido pelo P2Rank."""
    java = shutil.which("java")
    if not java:
        return {"found": False, "version": None, "major": None, "ok": False}
    try:
        out = subprocess.run([java, "-version"], capture_output=True,
                             text=True, timeout=5)
        txt = (out.stderr or out.stdout or "").splitlines()
        ver = txt[0].split('"')[1] if txt and '"' in txt[0] else "?"
    except Exception:
        ver = "?"
    try:
        major = int(ver.split(".")[0])
    except (ValueError, IndexError):
        major = None
    return {"found": True, "version": ver, "major": major,
            "ok": major is not None and major >= JAVA_MIN_MAJOR}


def detect_p2rank() -> Optional[str]:
    """Caminho do executável `prank`, se instalado.

    Mesma ordem de busca de pipeline_generate._p2rank_bin, para que o que o
    instalador mostra seja o que o pipeline vai encontrar.
    """
    env_home = os.environ.get("P2RANK_HOME")
    if env_home:
        cand = Path(env_home) / "prank"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which("prank")
    if found:
        return found
    for base in (TOOLS_DIR, Path.home(), Path("/opt")):
        if not base.is_dir():
            continue
        try:
            for cand in sorted(base.glob("p2rank*/prank")):
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)
        except OSError:
            continue
    return None


def install_p2rank(version: str = P2RANK_VERSION, log=print) -> bool:
    """Baixa e extrai o P2Rank em <projeto>/tools/. Não requer root."""
    import tarfile
    import urllib.request

    url = P2RANK_URL_TMPL.format(v=version)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tarball = TOOLS_DIR / f"p2rank_{version}.tar.gz"
    try:
        log(f"$ download {url}")
        urllib.request.urlretrieve(url, tarball)
        log(f"$ tar -xzf {tarball.name} -C {TOOLS_DIR}")
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(TOOLS_DIR)
        tarball.unlink(missing_ok=True)
    except Exception as exc:
        log(f"FAILED  P2Rank download/extract: {exc}")
        return False

    prank = detect_p2rank()
    if not prank:
        log("FAILED  P2Rank extracted but 'prank' not found in tools/")
        return False
    try:
        os.chmod(prank, 0o755)
    except OSError:
        pass

    java = detect_java()
    if not java["ok"]:
        atual = java["version"] or "ausente"
        log(f"WARNING  P2Rank instalado em {prank}, mas o Java disponível é "
            f"{atual} e ele exige {JAVA_MIN_MAJOR}+.")
        log(f"         Instale com: {JAVA_INSTALL_HINT}")
    else:
        log(f"OK  P2Rank {version} installed at {prank}")
    return True


def detect_xtb() -> Optional[str]:
    """Caminho do executável `xtb`, se instalado.

    Mesma ordem de busca de pipeline_voxelize._xtb_bin, para que o que o
    instalador mostra seja o que o pipeline vai encontrar.
    """
    env_home = os.environ.get("XTB_HOME")
    if env_home:
        cand = Path(env_home) / "bin" / "xtb"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which("xtb")
    if found:
        return found
    for base in (TOOLS_DIR, Path.home(), Path("/opt")):
        if not base.is_dir():
            continue
        try:
            for cand in sorted(base.glob("xtb*/bin/xtb")):
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)
        except OSError:
            continue
    return None


def install_xtb(version: str = XTB_VERSION, log=print) -> bool:
    """Baixa e extrai o xtb em <projeto>/tools/. Não requer root.

    O tarball oficial já vem com o binário estático em bin/xtb e os
    parâmetros GFN em share/xtb/ (lidos em tempo de execução, achados por
    caminho relativo ao binário — extrair preserva essa estrutura).
    """
    import tarfile
    import urllib.request

    url = XTB_URL_TMPL.format(v=version)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    tarball = TOOLS_DIR / f"xtb-{version}-linux-x86_64.tar.xz"
    try:
        log(f"$ download {url}")
        urllib.request.urlretrieve(url, tarball)
        log(f"$ tar -xJf {tarball.name} -C {TOOLS_DIR}")
        with tarfile.open(tarball, "r:xz") as tf:
            tf.extractall(TOOLS_DIR)
        tarball.unlink(missing_ok=True)
    except Exception as exc:
        log(f"FAILED  xtb download/extract: {exc}")
        return False

    xtb_bin = detect_xtb()
    if not xtb_bin:
        log("FAILED  xtb extracted but 'bin/xtb' not found in tools/")
        return False
    try:
        os.chmod(xtb_bin, 0o755)
    except OSError:
        pass
    log(f"OK  xtb {version} installed at {xtb_bin}")
    return True


# (display, key, default_spec, description, auto_installable)
EXTERNAL_TOOLS: list[tuple] = [
    ("P2Rank", "__p2rank__", P2RANK_VERSION,
     "Heuristic binding-site search (Step 9). Java program, installed into "
     "tools/. Requires Java 17+.", True),
    ("Java (JRE)", "__java__", JAVA_INSTALL_HINT,
     f"Required by P2Rank (version {JAVA_MIN_MAJOR} or newer). Needs root: "
     "run the command shown in a terminal.", False),
    ("xtb", "__xtb__", XTB_VERSION,
     "Semi-empirical (GFN-xTB) electronic descriptors for the site (Step 5). "
     "Static binary, no other dependency, installed into tools/.", True),
]


class _InstallWorker(QThread):
    """Executa a instalação em thread separada, emitindo sinais de log/progresso."""
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    done_signal     = pyqtSignal(bool)   # True = sucesso total

    def __init__(self, python_bin: str, tasks: list[tuple], parent=None):
        """
        tasks: lista de (display_name, spec, kind), com kind em
        {"pip", "p2rank", "xtb"}. Tuplas de 2 elementos são aceitas e
        tratadas como "pip", para não quebrar chamadas anteriores a esta
        classe.
        """
        super().__init__(parent)
        self.python_bin = python_bin
        self.tasks      = [t if len(t) == 3 else (t[0], t[1], "pip") for t in tasks]

    def run(self):
        total   = len(self.tasks)
        success = True
        for i, (name, spec, kind) in enumerate(self.tasks):
            self.log_signal.emit(f"\n--- Installing {name} ---")

            if kind == "p2rank":
                ok = install_p2rank(spec or P2RANK_VERSION,
                                    log=self.log_signal.emit)
                success = success and ok
            elif kind == "xtb":
                ok = install_xtb(spec or XTB_VERSION, log=self.log_signal.emit)
                success = success and ok
            else:
                self.log_signal.emit(f"$ pip install {spec}")
                res = _run([self.python_bin, "-m", "pip", "install"] + spec.split())
                if res.returncode == 0:
                    self.log_signal.emit(f"OK  {name} installed.")
                else:
                    lines = [l for l in res.stdout.splitlines()
                             if any(kw in l for kw in ("ERROR", "error", "Failed", "Could not"))]
                    self.log_signal.emit(f"FAILED  {name} — installation error:")
                    for l in lines[:6]:
                        self.log_signal.emit(f"   {l}")
                    success = False

            pct = int((i + 1) / total * 100)
            self.progress_signal.emit(pct)
        self.done_signal.emit(success)


class RequirementsInstaller(QWidget):
    """
    Janela de instalação de dependências do CODEEP.
    Verifica quais pacotes já estão instalados e permite instalar os ausentes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CODEEP — Install Requirements")
        self.setStyleSheet(_SS_WIN)
        self.setMinimumSize(820, 680)
        self.resize(900, 740)

        self._hw: Optional[dict]       = None
        self._worker: Optional[_InstallWorker] = None

        # Mapeamentos internos
        self._checks: dict[str, QCheckBox]  = {}  # import_name → checkbox
        self._edits:  dict[str, QLineEdit]  = {}  # import_name → line edit
        self._status: dict[str, QLabel]     = {}  # import_name → label de status

        self._build_ui()
        self._detect_and_refresh()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ── Cabeçalho ──────────────────────────────────────────────────────
        hdr = QLabel("CODEEP — Install Requirements")
        hdr.setStyleSheet("font-size: 13pt; font-weight: bold; color: #F57C6F;")
        root.addWidget(hdr)

        # Detecta se já estamos no venv correto
        _venv_py  = str(Path.home() / ".venv" / "CODEEP" / "bin" / "python")
        _in_venv  = (
            Path(sys.executable).resolve() == Path(_venv_py).resolve()
        ) if Path(_venv_py).is_file() else False
        _venv_dir = str(Path.home() / ".venv" / "CODEEP")
        _py_info  = f"{sys.executable}  ({sys.version.split()[0]})"

        if _in_venv:
            _env_color = "#27AE60"
            _env_note  = "running inside the CODEEP venv — packages will be installed here"
        else:
            _env_color = "#E67E22"
            _env_note  = "not running inside the venv — restart via:  python3 CODEEP.py"

        desc = QLabel(
            f"Virtual environment:  <b>{_venv_dir}</b><br>"
            f"Active Python:  <span style='color:#5A6A7E;'>{_py_info}</span><br>"
            f"<span style='color:{_env_color};'>{_env_note}</span>"
        )
        desc.setTextFormat(Qt.RichText)
        desc.setStyleSheet("font-size: 9pt; color: #6E8CA8;")
        root.addWidget(desc)

        # ── Tabela de pacotes ──────────────────────────────────────────────
        pkg_grp = QGroupBox("Packages")
        pkg_lay = QGridLayout(pkg_grp)
        pkg_lay.setHorizontalSpacing(12)
        pkg_lay.setVerticalSpacing(6)

        # Cabeçalho da tabela
        for col, (text, width) in enumerate([
            ("Install", 60), ("Status", 28), ("Package", 150),
            ("Version / Spec (editable)", 230), ("Description", 280),
        ]):
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setFixedWidth(width)
            lbl.setStyleSheet("color: #F57C6F;")
            pkg_lay.addWidget(lbl, 0, col)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #D0D7E3;")
        pkg_lay.addWidget(sep, 1, 0, 1, 5)

        for row_i, (display, import_name, default_spec, description, critical) in \
                enumerate(PACKAGES_CATALOG, start=2):

            # Checkbox de seleção
            chk = QCheckBox()
            chk.setChecked(True)
            self._checks[import_name] = chk
            pkg_lay.addWidget(chk, row_i, 0, Qt.AlignCenter)

            # Ícone de status (atualizado após detecção)
            lbl_status = QLabel("…")
            lbl_status.setFixedWidth(26)
            lbl_status.setAlignment(Qt.AlignCenter)
            lbl_status.setStyleSheet("font-size: 12pt;")
            self._status[import_name] = lbl_status
            pkg_lay.addWidget(lbl_status, row_i, 1)

            # Nome do pacote
            lbl_name = QLabel(display)
            lbl_name.setFixedWidth(148)
            lbl_name.setStyleSheet("font-weight: bold; color: #C9D1D9;")
            pkg_lay.addWidget(lbl_name, row_i, 2)

            # Campo de versão editável
            ed = QLineEdit(default_spec)
            ed.setFixedWidth(228)
            ed.setToolTip(f"pip spec for {display}. Edit the version if needed.")
            self._edits[import_name] = ed
            pkg_lay.addWidget(ed, row_i, 3)

            # Descrição do pacote
            lbl_desc = QLabel(description)
            lbl_desc.setStyleSheet("font-size: 9pt; color: #6E8CA8;")
            lbl_desc.setWordWrap(True)
            pkg_lay.addWidget(lbl_desc, row_i, 4)

        # ── Ferramentas externas (não-pip) ──────────────────────────────────
        base_row = len(PACKAGES_CATALOG) + 2
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #D0D7E3;")
        pkg_lay.addWidget(sep2, base_row, 0, 1, 5)

        for off, (display, key, default_spec, description, auto) in \
                enumerate(EXTERNAL_TOOLS, start=base_row + 1):
            chk = QCheckBox()
            chk.setChecked(auto)
            # Java exige root: o instalador não o executa, só reporta.
            chk.setEnabled(auto)
            if not auto:
                chk.setToolTip("Requires root — run the command in a terminal.")
            self._checks[key] = chk
            pkg_lay.addWidget(chk, off, 0, Qt.AlignCenter)

            lbl_status = QLabel("…")
            lbl_status.setFixedWidth(26)
            lbl_status.setAlignment(Qt.AlignCenter)
            lbl_status.setStyleSheet("font-size: 12pt;")
            self._status[key] = lbl_status
            pkg_lay.addWidget(lbl_status, off, 1)

            lbl_name = QLabel(display)
            lbl_name.setFixedWidth(148)
            lbl_name.setStyleSheet("font-weight: bold; color: #C9D1D9;")
            pkg_lay.addWidget(lbl_name, off, 2)

            ed = QLineEdit(default_spec)
            ed.setFixedWidth(228)
            ed.setReadOnly(not auto)
            ed.setToolTip(f"Version of {display}." if auto
                          else "Command to run in a terminal.")
            self._edits[key] = ed
            pkg_lay.addWidget(ed, off, 3)

            lbl_desc = QLabel(description)
            lbl_desc.setStyleSheet("font-size: 9pt; color: #6E8CA8;")
            lbl_desc.setWordWrap(True)
            pkg_lay.addWidget(lbl_desc, off, 4)

        # Scroll area — encapsula o grid de pacotes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        pkg_grp_wrap = QWidget()
        wrap_lay = QVBoxLayout(pkg_grp_wrap)
        wrap_lay.setContentsMargins(0, 0, 0, 0)
        wrap_lay.addWidget(pkg_grp)
        scroll.setWidget(pkg_grp_wrap)
        scroll.setMinimumHeight(320)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(scroll, 1)

        # ── PyTorch CUDA Options ───────────────────────────────────────────
        torch_grp = QGroupBox("PyTorch CUDA Options")
        torch_lay = QHBoxLayout(torch_grp)
        torch_lay.addWidget(QLabel("CUDA variant:"))
        self.ed_torch_variant = QLineEdit("cu128")
        self.ed_torch_variant.setFixedWidth(80)
        self.ed_torch_variant.setToolTip("cu128, cu121, cu118 or cpu")
        torch_lay.addWidget(self.ed_torch_variant)
        torch_lay.addSpacing(20)
        torch_lay.addWidget(QLabel("Index URL:"))
        self.ed_torch_index = QLineEdit("https://download.pytorch.org/whl/cu128")
        self.ed_torch_index.setMinimumWidth(320)
        torch_lay.addWidget(self.ed_torch_index)
        torch_lay.addStretch()
        torch_grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(torch_grp)

        # ── Progress bar e log ─────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setMinimum(0); self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setFormat("Progress: %p%")
        root.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(_SS_LOG)
        self.log.setFixedHeight(160)
        root.addWidget(self.log)

        # ── Botões — sem ícones ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_check = QPushButton("Check again")
        self.btn_check.setStyleSheet(_SS_BTN_SECONDARY)
        self.btn_check.clicked.connect(self._detect_and_refresh)

        self.btn_select_all = QPushButton("Select all")
        self.btn_select_all.setStyleSheet(_SS_BTN_SECONDARY)
        self.btn_select_all.clicked.connect(lambda: self._set_all_checks(True))

        self.btn_select_missing = QPushButton("Missing only")
        self.btn_select_missing.setStyleSheet(_SS_BTN_SECONDARY)
        self.btn_select_missing.clicked.connect(self._select_only_missing)

        self.btn_install = QPushButton("Install selected")
        self.btn_install.setStyleSheet(_SS_BTN_INSTALL)
        self.btn_install.setFixedHeight(36)
        self.btn_install.clicked.connect(self._run_install)

        self.btn_close = QPushButton("Close")
        self.btn_close.setStyleSheet(_SS_BTN_CLOSE)
        self.btn_close.clicked.connect(self.close)

        for b in [self.btn_check, self.btn_select_all,
                  self.btn_select_missing, self.btn_install, self.btn_close]:
            btn_row.addWidget(b)

        root.addLayout(btn_row)

    # ── Detecção / atualização dos status ──────────────────────────────────
    def _detect_and_refresh(self):
        QApplication.processEvents()

        # Detecta hardware apenas para ajustar variante do PyTorch
        self._hw = detect_hardware()
        gpu = self._hw["gpu"]

        # Ajusta spec do PyTorch com base no hardware detectado
        if gpu["available"]:
            torch_info = select_torch_variant(gpu)
            self._edits["torch"].setText(torch_info["version"])
            self.ed_torch_variant.setText(torch_info["tag"])
            self.ed_torch_index.setText(torch_info["index_url"])

        # Verifica cada pacote e atualiza ícone e checkbox
        for display, import_name, _, _, _ in PACKAGES_CATALOG:
            installed = _pkg_installed(import_name)
            ver       = _pkg_version(import_name)
            lbl       = self._status[import_name]
            chk       = self._checks[import_name]
            if installed:
                lbl.setText("✅")
                lbl.setToolTip(f"Installed (version: {ver or '?'})")
                chk.setChecked(False)   # já instalado → desmarca por padrão
            else:
                lbl.setText("⚠️")
                lbl.setToolTip("Not found in current Python environment")
                chk.setChecked(True)

        # Ferramentas externas
        prank = detect_p2rank()
        java  = detect_java()

        lbl, chk = self._status["__p2rank__"], self._checks["__p2rank__"]
        if prank:
            lbl.setText("✅"); lbl.setToolTip(f"Installed at {prank}")
            chk.setChecked(False)
        else:
            lbl.setText("⚠️"); lbl.setToolTip("Not found — Step 9 heuristic search unavailable")
            chk.setChecked(True)

        lbl = self._status["__java__"]
        if java["ok"]:
            lbl.setText("✅"); lbl.setToolTip(f"Java {java['version']}")
        elif java["found"]:
            lbl.setText("⚠️")
            lbl.setToolTip(f"Java {java['version']} is older than "
                           f"{JAVA_MIN_MAJOR} — P2Rank will not run")
        else:
            lbl.setText("⚠️"); lbl.setToolTip("Not installed — required by P2Rank")

        if prank and not java["ok"]:
            self._log(f"WARNING  P2Rank is installed but Java "
                      f"{java['version'] or 'is missing'}; it requires "
                      f"{JAVA_MIN_MAJOR}+. Run: {JAVA_INSTALL_HINT}")

        xtb_bin = detect_xtb()
        lbl, chk = self._status["__xtb__"], self._checks["__xtb__"]
        if xtb_bin:
            lbl.setText("✅"); lbl.setToolTip(f"Installed at {xtb_bin}")
            chk.setChecked(False)
        else:
            lbl.setText("⚠️")
            lbl.setToolTip("Not found — Step 5 falls back to the heuristic "
                           "approximation (no QM electronic descriptors)")
            chk.setChecked(True)

        self._log(f"Check complete. Python: {sys.executable}")

    # ── Helpers de seleção ─────────────────────────────────────────────────
    def _set_all_checks(self, state: bool):
        for chk in self._checks.values():
            chk.setChecked(state)

    def _select_only_missing(self):
        for import_name, chk in self._checks.items():
            if import_name == "__p2rank__":
                chk.setChecked(detect_p2rank() is None)
                continue
            if import_name == "__xtb__":
                chk.setChecked(detect_xtb() is None)
                continue
            if import_name == "__java__":
                continue          # não instalável por aqui (exige root)
            installed = _pkg_installed(import_name)
            chk.setChecked(not installed)

    # ── Instalação ─────────────────────────────────────────────────────────
    def _run_install(self):
        # Monta lista de tarefas
        tasks: list[tuple[str, str]] = []
        for display, import_name, _, _, _ in PACKAGES_CATALOG:
            chk = self._checks[import_name]
            if not chk.isChecked():
                continue
            spec = self._edits[import_name].text().strip()
            if not spec:
                continue
            # PyTorch recebe tratamento especial (--index-url)
            if import_name == "torch":
                index = self.ed_torch_index.text().strip()
                spec_full = (
                    f"{spec} torchvision torchaudio "
                    f"--index-url {index} --no-cache-dir"
                )
                tasks.append((display, spec_full, "pip"))
            else:
                tasks.append((display, spec, "pip"))

        # Ferramentas externas selecionadas
        if self._checks["__p2rank__"].isChecked():
            tasks.append(("P2Rank",
                          self._edits["__p2rank__"].text().strip() or P2RANK_VERSION,
                          "p2rank"))
        if self._checks["__xtb__"].isChecked():
            tasks.append(("xtb",
                          self._edits["__xtb__"].text().strip() or XTB_VERSION,
                          "xtb"))

        if not tasks:
            QMessageBox.information(self, "No packages selected",
                                    "No packages selected for installation.")
            return

        pybin    = sys.executable
        venv_dir = str(Path.home() / ".venv" / "CODEEP")
        _venv_py = str(Path.home() / ".venv" / "CODEEP" / "bin" / "python")
        _in_venv = False
        try:
            _in_venv = Path(pybin).resolve() == Path(_venv_py).resolve()
        except Exception:
            pass

        if not _in_venv:
            resp = QMessageBox.warning(
                self, "Warning — Environment",
                f"CODEEP is not running inside the CODEEP venv.\n\n"
                f"Active Python: {pybin}\n"
                f"Expected venv: {venv_dir}\n\n"
                "Restart CODEEP with:\n"
                "  python3 CODEEP.py\n\n"
                "Install into the current Python anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return

        self._log(f"\nInstallation Python : {pybin}")
        self._log(f"Virtual environment : {venv_dir}")
        self._log(f"Inside venv         : {'yes' if _in_venv else 'no'}")
        self._log(f"Packages to install : {len(tasks)}\n")

        self.btn_install.setEnabled(False)
        self.progress.setValue(0)

        self._worker = _InstallWorker(pybin, tasks, parent=self)
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self.progress.setValue)
        self._worker.done_signal.connect(self._install_done)
        self._worker.start()

    def _install_done(self, success: bool):
        self.btn_install.setEnabled(True)
        self.progress.setValue(100)
        self._detect_and_refresh()
        if success:
            self._log("\nAll packages installed successfully!")
            QMessageBox.information(self, "Done",
                                    "Installation complete!\n\n"
                                    "Restart CODEEP to load the newly installed packages.")
        else:
            self._log("\nSome packages could not be installed. Check the log above.")
            QMessageBox.warning(self, "Warning",
                                "Some packages were not installed.\n"
                                "Check the log for details.")

    # ── Log ────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


# ==========================================================================
# ─── TESTE DIRETO ─────────────────────────────────────────────────────────
# ==========================================================================

if __name__ == "__main__":
    # Bootstrap antes de qualquer Qt
    if not bootstrap_pyqt5(interactive=True):
        sys.exit(1)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = RequirementsInstaller()
    w.show()
    sys.exit(app.exec_())
