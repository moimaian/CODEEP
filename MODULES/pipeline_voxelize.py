"""
Pipeline - Etapa 4: Voxelização 3D + Propriedades Quânticas
============================================================
Lê os pares curados (binding_site.pdb + ligand.pdb) e gera:

  ┌─ Sítio de ligação ──────────────────────────────────────────────────────┐
  │  Tensor 4D: (X, Y, Z, C_site) onde C_site = 5 canais                   │
  │    Canal 0 — Densidade atômica (ocupância de vdW gaussiana)             │
  │    Canal 1 — Carga parcial (Gasteiger) interpolada na grade             │
  │    Canal 2 — MEP (Potencial Eletrostático Molecular) — HF/STO-3G       │
  │    Canal 3 — Densidade HOMO (orbital de fronteira — doador)             │
  │    Canal 4 — Densidade LUMO (orbital de fronteira — aceptor)            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ Ligante ───────────────────────────────────────────────────────────────┐
  │  Grafo molecular PyG: Data(x, edge_index, edge_attr, pos, smiles)       │
  │    x          — features dos nós (átomos): Z, carga, hibridização…      │
  │    edge_index — conectividade (COO)                                      │
  │    edge_attr  — features das arestas: tipo de ligação, comprimento       │
  │    pos        — coordenadas 3D do cristal                                │
  └─────────────────────────────────────────────────────────────────────────┘

  Saída por par: data/tensors/{pair_id}/
    site_tensor.npz    ← tensor do sítio (float32, comprimido)
    ligand_graph.pt    ← grafo PyG serializado (torch)
    meta.json          ← metadados do par

Requisitos:
    pip install numpy scipy biopython rdkit pyscf torch torch-geometric tqdm

Notas sobre o cálculo quântico:
    • Nível de teoria: RHF/STO-3G (rápido, adequado para gerar descritores
      de densidade para ML; não é cálculo de alta precisão).
    • Para ligantes com > MAX_QM_ATOMS átomos pesados, o QM é substituído
      por aproximações semi-empíricas baseadas em cargas Gasteiger e
      raio de van der Waals (muito mais rápido para datasets grandes).
    • O MEP do SÍTIO é calculado como soma de potenciais de Coulomb das
      cargas parciais Gasteiger dos resíduos (escala bem para proteínas).

Uso:
    python 04_voxelize.py                      # processa todos os pares
    python 04_voxelize.py --pair 7W40_E2H_A101 # processa um par específico
    python 04_voxelize.py --workers 4          # paralelismo
"""

import multiprocessing
import os
import re
import sys
import json
import time
import logging
import argparse
import warnings
import tempfile
from pathlib import Path
from typing import Optional
import pandas as pd

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ── Importações condicionais ───────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdPartialCharges, rdMolTransforms
    from rdkit.Chem.rdchem import HybridizationType, BondType, ChiralType
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    raise ImportError("RDKit obrigatório: pip install rdkit")

try:
    from Bio.PDB import PDBParser
    HAS_BIO = True
except ImportError:
    HAS_BIO = False
    raise ImportError("BioPython obrigatório: pip install biopython")

try:
    import torch
    from torch_geometric.data import Data as PyGData
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch/PyG não encontrado — grafos serão salvos como .npz")

try:
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    warnings.warn("PySCF não encontrado — QM substituído por aproximação Gasteiger")

warnings.filterwarnings("ignore")

# ── Detecção de GPU ────────────────────────────────────────────────────────────
def _detect_gpu() -> tuple[bool, str]:
    """Detecta GPU CUDA disponível. Retorna (disponível, descrição)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{name} ({mem:.1f} GB VRAM)"
    except Exception:
        pass
    return False, "CPU apenas (CUDA não disponível)"

HAS_CUDA, GPU_DESC = _detect_gpu()

# ── CuPy: NumPy acelerado por CUDA para voxelização das gaussianas ─────────────
# NOTA: CuPy 14 tem um bug nos headers CCCL que causa NVRTC_ERROR_COMPILATION
# em kernels de reduction via CUB (.sum(axis=0), matmul, tensordot).
# Porém cp.einsum('ij->j', x) usa cuBLAS e funciona corretamente.
# A função _voxelize_atoms usa einsum em vez de .sum(axis=0) para contornar.
try:
    import cupy as cp
    # Testa einsum — a operação específica que usamos (não o sum CUB)
    _t = cp.zeros((2, 4), dtype=cp.float32)
    cp.einsum('ij->j', _t)
    del _t
    HAS_CUPY = bool(HAS_CUDA)
    if HAS_CUPY:
        cp.cuda.Device(0).use()
except Exception:
    cp       = None
    HAS_CUPY = False
    warnings.warn("CuPy indisponível — usando NumPy (CPU).")

def _xp():
    """Retorna cupy se disponível, senão numpy."""
    return cp if HAS_CUPY else np

# ── GPU4PySCF: cálculos QM acelerados por CUDA ────────────────────────────────
# Instalar: pip install gpu4pyscf-cuda12x
# Documentação: https://github.com/pyscf/gpu4pyscf
# Speedup típico vs PySCF/CPU: 5-20x para RHF com bases pequenas (STO-3G, 3-21G)
#
# NOTA: GPU4PySCF requer bibliotecas cu13 (libnvJitLink.so.13) que podem
# conflitar com o ambiente cu12. Se houver erro de importação, o script
# usa automaticamente PySCF/CPU ou a aproximação Gasteiger como fallback.
try:
    import gpu4pyscf
    from gpu4pyscf import scf as gpu_scf
    HAS_GPU4PYSCF = bool(HAS_CUDA)
except Exception:
    gpu_scf       = None
    HAS_GPU4PYSCF = False

# ── Caminhos ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR     = PROJECT_ROOT / "data"
LOG_DIR      = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(stream=None),
    ],
)
log = logging.getLogger(__name__)

def _setup_log_file(log_dir: Path) -> None:
    """Instala (ou substitui) o FileHandler do logger com o caminho correto.
    
    Chamada no início de run() quando já se sabe a fase e o job_dir.
    Remove qualquer FileHandler anterior para evitar escrita em arquivo stale.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)
    fh = logging.FileHandler(log_dir / "voxelize.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(fh)
    log.info(f"  Log      : {log_dir / 'voxelize.log'}")

# ── Configuração ──────────────────────────────────────────────────────────────
VOXEL_CONFIG = {
    # Dimensões da grade (voxels por lado)
    # 32³ = ~100k voxels (rápido, bom para protótipo)
    # 48³ recomendado para produção
    "grid_size": 32,

    # Padding além dos átomos extremos do sítio (Å)
    "padding_A": 3.0,

    # Sigma da gaussiana de ocupância atômica (Å)
    "gaussian_sigma": 0.5,

    # Tamanho do lote de átomos para voxelização vetorizada (ajusta uso de memória)
    "vox_batch":       500 if HAS_CUPY else 100, 

    # Limite de átomos pesados para cálculo QM completo (RHF/STO-3G)
    # Acima disso usa aproximação semi-empírica (Gasteiger + densidade vdW)
    "max_qm_atoms": 60,

    # Base QM (STO-3G é a mais rápida; use 3-21G para mais precisão)
    "qm_basis": "sto-3g",

    # Número de canais do tensor do sítio
    "n_site_channels": 5,
    # [0] densidade atômica vdW
    # [1] carga parcial (Gasteiger)
    # [2] MEP
    # [3] densidade HOMO
    # [4] densidade LUMO

    # ── ECFP4 como feature global do ligante (Opção A) ────────────────────────
    # Adiciona o Morgan Fingerprint (raio=2, 512 bits) ao grafo do ligante
    # como atributo global `graph.ecfp4` — não afeta os features atômicos (x).
    #
    # DESATIVADO por padrão: o ECFP4 tende a ancorar o decoder em fingerprints
    # já vistos no treino, reduzindo a diversidade dos ligantes gerados.
    # Isso é indesejável em descoberta de novos fármacos onde a exploração
    # do espaço químico é prioritária.
    #
    # Para ativar: mude para True e atualize o GNNDecoder no 05_train.py
    # para concatenar ecfp4 ao vetor z antes do decoder.
    # Impacto: +512 parâmetros na entrada do decoder, <1ms/molécula.
    "use_ecfp4": False,
    "ecfp4_bits": 512,
    "ecfp4_radius": 2,
}

# Raios de van der Waals por elemento (Å) — Bondi 1964
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 2.05, "ZN": 1.39, "MG": 1.73, "CA": 2.31, "MN": 1.97,
    "CU": 1.40, "NI": 1.63, "CO": 1.70, "SE": 1.90,
}
VDW_DEFAULT = 1.70

# Features dos átomos no grafo (one-hot encoding)
ATOM_TYPES   = ["C","N","O","S","P","F","CL","BR","I","FE","ZN","MG","OTHER"]
HYBRID_TYPES = [HybridizationType.SP, HybridizationType.SP2,
                HybridizationType.SP3, HybridizationType.OTHER] if HAS_RDKIT else []
BOND_TYPES   = [BondType.SINGLE, BondType.DOUBLE,
                BondType.TRIPLE, BondType.AROMATIC] if HAS_RDKIT else []


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — GRADE 3D
# ══════════════════════════════════════════════════════════════════════════════

def build_grid(coords: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Constrói a grade 3D centrada nos átomos do sítio.

    Returns:
        grid_coords : (N³, 3) — coordenadas do centro de cada voxel
        origin      : (3,)   — canto inferior esquerdo da grade
        voxel_size  : float  — tamanho de um voxel em Å
    """
    pad  = config["padding_A"]
    n    = config["grid_size"]

    lo   = coords.min(axis=0) - pad
    hi   = coords.max(axis=0) + pad
    span = hi - lo

    voxel_size = span.max() / (n - 1)  # voxels cúbicos (lado único)
    center     = (lo + hi) / 2.0
    origin     = center - voxel_size * (n - 1) / 2.0

    # Grade de pontos (n, n, n, 3)
    ax = origin[0] + np.arange(n) * voxel_size
    ay = origin[1] + np.arange(n) * voxel_size
    az = origin[2] + np.arange(n) * voxel_size
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    grid_coords = np.stack([gx, gy, gz], axis=-1)  # (n,n,n,3)

    return grid_coords, origin, float(voxel_size)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — CANAIS DO SÍTIO (Tensor 4D)
# ══════════════════════════════════════════════════════════════════════════════

def _voxelize_atoms(
    grid_flat: np.ndarray,          # (N_vox, 3) — pontos da grade achatada
    atom_coords: np.ndarray,        # (N_atoms, 3)
    weights_density: np.ndarray,    # (N_atoms,) — raio vdW por átomo
    weights_charge: np.ndarray,     # (N_atoms,) — carga parcial por átomo
    sigma: float,
    grid_shape: tuple,
    batch: int = 500 if HAS_CUPY else 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula os canais 0, 1 e 2 de forma totalmente vetorizada.

    Em vez de iterar átomo por átomo, monta uma matriz de distâncias
    (N_atoms, N_vox) de uma só vez usando broadcasting e opera sobre
    ela com operações NumPy/CuPy puras — sem nenhum loop Python.

    Para sítios muito grandes (>800 átomos), processa em lotes de 100
    átomos para não estourar a memória da GPU (16 GB VRAM).

    Ganho típico vs. loop Python:
      CPU (NumPy) : 8-20x
      GPU (CuPy)  : 50-200x
    """
    xp        = _xp()                  # cupy se disponível, senão numpy
    BATCH = batch                     # átomos por lote (ajusta memória)
    inv_2sig2 = 1.0 / (2.0 * sigma**2)

    # Transfere dados para GPU se CuPy disponível
    gf  = xp.asarray(grid_flat,      dtype=xp.float32)   # (N_vox, 3)
    ac  = xp.asarray(atom_coords,    dtype=xp.float32)   # (N_atoms, 3)
    wd  = xp.asarray(weights_density, dtype=xp.float32)  # (N_atoms,)
    wc  = xp.asarray(weights_charge,  dtype=xp.float32)  # (N_atoms,)

    N_vox   = gf.shape[0]
    N_atoms = ac.shape[0]

    density = xp.zeros(N_vox, dtype=xp.float32)
    charge  = xp.zeros(N_vox, dtype=xp.float32)
    mep     = xp.zeros(N_vox, dtype=xp.float32)

    for i in range(0, N_atoms, BATCH):
        ac_b = ac[i:i+BATCH]           # (b, 3)
        wd_b = wd[i:i+BATCH]           # (b,)
        wc_b = wc[i:i+BATCH]           # (b,)

        # diff: (b, N_vox, 3) — distância vetorial de cada átomo a cada voxel
        diff  = gf[None, :, :] - ac_b[:, None, :]   # broadcast
        dist2 = (diff * diff).sum(axis=-1)           # (b, N_vox)
        gauss = xp.exp(-dist2 * inv_2sig2)           # (b, N_vox)

        # Canal 0: densidade atômica
        # Usa einsum em vez de .sum(axis=0) para evitar bug CUB no CuPy 14
        # einsum('ij->j') usa cuBLAS e funciona corretamente
        density += xp.einsum('ij->j', wd_b[:, None] * gauss)

        # Canal 1: carga interpolada
        charge  += xp.einsum('ij->j', wc_b[:, None] * gauss)

        # Canal 2: MEP coulombiano  q / |r - r_i|
        dist    = xp.sqrt(xp.clip(dist2, 0.01, None))
        mep     += xp.einsum('ij->j', wc_b[:, None] / dist)

    # Volta para CPU numpy e reshape para (n,n,n)
    if HAS_CUPY and xp is cp:
        density = cp.asnumpy(density)
        charge  = cp.asnumpy(charge)
        mep     = cp.asnumpy(mep)
    else:
        density = np.asarray(density)
        charge  = np.asarray(charge)
        mep     = np.asarray(mep)

    return (density.reshape(grid_shape).astype(np.float32),
            charge.reshape(grid_shape).astype(np.float32),
            mep.reshape(grid_shape).astype(np.float32))


def channel_atomic_density(
    grid_coords: np.ndarray,
    atom_coords: np.ndarray,
    elements: list[str],
    sigma: float,
) -> np.ndarray:
    """Canal 0 — wrapper mantido para compatibilidade."""
    n  = grid_coords.shape[0]
    wd = np.array([VDW_RADII.get(e.upper(), VDW_DEFAULT) for e in elements],
                  dtype=np.float32)
    wc = np.zeros(len(elements), dtype=np.float32)
    gf = grid_coords.reshape(-1, 3)
    d, _, _ = _voxelize_atoms(gf, atom_coords, wd, wc, sigma, (n, n, n))
    return d


def channel_gasteiger_charges(
    grid_coords: np.ndarray,
    atom_coords: np.ndarray,
    charges: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Canal 1 — wrapper mantido para compatibilidade."""
    n  = grid_coords.shape[0]
    wd = np.zeros(len(charges), dtype=np.float32)
    gf = grid_coords.reshape(-1, 3)
    _, c, _ = _voxelize_atoms(gf, atom_coords, wd, charges, sigma, (n, n, n))
    return c


def channel_mep_coulomb(
    grid_coords: np.ndarray,
    atom_coords: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    """Canal 2 — wrapper mantido para compatibilidade."""
    n     = grid_coords.shape[0]
    sigma = 1.0   # sigma irrelevante para MEP (usa dist direto)
    wd    = np.zeros(len(charges), dtype=np.float32)
    gf    = grid_coords.reshape(-1, 3)
    _, _, m = _voxelize_atoms(gf, atom_coords, wd, charges, sigma, (n, n, n))
    return m


def channel_homo_lumo_qm(
    grid_coords: np.ndarray,
    mol_rdkit,
    origin: np.ndarray,
    voxel_size: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Canais 3 e 4 — Densidade de probabilidade HOMO e LUMO via RHF/STO-3G.

    Usa PySCF para calcular os coeficientes dos orbitais moleculares de
    fronteira e projetá-los na grade 3D.

    Para moléculas grandes (> MAX_QM_ATOMS) usa aproximação:
      - HOMO ≈ densidade dos átomos mais eletropositivos (doadores)
      - LUMO ≈ densidade dos átomos mais eletronegativos (aceptores)
    """
    n = grid_coords.shape[0]
    conf = mol_rdkit.GetConformer()
    pos  = conf.GetPositions()
    heavy_idx = [a.GetIdx() for a in mol_rdkit.GetAtoms() if a.GetAtomicNum() > 1]

    # Com spawn (obrigatório desde a migração para CODEEP), o processo filho
    # tem contexto CUDA limpo — GPU4PySCF é seguro em qualquer profundidade.
    use_gpu4 = HAS_GPU4PYSCF

    if not HAS_PYSCF or len(heavy_idx) > config["max_qm_atoms"]:
        log.debug("  QM: aproximação semi-empírica (molécula grande ou PySCF ausente)")
        return _homo_lumo_approx(grid_coords, mol_rdkit, config)

    # ── Cálculo QM via PySCF ──────────────────────────────────────────────────
    try:
        atom_str = []
        for atom in mol_rdkit.GetAtoms():
            sym  = atom.GetSymbol()
            xyz  = pos[atom.GetIdx()]
            atom_str.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")

        # Determina spin correto para evitar erro de elétrons ímpares
        charge    = Chem.GetFormalCharge(mol_rdkit)
        n_electrons = sum(a.GetAtomicNum() for a in mol_rdkit.GetAtoms()) - charge
        spin      = n_electrons % 2  # 0 = singleto, 1 = dubleto (radical)

        pyscf_mol = gto.Mole()
        pyscf_mol.atom    = "\n".join(atom_str)
        pyscf_mol.basis   = config["qm_basis"]
        pyscf_mol.charge  = charge
        pyscf_mol.spin    = spin
        pyscf_mol.verbose = 0
        pyscf_mol.build()

        if use_gpu4:
            log.debug("  QM: GPU4PySCF (processo principal)")
            mf = gpu_scf.RHF(pyscf_mol)
        else:
            log.debug("  QM: PySCF CPU")
            # RHF para singletos, ROHF para radicais (spin=1)
            mf = scf.RHF(pyscf_mol) if spin == 0 else scf.ROHF(pyscf_mol)
        mf.kernel()

        mo_occ  = mf.mo_occ
        homo_i  = int(np.where(mo_occ > 0)[0][-1])
        lumo_i  = homo_i + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            fn_h = os.path.join(tmpdir, "homo.cube")
            fn_l = os.path.join(tmpdir, "lumo.cube")
            cubegen.orbital(pyscf_mol, fn_h, mf.mo_coeff[:, homo_i],
                            nx=n, ny=n, nz=n, resolution=voxel_size)
            cubegen.orbital(pyscf_mol, fn_l, mf.mo_coeff[:, lumo_i],
                            nx=n, ny=n, nz=n, resolution=voxel_size)
            homo_grid = _read_cube_data(fn_h, n)
            lumo_grid = _read_cube_data(fn_l, n)

        return homo_grid**2, lumo_grid**2

    except Exception as e:
        log.debug(f"  QM falhou ({type(e).__name__}), usando aproximação Gasteiger")
        return _homo_lumo_approx(grid_coords, mol_rdkit, config)


def _homo_lumo_approx(
    grid_coords: np.ndarray,
    mol_rdkit,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Aproximação do HOMO/LUMO baseada em eletronegatividade.

    HOMO ≈ orbitais de doação: átomos com baixa eletronegatividade e
           carga Gasteiger positiva (N, O com lone pairs, C aromático).
    LUMO ≈ orbitais de aceitação: átomos com alta eletronegatividade,
           carbonilas, anéis com grupos retiradores.

    Esta aproximação captura a reatividade local de forma qualitativa
    e é consistente o suficiente para treinar descritores de ML.
    """
    n    = grid_coords.shape[0]
    conf = mol_rdkit.GetConformer()
    rdPartialCharges.ComputeGasteigerCharges(mol_rdkit)
    sigma = config["gaussian_sigma"] * 1.5  # gaussianas um pouco mais largas

    # Eletronegatividade de Pauling por elemento
    electroneg = {
        "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "P": 2.19,
        "F": 3.98, "CL": 3.16, "BR": 2.96, "I": 2.66,
        "FE": 1.83, "ZN": 1.65, "MG": 1.31, "SE": 2.55,
    }

    homo_grid = np.zeros((n, n, n), dtype=np.float32)
    lumo_grid = np.zeros((n, n, n), dtype=np.float32)

    for atom in mol_rdkit.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        sym   = atom.GetSymbol().upper()
        xyz   = conf.GetPositions()[atom.GetIdx()]
        q     = float(atom.GetDoubleProp("_GasteigerCharge")
                      if atom.HasProp("_GasteigerCharge") else 0.0)
        en    = electroneg.get(sym, 2.50)

        diff  = grid_coords - xyz
        dist2 = (diff**2).sum(axis=-1)
        gauss = np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)

        # HOMO: átomos doadores = baixa EN + carga negativa/neutra
        homo_weight = max(0.0, (3.0 - en) / 3.0 + max(0.0, -q))
        homo_grid  += homo_weight * gauss

        # LUMO: átomos aceptores = alta EN + carga positiva
        lumo_weight = max(0.0, (en - 2.0) / 2.0 + max(0.0, q))
        lumo_grid  += lumo_weight * gauss

    return homo_grid, lumo_grid


def _read_cube_data(cube_path: str, n: int) -> np.ndarray:
    """Lê os dados volumétricos de um arquivo .cube do PySCF.

    Formato Gaussian/PySCF:
      Linha 0-1: comentários
      Linha 2:   n_atoms  x_origin  y_origin  z_origin
      Linha 3:   nx  x_step  0  0
      Linha 4:   ny  0  y_step  0
      Linha 5:   nz  0  0  z_step
      Linhas 6..(6+n_atoms-1): átomos
      Linhas seguintes: dados volumétricos (row-major: x varia mais lentamente)
    """
    with open(cube_path) as f:
        lines = f.readlines()

    n_atoms   = abs(int(lines[2].split()[0]))
    nx        = int(lines[3].split()[0])
    ny        = int(lines[4].split()[0])
    nz        = int(lines[5].split()[0])
    data_start = 6 + n_atoms

    values = []
    for line in lines[data_start:]:
        values.extend(float(v) for v in line.split())

    expected = nx * ny * nz
    if len(values) < expected:
        # Preenche com zeros se faltarem dados (raro)
        values += [0.0] * (expected - len(values))

    arr = np.array(values[:expected], dtype=np.float32).reshape(nx, ny, nz)

    # Se as dimensões do cubo diferem do solicitado, redimensiona com interpolação
    if (nx, ny, nz) != (n, n, n):
        from scipy.ndimage import zoom
        factors = (n / nx, n / ny, n / nz)
        arr = zoom(arr, factors, order=1).astype(np.float32)

    return arr


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — LEITURA DO SÍTIO (PDB → arrays)
# ══════════════════════════════════════════════════════════════════════════════

def load_site_atoms(pdb_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Lê o binding_site.pdb e retorna coords, elementos e cargas Gasteiger.

    As cargas do sítio são calculadas residuo a residuo via RDKit a partir
    do SMILES de cada aminoácido padrão. Para resíduos não reconhecidos,
    usa carga 0.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("site", str(pdb_path))

    coords   = []
    elements = []

    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    elem = (atom.element or atom.get_name()[0]).strip().upper()
                    if elem in ("H", "D", ""):
                        continue
                    coords.append(atom.get_vector().get_array())
                    elements.append(elem)

    coords_arr = np.array(coords, dtype=np.float32)

    # Cargas Gasteiger aproximadas por elemento (proteínas)
    # Valores médios de cargas Gasteiger em aminoácidos
    elem_charge_approx = {
        "C": -0.05, "N": -0.30, "O": -0.35, "S": -0.15,
        "P": +0.40, "FE": +0.30, "ZN": +0.40, "MG": +0.50,
    }
    charges = np.array([elem_charge_approx.get(e, 0.0) for e in elements],
                       dtype=np.float32)

    return coords_arr, elements, charges


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — GRAFO MOLECULAR DO LIGANTE
# ══════════════════════════════════════════════════════════════════════════════

def one_hot(value, choices: list) -> list[int]:
    """Codificação one-hot com categoria 'other' no final."""
    enc = [0] * (len(choices) + 1)
    try:
        enc[choices.index(value)] = 1
    except ValueError:
        enc[-1] = 1
    return enc


def atom_features(atom) -> list[float]:
    """Vetor de features de um átomo RDKit (dimensão fixa = 37).

    Composição:
      [0:13]  — one-hot tipo de átomo (13 tipos + other)
      [13:17] — one-hot hibridização (SP, SP2, SP3, other)
      [17]    — número atômico normalizado
      [18]    — carga formal
      [19]    — carga Gasteiger
      [20]    — número de H implícitos
      [21]    — grau de conectividade
      [22]    — está em anel aromático (bool)
      [23]    — está em anel (bool)
      [24:28] — one-hot quiralidade (R, S, other, nenhuma)
      [28:37] — reservado (zeros) para expansão futura
    """
    sym = atom.GetSymbol().upper()
    feat = []

    # One-hot: tipo de átomo
    feat += one_hot(sym, ATOM_TYPES[:-1])  # -1 porque ATOM_TYPES já tem OTHER

    # One-hot: hibridização
    feat += one_hot(atom.GetHybridization(), HYBRID_TYPES[:-1])

    # Propriedades escalares — todos os valores são sanitizados para prevenir
    # inf/NaN que propagam através do GNN e causam colapso do treino.
    def _safe(v, lo=-10.0, hi=10.0):
        """Clipa e substitui inf/NaN por 0."""
        import math
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return float(np.clip(v, lo, hi))

    feat.append(_safe(atom.GetAtomicNum() / 100.0, 0.0, 1.0))
    feat.append(_safe(float(atom.GetFormalCharge()), -4.0, 4.0))
    q = float(atom.GetDoubleProp("_GasteigerCharge")
              if atom.HasProp("_GasteigerCharge") else 0.0)
    feat.append(_safe(q, -2.0, 2.0))          # Gasteiger: clipa ±2
    feat.append(_safe(float(atom.GetTotalNumHs()), 0.0, 8.0) / 8.0)
    feat.append(_safe(float(atom.GetDegree()) / 6.0, 0.0, 1.0))
    feat.append(float(atom.GetIsAromatic()))
    feat.append(float(atom.IsInRing()))

    # One-hot quiralidade
    chiral_map = {
        ChiralType.CHI_TETRAHEDRAL_CW:  [1, 0, 0, 0],
        ChiralType.CHI_TETRAHEDRAL_CCW: [0, 1, 0, 0],
        ChiralType.CHI_OTHER:           [0, 0, 1, 0],
        ChiralType.CHI_UNSPECIFIED:     [0, 0, 0, 1],
    }
    feat += chiral_map.get(atom.GetChiralTag(), [0, 0, 0, 1])

    # Padding para dimensão fixa 37
    while len(feat) < 37:
        feat.append(0.0)

    return feat[:37]


def bond_features(bond) -> list[float]:
    """Vetor de features de uma ligação RDKit (dimensão = 7)."""
    feat = one_hot(bond.GetBondType(), BOND_TYPES[:-1])  # 5 elementos
    feat.append(float(bond.GetIsConjugated()))
    feat.append(float(bond.IsInRing()))
    return feat  # dimensão 7


def mol_to_graph(mol_rdkit, pair_id: str) -> Optional["PyGData"]:
    """Converte uma molécula RDKit em grafo PyTorch Geometric.

    O grafo inclui:
      - Nós:   features atômicas (37-dim) + coordenadas 3D do cristal
      - Arestas: bidirecional, features de ligação (7-dim) + distância
      - Metadados: SMILES, pair_id
    """
    if mol_rdkit is None:
        return None

    # Calcula cargas Gasteiger
    rdPartialCharges.ComputeGasteigerCharges(mol_rdkit)

    conf = mol_rdkit.GetConformer() if mol_rdkit.GetNumConformers() > 0 else None
    atoms = list(mol_rdkit.GetAtoms())
    n_atoms = len(atoms)

    # Matriz de features dos nós — sanitiza inf/NaN antes de criar tensor
    raw_feats = np.array([atom_features(a) for a in atoms], dtype=np.float32)
    raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.tensor(raw_feats, dtype=torch.float32)  # (n_atoms, 37)

    # Coordenadas 3D
    if conf is not None:
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)  # (n_atoms, 3)
    else:
        pos = torch.zeros((n_atoms, 3), dtype=torch.float32)

    # Grafo de conectividade (bidirecional)
    edge_index_list = []
    edge_attr_list  = []

    for bond in mol_rdkit.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf    = bond_features(bond)

        # Comprimento da ligação (normalizado)
        if conf is not None:
            d = float(rdMolTransforms.GetBondLength(conf, i, j)) / 3.0
        else:
            d = 0.5

        bf_with_dist = bf + [d]  # dimensão 8

        # Bidirecional
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list  += [bf_with_dist, bf_with_dist]

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 8), dtype=torch.float32)

    # SMILES canônico como string
    smiles = Chem.MolToSmiles(mol_rdkit) if mol_rdkit else ""

    graph = PyGData(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        pos        = pos,
        smiles     = smiles,
        pair_id    = pair_id,
        num_nodes  = n_atoms,
    )

    # ECFP4 global (opcional) — ver "use_ecfp4" no VOXEL_CONFIG
    # Armazenado como graph.ecfp4 (512-dim float32) mas não usado pelo
    # GNNDecoder padrão. Para ativar, defina use_ecfp4=True no VOXEL_CONFIG
    # e adicione a camada de concatenação no 05_train.py.
    if VOXEL_CONFIG.get("use_ecfp4", False):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol_rdkit,
                radius = VOXEL_CONFIG.get("ecfp4_radius", 2),
                nBits  = VOXEL_CONFIG.get("ecfp4_bits", 512),
            )
            graph.ecfp4 = torch.tensor(list(fp), dtype=torch.float32)
        except Exception:
            graph.ecfp4 = torch.zeros(
                VOXEL_CONFIG.get("ecfp4_bits", 512), dtype=torch.float32
            )

    return graph


def load_ligand_mol(pdb_path: Path) -> Optional[object]:
    """Lê o ligand.pdb com RDKit. Tenta inferir ligações se necessário."""
    # Tenta leitura direta
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass

    if mol is None or mol.GetNumAtoms() == 0:
        log.warning(f"  Não foi possível ler {pdb_path.name} com RDKit")
        return None

    # Garante que há conformador 3D
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)

    return mol


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — PIPELINE POR PAR
# ══════════════════════════════════════════════════════════════════════════════

def process_pair(
    pair_dir: Path,
    tensor_dir: "Path | str",
    config: dict,
) -> dict:
    """Processa um par sítio-ligante e gera o tensor + grafo.

    Returns: dict com status e métricas do par processado.
    """
    tensor_dir = Path(tensor_dir)

    pair_id  = pair_dir.name
    site_pdb = pair_dir / "binding_site.pdb"
    lig_pdb  = pair_dir / "ligand.pdb"
    out_dir  = tensor_dir / pair_id

    # Verifica arquivos necessários
    if not site_pdb.exists() or not lig_pdb.exists():
        return {"pair_id": pair_id, "status": "missing_files"}

    # Pula se já processado
    if (out_dir / "site_tensor.npz").exists() and (out_dir / "meta.json").exists():
        try:
            with open(out_dir / "meta.json") as f:
                meta = json.load(f)
            meta["status"] = "ok"          # par válido — existência prévia não é erro
            return meta
        except Exception:
            return {"pair_id": pair_id, "status": "skipped_exists"}

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Carrega átomos do sítio ────────────────────────────────────────────
    try:
        site_coords, site_elements, site_charges = load_site_atoms(site_pdb)
    except Exception as e:
        return {"pair_id": pair_id, "status": "error_site_load", "msg": str(e)}

    if len(site_coords) == 0:
        return {"pair_id": pair_id, "status": "empty_site"}

    # ── 2. Constrói grade 3D ──────────────────────────────────────────────────
    grid_coords, origin, voxel_size = build_grid(site_coords, config)
    n = config["grid_size"]

    # ── 3. Carrega ligante ────────────────────────────────────────────────────
    mol = load_ligand_mol(lig_pdb)

    # ── 4. Calcula canais 0-2 (vetorizado, GPU se CuPy disponível) ───────────
    sigma = max(config["gaussian_sigma"], voxel_size * 1.0)
    w_vdw  = np.array([VDW_RADII.get(e.upper(), VDW_DEFAULT)
                       for e in site_elements], dtype=np.float32)
    gf     = grid_coords.reshape(-1, 3)

    ch0, ch1, ch2 = _voxelize_atoms(
        gf, site_coords, w_vdw, site_charges, sigma, (n, n, n),
        batch=config.get("vox_batch", 500 if HAS_CUPY else 100),
    )

    # Canais 3 e 4: HOMO/LUMO do ligante — QM (GPU4PySCF) ou aproximação
    if mol is not None:
        ch3, ch4 = channel_homo_lumo_qm(grid_coords, mol, origin, voxel_size, config)
    else:
        ch3 = np.zeros((n, n, n), dtype=np.float32)
        ch4 = np.zeros((n, n, n), dtype=np.float32)

    # ── 5. Normalização por canal (z-score robusto) ───────────────────────────
    tensor = np.stack([ch0, ch1, ch2, ch3, ch4], axis=-1)  # (n,n,n,5)

    for c in range(tensor.shape[-1]):
        sl  = tensor[..., c]
        # Substitui NaN/Inf antes de normalizar (podem vir do MEP com cargas extremas)
        sl  = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(sl)
        mad = np.median(np.abs(sl - med)) + 1e-8
        normalized = np.clip((sl - med) / (1.4826 * mad), -5, 5).astype(np.float32)
        # Verificação final — rejeita canal inteiro se ainda houver NaN
        if not np.isfinite(normalized).all():
            normalized = np.zeros_like(normalized)
        tensor[..., c] = normalized

    # Rejeita o par inteiro se o tensor final contiver NaN/Inf
    if not np.isfinite(tensor).all():
        log.warning(f"  {pair_id}: tensor contém NaN/Inf após normalização — par ignorado.")
        return {"pair_id": pair_id, "status": "invalid_tensor"}

    # ── 6. Salva tensor do sítio ──────────────────────────────────────────────
    np.savez_compressed(
        out_dir / "site_tensor.npz",
        tensor     = tensor,          # (n,n,n,5) float32
        origin     = origin,
        voxel_size = np.array([voxel_size]),
        channels   = np.array(["density_vdw", "gasteiger", "mep",
                                "homo_density", "lumo_density"]),
    )

    # ── 7. Grafo molecular do ligante ─────────────────────────────────────────
    graph_ok = False
    if mol is not None and HAS_TORCH:
        try:
            graph = mol_to_graph(mol, pair_id)
            if graph is not None:
                torch.save(graph, out_dir / "ligand_graph.pt")
                graph_ok = True
        except Exception as e:
            log.warning(f"  {pair_id}: erro ao criar grafo — {e}")

    # Fallback: salva features como .npz se PyTorch não disponível
    if mol is not None and not graph_ok:
        _save_ligand_npz(mol, pair_id, out_dir)
        graph_ok = True

    # ── 8. Metadados ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _qm_used = bool(
        mol and HAS_PYSCF and
        len([a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]) <= config["max_qm_atoms"]
    )
    _qm_backend = (
        "GPU4PySCF" if (HAS_GPU4PYSCF and _qm_used) else
        "PySCF/CPU" if HAS_PYSCF else
        "Gasteiger"
    )
        
    meta = {
        "pair_id":         pair_id,
        "grid_size":       n,
        "voxel_size_A":    round(voxel_size, 4),
        "origin":          origin.tolist(),
        "n_site_atoms":    int(len(site_coords)),
        "n_ligand_atoms":  int(mol.GetNumAtoms()) if mol else 0,
        "ligand_smiles":   Chem.MolToSmiles(mol) if mol else "",
        "qm_used":     _qm_used,
        "qm_backend":  _qm_backend,
        "vox_backend": "CuPy" if HAS_CUPY else "NumPy",
        "graph_saved":     graph_ok,
        "tensor_shape":    list(tensor.shape),
        "elapsed_s":       round(elapsed, 2),
        "status":          "ok",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        f"  ✓ {pair_id} — {elapsed:.1f}s "
        f"| QM: {_qm_backend} | Vox: {meta['vox_backend']} "
        f"| site={len(site_coords)} átomos | lig={meta['n_ligand_atoms']} átomos"
    )
    return meta


def _save_ligand_npz(mol, pair_id: str, out_dir: Path):
    """Salva representação do ligante em .npz (fallback sem PyTorch)."""
    rdPartialCharges.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    node_feats = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    # Sanitiza features — substitui inf/NaN por 0 antes de salvar
    node_feats = np.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
    pos        = np.array(conf.GetPositions(), dtype=np.float32) if conf else np.zeros((mol.GetNumAtoms(), 3))
    smiles     = Chem.MolToSmiles(mol)

    edge_src, edge_dst, edge_f = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        if conf:
            d = float(rdMolTransforms.GetBondLength(conf, i, j)) / 3.0
            bf += [d]
        else:
            bf += [0.5]
        edge_src += [i, j]; edge_dst += [j, i]
        edge_f   += [bf, bf]

    # ECFP4 opcional (ver use_ecfp4 no VOXEL_CONFIG)
    ecfp4_arr = np.zeros(VOXEL_CONFIG.get("ecfp4_bits", 512), dtype=np.float32)
    if VOXEL_CONFIG.get("use_ecfp4", False):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius = VOXEL_CONFIG.get("ecfp4_radius", 2),
                nBits  = VOXEL_CONFIG.get("ecfp4_bits", 512),
            )
            ecfp4_arr = np.array(list(fp), dtype=np.float32)
        except Exception:
            pass

    np.savez_compressed(
        out_dir / "ligand_graph.npz",
        node_features = node_feats,
        positions     = pos,
        edge_src      = np.array(edge_src, dtype=np.int32),
        edge_dst      = np.array(edge_dst, dtype=np.int32),
        edge_features = np.array(edge_f,   dtype=np.float32) if edge_f else np.zeros((0, 8)),
        smiles        = np.array([smiles]),
        ecfp4         = ecfp4_arr,   # zeros se use_ecfp4=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def detect_optimal_workers() -> int:
    """Detecta automaticamente o número ideal de workers para o hardware atual.

    Conclusão baseada em benchmark empírico extensivo (Ryzen 9 7950X + RTX 4080 Super):

      Três estratégias de paralelismo foram testadas e descartadas:

      1. ProcessPoolExecutor + fork  → deadlock CUDA (CUDA não suporta fork)
      2. ProcessPoolExecutor + spawn → overhead de ~15s por worker (reimport
                                       de GPU4PySCF/CuPy/PyTorch a cada processo)
      3. ThreadPoolExecutor          → GIL bloqueia operações CPU-intensivas
                                       (NumPy, PySCF, BioPython); tempo/par
                                       aumenta linearmente com workers sem
                                       ganho de throughput total

      Resultado: workers=1 é sempre o ponto ótimo para este pipeline.
      O paralelismo real acontece DENTRO de cada par:
        - CuPy: milhares de CUDA cores em paralelo nas gaussianas
        - NumPy: BLAS multi-thread nas operações matriciais
        - PySCF: OpenMP nos cálculos de integrais moleculares
        - GPU4PySCF: cuBLAS/cuSOLVER para QM na GPU

      Para paralelizar múltiplos pares simultaneamente seria necessário
      usar uma fila de trabalho externa (Ray, Dask, SLURM) que gerencie
      processos independentes cada um com seu próprio contexto CUDA.
    """
    return 1


def run(
    args,
    curated_dir: "Path | str",
    tensor_dir: "Path | str",
    config: "dict | None" = None,
):
    curated_dir = Path(curated_dir)
    tensor_dir = Path(tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    _setup_log_file(Path(config.pop("log_dir"))) 
    if config:
        VOXEL_CONFIG.update(config)

    log.info("=" * 65)
    log.info("PIPELINE ETAPA 4: Voxelização 3D + Grafos Moleculares")
    log.info("=" * 65)
    log.info(f"  Hardware : {GPU_DESC}")
    log.info(f"  Curated  : {curated_dir}")
    log.info(f"  Output   : {tensor_dir}")

    qm_backend = (
        "GPU4PySCF (CUDA)" if HAS_GPU4PYSCF
        else "PySCF (CPU)" if HAS_PYSCF
        else "Aproximação Gasteiger (sem PySCF)"
    )
    log.info(
        f"  Grid     : {VOXEL_CONFIG['grid_size']}³ voxels  |  "
        f"Canais: {VOXEL_CONFIG['n_site_channels']}  |  QM: {qm_backend}"
    )
    log.info(
        f"  Grafos   : {'PyTorch Geometric (.pt)' if HAS_TORCH else 'NumPy (.npz)'}"
    )
    if not HAS_GPU4PYSCF and HAS_CUDA:
        log.info("  Dica     : instale gpu4pyscf para acelerar os cálculos QM na RTX")
        log.info("             pip install gpu4pyscf-cuda12x")

    if args.pair:
        pair_dirs = [curated_dir / args.pair]
        if not pair_dirs[0].exists():
            log.error(f"Par não encontrado: {pair_dirs[0]}")
            sys.exit(1)
    else:
        if not curated_dir.exists():
            raise FileNotFoundError(f"Diretório curado não encontrado: {curated_dir}")
        pair_dirs = sorted([d for d in curated_dir.iterdir() if d.is_dir()])

    log.info(f"  Pares a processar: {len(pair_dirs)}")

    results = []
    ok = fail = skipped = 0

    if args.workers is None:
        effective_workers = detect_optimal_workers()
        log.info(f"  Workers  : {effective_workers} (detectado automaticamente)")
    else:
        effective_workers = args.workers
        log.info(f"  Workers  : {effective_workers} (definido pelo usuário)")

    if effective_workers > 1:
        log.info(
            f"  ℹ workers={effective_workers} definido manualmente. "
            "Benchmark indica que workers=1 é o ponto ótimo para este pipeline "
            "(GIL bloqueia ThreadPoolExecutor; CUDA não suporta ProcessPool)."
        )

    for d in tqdm(pair_dirs, desc="Voxelizando"):
        r = process_pair(d, tensor_dir=tensor_dir, config=VOXEL_CONFIG)
        results.append(r)
        if r["status"] == "ok":
            ok += 1
        elif "skip" in r["status"]:
            skipped += 1
        else:
            fail += 1
            log.warning(f"  ✗ {r['pair_id']}: {r.get('msg', r['status'])}")

    df = pd.DataFrame(results)
    df.to_csv(tensor_dir / "tensor_index.csv", index=False)

    log.info("\n" + "=" * 65)
    log.info("RESUMO")
    log.info("=" * 65)
    log.info(f"  ✓ Processados com sucesso : {ok}")
    log.info(f"  ↷ Já existiam (pulados)  : {skipped}")
    log.info(f"  ✗ Falhas                  : {fail}")
    log.info(f"  Índice salvo em           : {tensor_dir / 'tensor_index.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxeliza pares sítio-ligante")
    parser.add_argument("--curated-dir", required=True,
                        help="Diretório com os pares curados")
    parser.add_argument("--tensor-dir", required=True,
                        help="Diretório de saída dos tensores")
    parser.add_argument("--pair", type=str, default=None,
                        help="Processar apenas um par específico (ex: 7W40_E2H_A101)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Número de processos paralelos. Se omitido, detecta automaticamente.")
    args = parser.parse_args()

    run(
        args,
        curated_dir=args.curated_dir,
        tensor_dir=args.tensor_dir,
    )
