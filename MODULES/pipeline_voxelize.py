"""
Pipeline - Etapa 4: Voxelização 3D + Propriedades Quânticas
============================================================
Lê os pares curados (binding_site.pdb + ligand.pdb) e gera:

  ┌─ Sítio de ligação ──────────────────────────────────────────────────────┐
  │  Tensor 4D: (X, Y, Z, C_site) onde C_site = 5 canais                   │
  │    Canal 0 — Densidade atômica (ocupância de vdW gaussiana)             │
  │    Canal 1 — Carga parcial (Gasteiger) interpolada na grade             │
  │    Canal 2 — MEP (Potencial Eletrostático Molecular) — Coulomb          │
  │    Canal 3 — Campo doador de elétrons (HOMO-like, semi-empírico)        │
  │    Canal 4 — Campo aceptor de elétrons (LUMO-like, semi-empírico)       │
  │                                                                           │
  │  Os 5 canais são computados EXCLUSIVAMENTE a partir dos átomos do        │
  │  próprio bolso de ligação (binding_site.pdb) — nunca do ligante. Isso    │
  │  garante que o tensor do sítio seja idêntico em treino e em inferência   │
  │  (pipeline_generate.py constrói o mesmo tensor sem precisar de nenhum    │
  │  ligante de referência). Ver channel_site_donor_acceptor().              │
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

Notas sobre os canais 2-4 do sítio:
    • Canal 2 (MEP) é a soma de potenciais de Coulomb das cargas parciais
      dos átomos do sítio (escala bem para proteínas inteiras).
    • Canais 3-4 (doador/aceptor, HOMO/LUMO-like) são calculados a partir
      dos átomos do SÍTIO — nunca do ligante. Este é o substituto de uma
      versão anterior que calculava HOMO/LUMO via QM (RHF/STO-3G) sobre o
      LIGANTE co-cristalizado: isso vazava a "resposta" para dentro do
      condicionamento do encoder durante o treino, já que na geração real
      não existe ligante de referência e esses canais seriam sempre zero
      (ver pipeline_generate.pdb_to_site_tensor). As funções antigas
      (channel_homo_lumo_qm/_homo_lumo_approx) foram mantidas no código
      mas não são mais usadas na construção do tensor.
    • Como o sítio é protonado por resíduo (ver protonate_residue, pH
      configurável), há três níveis de teoria disponíveis para os canais
      3-4 (config["site_qm_method"]):
        1. "rhf_sto3g_fragments" (padrão) — RHF/STO-3G real por RESÍDUO
           (fragmentação tipo FMO, cada fragmento protonado e capado com H
           nas bordas cortadas — sempre pequeno o bastante para caber sob
           max_qm_atoms, ao contrário de tentar o sítio inteiro de uma vez,
           que seria inviável). Densidade condensada por átomo via
           população de Mulliken do HOMO/LUMO. Mais rigoroso, porém lento
           em escala (ordem de minutos por par).
        2. "gfn_xtb_fragments" (RECOMENDADO se o binário `xtb` estiver
           disponível) — GFN-xTB por RESÍDUO, em paralelo (várias chamadas
           de xtb simultâneas via ThreadPoolExecutor, threads por chamada
           limitadas para não estourar os núcleos — ver
           _pick_fragment_parallelism). Usa os índices de Fukui condensados
           (f+/f-) como doador/aceptor. Mesma unidade de fragmentação do
           RHF/STO-3G — validado: ~0.1-0.2s por sítio inteiro (dezenas de
           resíduos), robusto mesmo em sítios grandes/multi-cadeia.
           Requer o binário `xtb` no PATH (`conda install -c conda-forge
           xtb`, ou crie um env dedicado e link o binário em ~/.local/bin).
        3. "heuristic" — aproximação por eletronegatividade de Pauling +
           carga parcial (mais rápida, sem QM) — também o fallback
           automático das demais quando o cálculo real falha (por
           resíduo/fragmento nos métodos fragmentados).

      Havia um quarto nível, "gfn_xtb_full" (GFN-xTB no sítio protonado
      INTEIRO, sem fragmentar), removido do código: travava com
      segmentation fault em sítios reais (o xtb detecta múltiplos
      fragmentos moleculares desconectados, já que o sítio é uma
      concatenação de resíduos capados independentemente, e crasha ao
      tentar restringi-los com uma força de "constraining FC"), além de
      convergência lenta (minutos) mesmo quando não crashava. Configs
      antigas com site_qm_method="gfn_xtb_full" caem no fallback
      "heuristic" (valor não reconhecido, ver channel_site_orbital_field).
    • Para ligantes com > MAX_QM_ATOMS átomos pesados na etapa de QM do
      ligante (hoje não usada no tensor, ver acima), o cálculo cairia
      para a aproximação semi-empírica correspondente.

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
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Testa o padrão REAL usado em _voxelize_atoms, não só o einsum. Um
    # teste que só exercita einsum (via cuBLAS) pode passar mesmo quando o
    # NVRTC — o compilador JIT que toda redução/kernel elementwise usa —
    # está quebrado ou ausente (ex.: libnvrtc.so.12 não encontrado). Nesse
    # caso o smoke test antigo dava HAS_CUPY=True, e o primeiro par
    # processado de verdade travava o programa com um traceback bruto do
    # CuPy — incompreensível para quem não programa, e só descoberto horas
    # depois de iniciada a curadoria/voxelização.
    _t   = cp.zeros((2, 4), dtype=cp.float32)
    _d2  = (_t * _t).sum(axis=-1)          # redução NVRTC (o ponto de falha real)
    _e   = cp.exp(-_d2)                    # kernel elementwise NVRTC
    cp.einsum('ij->j', _t)                 # cuBLAS — não cobre o caso acima sozinho
    del _t, _d2, _e
    HAS_CUPY = bool(HAS_CUDA)
    if HAS_CUPY:
        cp.cuda.Device(0).use()
except Exception as _cupy_exc:
    cp       = None
    HAS_CUPY = False
    warnings.warn(
        f"CuPy indisponível ou com falha de compilação JIT (NVRTC) — "
        f"usando NumPy (CPU) para a voxelização, mais lento. "
        f"Causa: {type(_cupy_exc).__name__}: {_cupy_exc}"
    )

def _xp():
    """Retorna cupy se disponível, senão numpy."""
    return cp if HAS_CUPY else np


def _qm_to_numpy(arr):
    """Converte para numpy resultados de QM que podem vir em cupy.

    GPU4PySCF mantém mo_coeff/mo_occ em cupy.ndarray; PySCF/CPU já retorna
    numpy. Como pyscf_mol.intor(...) (ex. matriz de overlap) é sempre CPU/
    numpy, misturar os dois sem converter lança TypeError — usada em
    _fragment_qm_weights antes de qualquer operação com arrays numpy.
    """
    if HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

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

    # Limite de átomos pesados para cálculo QM completo (RHF/STO-3G).
    # Usado tanto para o ligante (channel_homo_lumo_qm, hoje não conectado
    # ao tensor — ver notas no topo do arquivo) quanto para CADA FRAGMENTO
    # residual do sítio em site_qm_method="rhf_sto3g_fragments" (um resíduo
    # protonado raramente passa de ~25 átomos, então o limite quase sempre
    # é satisfeito — ao contrário de tentar o sítio inteiro de uma vez).
    # Acima disso usa aproximação semi-empírica (eletronegatividade+carga).
    "max_qm_atoms": 60,

    # Base QM (STO-3G é a mais rápida; use 3-21G para mais precisão)
    "qm_basis": "sto-3g",

    # ── Protonação do sítio (canais 3-4) ───────────────────────────────────
    # pH usado para decidir o estado de protonação de resíduos ionizáveis
    # (Asp/Glu/His/Cys/Tyr/Lys/Arg) via regra de limiar de Henderson-
    # Hasselbalch sobre pKa de tabela — ver RESIDUE_PKA e protonate_residue.
    "ph": 7.4,

    # False = comportamento legado: sem protonação, sem H explícitos, carga
    # aproximada por elemento (ver load_site_atoms). True = load_site_atoms_
    # protonated, com H explícitos e cargas Gasteiger reais por resíduo.
    "protonate_site": True,

    # Método para os canais 3-4 (doador/aceptor de elétrons do sítio):
    #   "rhf_sto3g_fragments" — QM real por resíduo (fragmentação FMO-like)
    #   "gfn_xtb_fragments"   — GFN-xTB por resíduo, em paralelo (requer o
    #                           binário `xtb` no PATH — não instalado por
    #                           padrão, ver _xtb_fragment_weights)
    #   "heuristic"           — eletronegatividade de Pauling + carga (sem
    #                           QM; também o fallback automático das outras
    #                           duas quando o cálculo real falha)
    "site_qm_method": "rhf_sto3g_fragments",

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

# Chaves de VOXEL_CONFIG que realmente mudam os VALORES do tensor do sítio
# (grade/canais 0-4) — únicas que precisam ser reproduzidas EXATAMENTE na
# geração (Etapa 9) para que o encoder treinado receba um tensor da mesma
# distribuição do treino (ver process_pair/pdb_to_site_tensor). Ficam de
# fora de propósito: "vox_batch" (só afeta velocidade, não o resultado),
# "n_site_channels" (constante arquitetural, não é escolha do usuário) e
# "use_ecfp4"/"ecfp4_*" (afetam o grafo do ligante no treino, não o tensor
# do sítio consumido pelo encoder).
VOXEL_CONFIG_TENSOR_KEYS = (
    "grid_size", "padding_A", "gaussian_sigma", "max_qm_atoms",
    "qm_basis", "ph", "protonate_site", "site_qm_method",
)


def save_voxel_config_snapshot(tensor_dir: "Path | str", config: "dict | None" = None) -> None:
    """Grava um retrato da config de voxelização REALMENTE usada nesta rodada
    (ver VOXEL_CONFIG_TENSOR_KEYS) em <tensor_dir>/voxel_config.json.

    Consumido por pipeline_train.train() (que embute esse retrato no
    best_model.pt salvo) e por pipeline_generate.load_model() (que o lê de
    volta do checkpoint) — fecha o elo que faltava entre "com que config o
    tensor de treino foi construído" e "com que config o tensor de um sítio
    novo precisa ser construído na geração", sem depender do usuário lembrar
    e repetir manualmente os mesmos valores na Etapa 9 da interface.
    """
    cfg = config if config is not None else VOXEL_CONFIG
    snapshot = {k: cfg[k] for k in VOXEL_CONFIG_TENSOR_KEYS if k in cfg}
    out_path = Path(tensor_dir) / "voxel_config.json"
    try:
        out_path.write_text(json.dumps(snapshot, indent=2))
        log.info(f"  Retrato da config de voxelização salvo em {out_path}")
    except Exception as e:
        log.warning(f"  Não foi possível salvar {out_path}: {e}")


# Raios de van der Waals por elemento (Å) — Bondi 1964
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 2.05, "ZN": 1.39, "MG": 1.73, "CA": 2.31, "MN": 1.97,
    "CU": 1.40, "NI": 1.63, "CO": 1.70, "SE": 1.90,
}
VDW_DEFAULT = 1.70

# Eletronegatividade de Pauling por elemento — usada para estimar caráter
# doador/aceptor de elétrons (aproximação semi-empírica de HOMO/LUMO nos
# canais 3-4 do tensor do sítio; ver channel_site_donor_acceptor).
ELECTRONEGATIVITY = {
    "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "P": 2.19,
    "F": 3.98, "CL": 3.16, "BR": 2.96, "I": 2.66,
    "FE": 1.83, "ZN": 1.65, "MG": 1.31, "SE": 2.55, "CA": 1.00,
    "MN": 1.55, "CU": 1.90, "NI": 1.91, "CO": 1.88,
}
ELECTRONEGATIVITY_DEFAULT = 2.50

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
    compute_mep: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula três campos gaussianos vetorizados a partir de pesos por átomo.

    Genérico: não é específico dos canais 0-2 — qualquer par de pesos por
    átomo (weights_density, weights_charge) pode ser passado para obter
    dois campos gaussianos independentes (ex.: doador/aceptor de elétrons
    em channel_site_donor_acceptor). O terceiro retorno (mep) é sempre a
    soma de Coulomb `weights_charge / dist`; passe compute_mep=False para
    pular esse termo quando ele não for usado (economiza uma sqrt+divisão
    por átomo/voxel).

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

        # Canal 2: MEP coulombiano  q / |r - r_i|  (pulado se compute_mep=False)
        if compute_mep:
            dist = xp.sqrt(xp.clip(dist2, 0.01, None))
            mep += xp.einsum('ij->j', wc_b[:, None] / dist)

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


def _electronegativity_donor_acceptor_weights(
    elements: list[str],
    charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Núcleo de channel_site_donor_acceptor: pesos doador/aceptor por átomo
    a partir de eletronegatividade de Pauling + carga parcial — sem
    espalhar na grade ainda. Extraído para reuso como fallback POR
    FRAGMENTO em channel_site_orbital_field (quando o QM real falha só
    para um resíduo, o resto do sítio não precisa cair nessa aproximação).

      Doador (HOMO-like): átomos eletronegativamente pobres e/ou com carga
        parcial negativa — regiões nucleofílicas, propensas a doar
        densidade eletrônica.
      Aceptor (LUMO-like): átomos eletronegativos e/ou com carga parcial
        positiva — regiões eletrofílicas, propensas a aceitar densidade
        eletrônica.
    """
    elec = np.array(
        [ELECTRONEGATIVITY.get(e.upper(), ELECTRONEGATIVITY_DEFAULT)
         for e in elements],
        dtype=np.float32,
    )
    q = np.asarray(charges, dtype=np.float32)
    donor_w    = np.clip((3.0 - elec) / 3.0 + np.clip(-q, 0.0, None), 0.0, None)
    acceptor_w = np.clip((elec - 2.0) / 2.0 + np.clip(q, 0.0, None),  0.0, None)
    return donor_w, acceptor_w


def channel_site_donor_acceptor(
    grid_flat: np.ndarray,
    site_coords: np.ndarray,
    site_elements: list[str],
    site_charges: np.ndarray,
    sigma: float,
    grid_shape: tuple,
    batch: int = 500 if HAS_CUPY else 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Canais 3-4 — Campo doador/aceptor de elétrons do SÍTIO (HOMO/LUMO-like),
    nível "heuristic" (eletronegatividade + carga, sem QM — ver
    _electronegativity_donor_acceptor_weights). Espalha os pesos na grade
    com as mesmas gaussianas 3D dos canais 0-2 (reaproveita
    _voxelize_atoms, inclusive aceleração CuPy).

    Este é o nível mais rápido dos três disponíveis em
    config["site_qm_method"] (ver channel_site_orbital_field para os
    níveis "rhf_sto3g_fragments" e "gfn_xtb_fragments") — e também o
    fallback automático dos outros dois quando o cálculo QM real falha ou
    está indisponível.

    Por que a partir do SÍTIO e não do ligante:
      Esta é a substituição direta de uma versão anterior que calculava
      HOMO/LUMO via QM (RHF/STO-3G) do ligante co-cristalizado. Aquilo
      causava um vazamento de dados treino↔inferência: o encoder recebia
      como entrada a densidade eletrônica da própria molécula que deveria
      gerar, algo que não existe na geração real (não há ligante de
      referência). Calculando o campo a partir dos átomos do próprio
      sítio (sempre disponíveis, com ou sem ligante), o tensor fica
      idêntico em treino e em inferência — sem vazamento — e ainda carrega
      sinal quimicamente relevante: análogo aos mapas de complementaridade
      doador/aceptor de métodos clássicos como GRID (Goodford, 1985) e a
      featurizations voxelizadas de bolsos de ligação usadas em ML
      estrutural (ex. Ragoza et al., 2017).
    """
    donor_w, acceptor_w = _electronegativity_donor_acceptor_weights(
        site_elements, site_charges
    )
    donor, acceptor, _ = _voxelize_atoms(
        grid_flat, site_coords, donor_w, acceptor_w, sigma, grid_shape,
        batch=batch, compute_mep=False,
    )
    return donor, acceptor


def _fragment_qm_weights(
    mol: "Chem.Mol", config: dict,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Roda RHF/STO-3G no fragmento residual protonado (ver
    protonate_residue) e retorna pesos doador/aceptor CONDENSADOS por
    átomo — população de Mulliken do HOMO/LUMO — na mesma escala usada
    pela aproximação semi-empírica, prontos para serem espalhados na
    grade pela mesma _voxelize_atoms.

    Por que população condensada e não um cubo 3D real (como
    channel_homo_lumo_qm fazia para o ligante): somar vários cubos
    pequenos, um por resíduo, exigiria realinhar cada grade local na
    grade global do sítio — complexidade e custo extras sem ganho real,
    já que o que importa para o campo espalhado por gaussianas é o peso
    por átomo, não a forma exata do orbital dentro do fragmento. A
    população de Mulliken (c · S·c, com S = matriz de overlap AO) é o
    jeito padrão de condensar um orbital molecular em contribuições
    atômicas.

    Retorna None se PySCF ausente, fragmento maior que
    config["max_qm_atoms"], SCF sem orbital virtual, ou qualquer exceção
    (não converge, geometria degenerada etc.) — o chamador cai para o
    peso de eletronegatividade nesse fragmento específico.
    """
    if not HAS_PYSCF:
        return None
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy > config.get("max_qm_atoms", 60):
        return None

    try:
        conf = mol.GetConformer()
        atom_str = []
        for atom in mol.GetAtoms():
            xyz = conf.GetAtomPosition(atom.GetIdx())
            atom_str.append(f"{atom.GetSymbol()} {xyz.x:.6f} {xyz.y:.6f} {xyz.z:.6f}")

        charge      = Chem.GetFormalCharge(mol)
        n_electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
        spin        = n_electrons % 2   # 0 = singleto, 1 = dubleto (radical)

        pyscf_mol = gto.Mole()
        pyscf_mol.atom    = "\n".join(atom_str)
        pyscf_mol.basis   = config.get("qm_basis", "sto-3g")
        pyscf_mol.charge  = charge
        pyscf_mol.spin    = spin
        pyscf_mol.verbose = 0
        pyscf_mol.build()

        if HAS_GPU4PYSCF:
            mf = gpu_scf.RHF(pyscf_mol)
        else:
            mf = scf.RHF(pyscf_mol) if spin == 0 else scf.ROHF(pyscf_mol)
        mf.kernel()

        # GPU4PySCF mantém mo_coeff/mo_occ em cupy — converte para numpy
        # antes de misturar com S (sempre CPU/numpy, via pyscf_mol.intor).
        mo_occ    = _qm_to_numpy(mf.mo_occ)
        mo_coeff  = _qm_to_numpy(mf.mo_coeff)
        homo_i = int(np.where(mo_occ > 0)[0][-1])
        lumo_i = homo_i + 1
        if lumo_i >= mo_coeff.shape[1]:
            return None   # sem orbital virtual (fragmento saturado, raro)

        S = pyscf_mol.intor("int1e_ovlp")
        ao_slices = pyscf_mol.aoslice_by_atom()
        homo_c = mo_coeff[:, homo_i]
        lumo_c = mo_coeff[:, lumo_i]
        homo_pop_ao = homo_c * (S @ homo_c)   # população de Mulliken por AO
        lumo_pop_ao = lumo_c * (S @ lumo_c)

        n_atoms = mol.GetNumAtoms()
        donor_w    = np.zeros(n_atoms, dtype=np.float32)   # HOMO ≈ doador
        acceptor_w = np.zeros(n_atoms, dtype=np.float32)   # LUMO ≈ aceptor
        for i, row in enumerate(ao_slices):
            a0, a1 = int(row[2]), int(row[3])
            donor_w[i]    = float(homo_pop_ao[a0:a1].sum())
            acceptor_w[i] = float(lumo_pop_ao[a0:a1].sum())

        # Mulliken pode dar população levemente negativa em bases polarizadas
        # — não faz sentido como peso de intensidade gaussiana, clipa em 0.
        return np.clip(donor_w, 0.0, None), np.clip(acceptor_w, 0.0, None)
    except Exception as e:
        log.debug(f"  QM fragmento falhou: {type(e).__name__}: {e}")
        return None


def _xtb_bin() -> "str | None":
    """Localiza o binário `xtb`, com aviso único (não um por chamada — este
    dispatcher pode ser chamado dezenas de vezes por sítio no modo
    fragmentado).

    Ordem de busca: XTB_HOME, PATH, e <projeto>/tools/xtb*/bin/xtb — este
    último é onde CODEEP_requirements.install_xtb() o instala. Mesma ordem
    usada por CODEEP_requirements.detect_xtb(), para que o que a tela de
    Install Requirements mostra seja o que este pipeline de fato encontra.
    """
    env_home = os.environ.get("XTB_HOME")
    if env_home:
        cand = Path(env_home) / "bin" / "xtb"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    xtb_bin = shutil.which("xtb")
    if xtb_bin is None:
        for base in (PROJECT_ROOT / "tools", Path.home(), Path("/opt")):
            if not base.is_dir():
                continue
            try:
                for cand in sorted(base.glob("xtb*/bin/xtb")):
                    if cand.is_file() and os.access(cand, os.X_OK):
                        xtb_bin = str(cand)
                        break
            except OSError:
                continue
            if xtb_bin:
                break

    if xtb_bin is None:
        log.warning(
            "  xtb não encontrado — instale pelo menu Install Requirements "
            "(baixa um binário estático, sem dependências) ou manualmente "
            "de https://github.com/grimme-lab/xtb. Usando aproximação "
            "semi-empírica."
        )
    return xtb_bin


def _run_xtb_vfukui(
    elements: list[str],
    coords: np.ndarray,
    charge: int,
    config: dict,
    xtb_bin: str,
    omp_threads: "int | None" = None,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Núcleo compartilhado: escreve um .xyz, roda `xtb --vfukui` via
    subprocess e extrai os índices de Fukui condensados por átomo:
        f(-) → suscetibilidade a ataque ELETROFÍLICO → caráter DOADOR (HOMO-like)
        f(+) → suscetibilidade a ataque NUCLEOFÍLICO → caráter ACEPTOR (LUMO-like)

    Usado por _xtb_fragment_weights (um resíduo por vez, ver
    channel_site_orbital_field/"gfn_xtb_fragments").

    omp_threads limita OMP_NUM_THREADS desta chamada — necessário quando
    várias chamadas rodam em paralelo (modo fragmentado, via
    ThreadPoolExecutor) para não estourar os núcleos disponíveis: o xtb
    usa todos os threads da máquina por padrão em cada chamada.

    Retorna None (→ chamador cai para a aproximação semi-empírica) se o
    cálculo falhar, não convergir ou não puder ser interpretado.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = Path(tmpdir) / "site.xyz"
            with open(xyz_path, "w") as f:
                f.write(f"{len(elements)}\n")
                f.write("CODEEP\n")
                for elem, xyz in zip(elements, coords):
                    f.write(f"{elem.capitalize():2s} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")

            env = os.environ.copy()
            if omp_threads is not None:
                env["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))

            # encoding/errors explícitos: a saída do xtb tem caracteres não-
            # ASCII (Å etc.) e text=True sozinho usa a codificação preferida
            # do locale do processo, que pode ser ASCII/C dependendo de como
            # o pipeline é lançado (systemd, cron, subprocess sem ambiente
            # herdado…) — decodificar sem isso derruba o cálculo inteiro com
            # UnicodeDecodeError, silenciosamente capturado pelo except.
            result = subprocess.run(
                [xtb_bin, str(xyz_path), "--gfn", "2",
                 "--chrg", str(int(charge)), "--vfukui"],
                cwd=tmpdir, capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=env,
                timeout=config.get("xtb_timeout_s", 600),
            )
            if result.returncode != 0:
                log.debug(f"  xtb falhou (código {result.returncode}): "
                          f"{result.stderr[-500:] if result.stderr else '(sem stderr)'}")
                return None

            # xtb >= 6.x (validado com 6.7.1) não grava mais fukui.txt em
            # disco — os índices saem só no stdout, sob "Fukui functions:".
            # Mantemos a leitura do arquivo como fallback para versões/
            # configs que ainda o gravem.
            fukui_path = Path(tmpdir) / "fukui.txt"
            if fukui_path.exists():
                parsed = _parse_xtb_fukui(
                    fukui_path.read_text(encoding="utf-8", errors="replace"),
                    len(elements),
                )
            else:
                parsed = _parse_xtb_fukui(result.stdout, len(elements))
            if parsed is None:
                log.debug("  xtb: não foi possível extrair os índices de Fukui "
                          "da saída (nem fukui.txt nem stdout)")
            return parsed
    except Exception as e:
        log.debug(f"  xtb: {type(e).__name__}: {e}")
        return None


def _pick_fragment_parallelism(n_fragments: int, config: dict) -> tuple[int, int]:
    """Decide quantos fragmentos rodar em paralelo (n_workers) e quantas
    threads OpenMP por chamada de xtb (threads_per_call), de forma que
    n_workers * threads_per_call fique perto do total de CPUs — sem isso,
    cada chamada de xtb tentaria usar todos os núcleos sozinha e chamadas
    concorrentes disputariam os mesmos núcleos (oversubscription), ficando
    mais lento que rodar em série.

    Fragmentos são pequenos (um resíduo, ~10-30 átomos) e escalam mal com
    muitas threads (overhead de paralelizar uma álgebra linear pequena
    supera o ganho) — por isso o critério prioriza mais fragmentos
    concorrentes com poucas threads cada, em vez de poucos fragmentos com
    muitas threads.
    """
    cpu_total = config.get("xtb_cpu_total") or os.cpu_count() or 1
    max_workers = config.get("xtb_max_workers")  # override manual, opcional
    n_workers = max(1, min(n_fragments, cpu_total, max_workers or cpu_total))
    threads_per_call = max(1, cpu_total // n_workers)
    return n_workers, threads_per_call


def _xtb_fragment_weights(
    mol: "Chem.Mol", config: dict, xtb_bin: str, omp_threads: int,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Roda GFN-xTB num único fragmento residual protonado (mesma unidade
    de _fragment_qm_weights, o equivalente em RHF/STO-3G) e retorna os
    índices de Fukui condensados por átomo, prontos para _voxelize_atoms.

    Cada fragmento é uma molécula única e covalentemente conectada
    (resíduo protonado e capado nas bordas cortadas por protonate_residue)
    — ao contrário do sítio inteiro concatenado (abordagem tentada e
    removida: o xtb detecta múltiplos fragmentos desconectados nesse caso
    e trava com segmentation fault), não dispara esse problema. Retorna
    None se o fragmento exceder config["max_qm_atoms"], se o cálculo
    falhar ou não convergir — o chamador cai para a aproximação
    semi-empírica só nesse fragmento.
    """
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_heavy > config.get("max_qm_atoms", 60):
        return None
    try:
        conf = mol.GetConformer()
        elements = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = np.array(
            [list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()],
            dtype=np.float32,
        )
        charge = Chem.GetFormalCharge(mol)
        return _run_xtb_vfukui(elements, coords, charge, config, xtb_bin, omp_threads)
    except Exception as e:
        log.debug(f"  xtb fragmento falhou: {type(e).__name__}: {e}")
        return None


def _parse_xtb_fukui(text: str, n_atoms: int) -> "tuple[np.ndarray, np.ndarray] | None":
    """Extrai os índices de Fukui condensados de `xtb --vfukui`, a partir
    do texto de fukui.txt (versões antigas) OU do stdout completo do xtb
    (validado com xtb 6.7.1, que só imprime no terminal). Formato real,
    sob o cabeçalho "Fukui functions:":

        Fukui functions:
             #        f(+)     f(-)     f(0)
             1N       0.081    0.132    0.106
             2C       0.018    0.011    0.015
             ...

    f(+) → caráter aceptor (LUMO-like); f(-) → caráter doador (HOMO-like).
    Como o stdout do xtb tem várias outras tabelas numéricas (iterações de
    SCF etc.) que colidiriam com um parsing ingênuo linha-a-linha do texto
    inteiro, isolamos primeiro o bloco logo após "Fukui functions:" e
    paramos no fim dele (linha em branco/separador ou índice fora de
    sequência), em vez de escanear o arquivo/stdout inteiro.
    """
    try:
        lines = text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if "Fukui functions:" in line:
                start = i + 1
                break
        if start is None:
            # fukui.txt de versões antigas pode não ter esse cabeçalho —
            # cai para o comportamento antigo (escaneia tudo).
            start = 0

        while start < len(lines) and lines[start].strip().startswith("#"):
            start += 1

        donor_w    = np.zeros(n_atoms, dtype=np.float32)
        acceptor_w = np.zeros(n_atoms, dtype=np.float32)
        idx = 0
        for line in lines[start:]:
            parts = line.split()
            if len(parts) < 4:
                break
            head = parts[0]
            if not (head[0].isdigit() or (head[0] == "-" and head[1:2].isdigit())):
                break
            try:
                fplus, fminus = float(parts[-3]), float(parts[-2])
            except (ValueError, IndexError):
                break
            if idx >= n_atoms:
                break
            acceptor_w[idx] = max(0.0, fplus)
            donor_w[idx]    = max(0.0, fminus)
            idx += 1
        if idx == 0:
            return None
        if idx != n_atoms:
            log.debug(f"  xtb: bloco de Fukui tinha {idx} átomos, esperado "
                      f"{n_atoms} — usando os {idx} primeiros.")
        return donor_w, acceptor_w
    except Exception:
        return None


def channel_site_orbital_field(
    grid_flat: np.ndarray,
    site_coords: np.ndarray,
    site_elements: list[str],
    site_charges: np.ndarray,
    fragments: list,
    sigma: float,
    grid_shape: tuple,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Dispatcher dos canais 3-4 — escolhe o nível de teoria conforme
    config["site_qm_method"]:
        "rhf_sto3g_fragments" — QM real por resíduo (ver _fragment_qm_weights),
            com fallback heurístico POR FRAGMENTO quando o QM falha só num
            resíduo (o resto do sítio continua com o valor real).
        "gfn_xtb_fragments" — GFN-xTB por resíduo, em paralelo (ver
            _xtb_fragment_weights/_pick_fragment_parallelism): mesma
            unidade de fragmentação do RHF/STO-3G, e paraleliza várias
            chamadas de xtb via ThreadPoolExecutor (thread, não processo —
            cada chamada já é um subprocess do SO, não compete pelo GIL
            nem toca CUDA, então não esbarra na restrição de workers=1 que
            existe para a parte de voxelização em GPU).
        "heuristic" (ou qualquer valor não reconhecido) — eletronegatividade
            + carga (channel_site_donor_acceptor), sem QM.

    Não fragmentar (rodar GFN-xTB no sítio protonado inteiro de uma vez)
    foi tentado e removido: o xtb detecta o sítio como múltiplos
    fragmentos moleculares desconectados (é uma concatenação de resíduos
    capados independentemente, não uma molécula covalentemente contínua)
    e trava com segmentation fault em sítios reais, além de convergência
    lenta (minutos) mesmo quando não crashava.

    Em todos os casos o resultado final é sempre um peso doador/aceptor
    por átomo espalhado na grade pela mesma _voxelize_atoms — unifica os
    níveis de teoria no mesmo formato de saída.

    Retorna (donor, acceptor, method_used) — o terceiro valor é o método
    que REALMENTE gerou os canais (não o configurado): "*_fragments" só se
    TODOS os resíduos tiveram QM real, "*_fragments_partial(n/m)" se uma
    parte caiu no fallback por resíduo, e "heuristic" em qualquer fallback
    total. Existe para que o chamador grave em meta.json o que aconteceu
    de verdade, em vez de ecoar cegamente config["site_qm_method"] mesmo
    quando o QM falha e o cálculo cai para a heurística.
    """
    method = config.get("site_qm_method", "rhf_sto3g_fragments")
    batch  = config.get("vox_batch", 500 if HAS_CUPY else 100)

    if method == "rhf_sto3g_fragments" and fragments and HAS_PYSCF:
        n_atoms = len(site_elements)
        donor_w    = np.zeros(n_atoms, dtype=np.float32)
        acceptor_w = np.zeros(n_atoms, dtype=np.float32)
        n_qm_ok = 0
        for frag in fragments:
            start, end, mol = frag["start"], frag["end"], frag["mol"]
            result = _fragment_qm_weights(mol, config) if mol is not None else None
            if result is not None:
                donor_w[start:end], acceptor_w[start:end] = result
                n_qm_ok += 1
            else:
                d, a = _electronegativity_donor_acceptor_weights(
                    site_elements[start:end], site_charges[start:end]
                )
                donor_w[start:end], acceptor_w[start:end] = d, a
        log.info(f"  QM fragmentado (RHF/STO-3G): {n_qm_ok}/{len(fragments)} "
                 f"resíduos com orbital real, resto por aproximação")
        donor, acceptor, _ = _voxelize_atoms(
            grid_flat, site_coords, donor_w, acceptor_w, sigma, grid_shape,
            batch=batch, compute_mep=False,
        )
        if n_qm_ok == 0:
            method_used = "heuristic"
        elif n_qm_ok == len(fragments):
            method_used = "rhf_sto3g_fragments"
        else:
            method_used = f"rhf_sto3g_fragments_partial({n_qm_ok}/{len(fragments)})"
        return donor, acceptor, method_used

    if method == "gfn_xtb_fragments" and fragments:
        xtb_bin = _xtb_bin()
        if xtb_bin is not None:
            n_atoms = len(site_elements)
            donor_w    = np.zeros(n_atoms, dtype=np.float32)
            acceptor_w = np.zeros(n_atoms, dtype=np.float32)
            qm_fragments = [f for f in fragments if f["mol"] is not None]
            n_workers, threads_per_call = _pick_fragment_parallelism(
                len(qm_fragments) or 1, config
            )
            results: dict[int, "tuple[np.ndarray, np.ndarray] | None"] = {}
            if qm_fragments:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    future_to_idx = {
                        ex.submit(
                            _xtb_fragment_weights, f["mol"], config, xtb_bin,
                            threads_per_call,
                        ): i
                        for i, f in enumerate(qm_fragments)
                    }
                    for fut in as_completed(future_to_idx):
                        results[future_to_idx[fut]] = fut.result()
            n_qm_ok = 0
            qi = 0
            for frag in fragments:
                start, end, mol = frag["start"], frag["end"], frag["mol"]
                result = results.get(qi) if mol is not None else None
                if mol is not None:
                    qi += 1
                if result is not None:
                    donor_w[start:end], acceptor_w[start:end] = result
                    n_qm_ok += 1
                else:
                    d, a = _electronegativity_donor_acceptor_weights(
                        site_elements[start:end], site_charges[start:end]
                    )
                    donor_w[start:end], acceptor_w[start:end] = d, a
            log.info(f"  QM fragmentado (GFN-xTB, {n_workers} paralelos x "
                     f"{threads_per_call} threads): {n_qm_ok}/{len(fragments)} "
                     f"resíduos com orbital real, resto por aproximação")
            donor, acceptor, _ = _voxelize_atoms(
                grid_flat, site_coords, donor_w, acceptor_w, sigma, grid_shape,
                batch=batch, compute_mep=False,
            )
            if n_qm_ok == 0:
                method_used = "heuristic"
            elif n_qm_ok == len(fragments):
                method_used = "gfn_xtb_fragments"
            else:
                method_used = f"gfn_xtb_fragments_partial({n_qm_ok}/{len(fragments)})"
            return donor, acceptor, method_used
        # xtb ausente — cai para a heurística abaixo

    donor, acceptor = channel_site_donor_acceptor(
        grid_flat, site_coords, site_elements, site_charges, sigma, grid_shape,
        batch=batch,
    )
    return donor, acceptor, "heuristic"


def channel_homo_lumo_qm(
    grid_coords: np.ndarray,
    mol_rdkit,
    origin: np.ndarray,
    voxel_size: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """HOMO/LUMO do LIGANTE via RHF/STO-3G — NÃO usada na construção do
    tensor do sítio (ver channel_site_donor_acceptor, que a substituiu para
    eliminar o vazamento de dados treino↔inferência descrito lá).

    Mantida no código porque o cálculo QM em si permanece válido e pode ser
    reaproveitado no futuro como *alvo* de uma loss auxiliar do decoder
    (prever a densidade eletrônica do ligante gerado, em vez de recebê-la
    como entrada do encoder) — isso adicionaria sinal químico sem
    reintroduzir o vazamento, mas exige uma mudança de arquitetura
    (cabeça extra no GNNDecoder + termo na loss) que ainda não foi feita.

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

        # GPU4PySCF mantém mo_coeff/mo_occ em cupy — cubegen.orbital (CPU)
        # exige numpy. Ver _qm_to_numpy.
        mo_occ   = _qm_to_numpy(mf.mo_occ)
        mo_coeff = _qm_to_numpy(mf.mo_coeff)
        homo_i  = int(np.where(mo_occ > 0)[0][-1])
        lumo_i  = homo_i + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            fn_h = os.path.join(tmpdir, "homo.cube")
            fn_l = os.path.join(tmpdir, "lumo.cube")
            cubegen.orbital(pyscf_mol, fn_h, mo_coeff[:, homo_i],
                            nx=n, ny=n, nz=n, resolution=voxel_size)
            cubegen.orbital(pyscf_mol, fn_l, mo_coeff[:, lumo_i],
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

    # Eletronegatividade de Pauling por elemento (tabela compartilhada com
    # channel_site_donor_acceptor, ver ELECTRONEGATIVITY no topo do módulo)
    electroneg = ELECTRONEGATIVITY

    homo_grid = np.zeros((n, n, n), dtype=np.float32)
    lumo_grid = np.zeros((n, n, n), dtype=np.float32)

    for atom in mol_rdkit.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        sym   = atom.GetSymbol().upper()
        xyz   = conf.GetPositions()[atom.GetIdx()]
        q     = float(atom.GetDoubleProp("_GasteigerCharge")
                      if atom.HasProp("_GasteigerCharge") else 0.0)
        en    = electroneg.get(sym, ELECTRONEGATIVITY_DEFAULT)

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

# Cargas aproximadas por elemento (médias grosseiras em aminoácidos) — usada
# como fallback quando a protonação por RDKit (protonate_residue) falha ou
# está desativada (config["protonate_site"]=False). NÃO é Gasteiger real —
# ver load_site_atoms_protonated para a carga real por átomo.
_ELEM_CHARGE_APPROX = {
    "C": -0.05, "N": -0.30, "O": -0.35, "S": -0.15,
    "P": +0.40, "FE": +0.30, "ZN": +0.40, "MG": +0.50,
}


def load_site_atoms(pdb_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Lê o binding_site.pdb e retorna coords, elementos e cargas aproximadas
    (tabela fixa por elemento, ver _ELEM_CHARGE_APPROX — NÃO é Gasteiger
    real por resíduo). Sem hidrogênios, sem protonação.

    Mantida por compatibilidade e como fallback rápido. Para o caminho
    recomendado (protonação por pH + cargas Gasteiger reais), ver
    load_site_atoms_protonated.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("site", str(pdb_path))

    coords   = []
    elements = []

    # Restrito ao modelo 0: arquivos vindos de estruturas de RMN (vários
    # MODELOS = mesmos átomos em conformações diferentes) duplicariam o
    # sítio por N sem ganho nenhum se todos os modelos fossem lidos — ver
    # a mesma correção em pipeline_curate.py (save_pdb) e em
    # load_site_atoms_protonated logo abaixo.
    for chain in struct[0]:
        for residue in chain:
            for atom in residue:
                elem = (atom.element or atom.get_name()[0]).strip().upper()
                if elem in ("H", "D", ""):
                    continue
                coords.append(atom.get_vector().get_array())
                elements.append(elem)

    coords_arr = np.array(coords, dtype=np.float32)
    charges = np.array([_ELEM_CHARGE_APPROX.get(e, 0.0) for e in elements],
                       dtype=np.float32)

    return coords_arr, elements, charges


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3B — PROTONAÇÃO DO SÍTIO (pH-dependente)
# ══════════════════════════════════════════════════════════════════════════════
# Módulo próprio do CODEEP, inspirado na etapa de preparo de proteína do
# CODOC (adição de hidrogênios conforme pKa padrão + pH definido pelo
# usuário), porém implementado aqui de forma independente e autocontida —
# sem importar CODOC.
#
# Limitação assumida: pKa "de tabela" (sem correção por microambiente
# estrutural, ao contrário de ferramentas como PROPKA). Aproximação de
# primeira ordem, consistente com o nível de rigor do resto do módulo.
#
# A protonação é sempre RE-DERIVADA da regra de pH, ignorando H eventuais
# já presentes no PDB de origem — garante que o tensor seja determinístico
# e reprodutível (mesmo resultado com ou sem H experimentais na estrutura).

STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# Átomos de backbone (cadeia principal) — usados para decidir o que fica
# travado no modo "induced-fit" da minimização final (ver
# pipeline_generate._minimize_complex_rdkit/_minimize_complex_gromacs):
# backbone travado (representa onde o resto da proteína real prende esse
# resíduo no lugar — não temos a cadeia inteira, só o sítio recortado),
# cadeia lateral livre para se ajustar ao ligante. Nota: o caminho GROMACS
# não reaproveita esta lista diretamente (reconstrói o próprio backbone
# flag a partir da topologia que o pdb2gmx gera, ver BACKBONE_ATOM_NAMES em
# pipeline_generate.py) — só o caminho RDKit usa esta lista aqui.
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}

# pKa padrão dos grupos ionizáveis em água a 25°C (Nelson & Cox, Lehninger
# Principles of Biochemistry) — não considera microambiente/estrutura.
RESIDUE_PKA = {
    "ASP": 3.9, "GLU": 4.1, "CYS": 8.3, "TYR": 10.1,
    "HIS": 6.0, "LYS": 10.5, "ARG": 12.5,
}

# Átomo ionizável de cada resíduo + nº de H no estado protonado/desprotonado.
# "acidic": estado protonado (pH<pKa) é NEUTRO (default do RDKit, sem
#   intervenção); estado desprotonado (pH>pKa) precisa de carga -1 explícita.
# "basic": estado protonado (pH<pKa) precisa de carga +1 explícita; estado
#   desprotonado (pH>pKa) é NEUTRO (default do RDKit, sem intervenção).
# HIS é tratada à parte (3 estados: HID/HIE neutros + HIP +1).
RESIDUE_IONIZABLE = {
    "ASP": {"kind": "acidic", "atom": "OD2", "h_protonated": 1, "h_deprotonated": 0},
    "GLU": {"kind": "acidic", "atom": "OE2", "h_protonated": 1, "h_deprotonated": 0},
    "CYS": {"kind": "acidic", "atom": "SG",  "h_protonated": 1, "h_deprotonated": 0},
    "TYR": {"kind": "acidic", "atom": "OH",  "h_protonated": 1, "h_deprotonated": 0},
    "LYS": {"kind": "basic",  "atom": "NZ",  "h_protonated": 3, "h_deprotonated": 2},
    "ARG": {"kind": "basic",  "atom": "NH2", "h_protonated": 2, "h_deprotonated": 1},
}


def determine_protonation(resname: str, ph: float) -> "dict | None":
    """Decide o estado de protonação de um resíduo ionizável no pH dado
    (regra de limiar de Henderson-Hasselbalch sobre o pKa de tabela:
    protonado se pH < pKa).

    Retorna None quando o resíduo não é ionizável OU quando o estado no pH
    dado já é o default do RDKit (nenhuma intervenção de carga necessária —
    ver comentário em RESIDUE_IONIZABLE). Retorna um dict
    {"atom", "charge", "n_h"} quando é preciso sobrescrever a carga formal
    do átomo ionizável antes de Chem.AddHs. HIS é tratada à parte
    (_apply_his_protonation).
    """
    info = RESIDUE_IONIZABLE.get(resname.upper())
    if info is None:
        return None
    pka = RESIDUE_PKA[resname.upper()]
    protonated = ph < pka
    default_is_protonated = (info["kind"] == "acidic")
    if protonated == default_is_protonated:
        return None
    if info["kind"] == "acidic":   # aqui só chega o ramo desprotonado
        charge = -1
    else:                          # "basic" — aqui só chega o ramo protonado
        charge = 1
    n_h = info["h_protonated"] if protonated else info["h_deprotonated"]
    return {"atom": info["atom"], "charge": charge, "n_h": n_h}


def _apply_his_protonation(mol: "Chem.Mol", ph: float) -> None:
    """HIS tem 3 estados usuais: HID/HIE (neutros, tautômeros ND1-H/NE2-H)
    e HIP (+1, ambos protonados).

    pH < pKa(HIS)≈6.0 → HIP (ND1 e NE2 protonados, carga total +1).
    pH ≥ pKa(HIS)      → tautômero neutro; o default do RDKit ao adicionar H
      num template sem H (sanitização + aromaticidade) já resolve para
      NE2-H (HIE), o tautômero mais comum como default em campos de força
      quando a estrutura não tem H experimentais para inferir o tautômero
      real — não é preciso nenhuma intervenção nesse ramo.
    """
    if ph >= RESIDUE_PKA["HIS"]:
        return
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetName().strip() if info else ""
        if name == "NE2":
            atom.SetNumExplicitHs(1); atom.SetNoImplicit(True); atom.SetFormalCharge(1)
        elif name == "ND1":
            atom.SetNumExplicitHs(1); atom.SetNoImplicit(True); atom.SetFormalCharge(0)


def _residue_pdb_block(residue) -> str:
    """Serializa um resíduo Bio.PDB (sem H) em um bloco PDB mínimo, no
    formato de colunas fixas esperado pelo RDKit (Chem.MolFromPDBBlock)."""
    lines = []
    i = 0
    for atom in residue.get_atoms():
        elem = (atom.element or atom.get_name()[0]).strip().upper()
        if elem in ("H", "D", ""):
            continue
        i += 1
        x, y, z = atom.get_coord()
        name = atom.get_name()
        name_field = f" {name:<3s}" if len(name) < 4 else name[:4]
        lines.append(
            f"ATOM  {i:5d} {name_field:<4s} {residue.get_resname():<3s} "
            f"A{residue.id[1]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {elem:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)


def protonate_residue(residue, ph: float = 7.4) -> "Chem.Mol | None":
    """Constrói um fragmento RDKit protonado (H explícitos com coordenadas
    3D) a partir de um resíduo Bio.PDB isolado, no pH informado.

    As ligações peptídicas cortadas nas bordas do fragmento (backbone N/C
    do resíduo, que na cadeia real se ligam ao resíduo vizinho) ficam com
    valência incompleta ao isolar o resíduo — o RDKit completa
    automaticamente com H ao sanitizar/adicionar hidrogênios. Este é
    exatamente o "capping" padrão de métodos QM fragmentados (FMO, Kitaura
    et al. 1999): a ligação peptídica cortada vira -NH2/-CHO em vez de
    -NH-C(=O)-. Aproximação aceitável para um descritor de ML — não para
    energia de ligação de alta precisão (validado manualmente: ver
    discussão no código-fonte/commit desta função).

    Retorna None em qualquer falha — o chamador cai para a aproximação
    semi-empírica (eletronegatividade) nesse resíduo.
    """
    resname = residue.get_resname().upper()
    try:
        pdb_block = _residue_pdb_block(residue)
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        if resname == "HIS":
            Chem.SanitizeMol(mol)
            _apply_his_protonation(mol, ph)
        else:
            Chem.SanitizeMol(mol)
            adj = determine_protonation(resname, ph)
            if adj is not None:
                for atom in mol.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info and info.GetName().strip() == adj["atom"]:
                        atom.SetNumExplicitHs(adj["n_h"])
                        atom.SetNoImplicit(True)
                        atom.SetFormalCharge(adj["charge"])
                        break

        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.SanitizeMol(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)
        return mol
    except Exception as e:
        log.debug(f"  Protonação falhou para {resname}{residue.id[1]}: "
                  f"{type(e).__name__}: {e}")
        return None


def protonate_site_residues(
    residues: list, ph: float = 7.4,
) -> tuple[np.ndarray, list[str], np.ndarray, list[dict]]:
    """Núcleo compartilhado de protonação: recebe uma lista JÁ SELECIONADA
    de resíduos Bio.PDB (objetos `Residue`, com TODOS os átomos de cada
    resíduo — não sub-conjuntos parciais, ver nota abaixo) e retorna os
    arrays combinados protonados + fragmentos por resíduo.

    Usada tanto por load_site_atoms_protonated (treino — lê o
    binding_site.pdb inteiro) quanto por pipeline_generate.pdb_to_site_tensor
    (inferência — resíduos selecionados por extract_site_from_pdb ou
    load_site_from_residues_pdb). Ambos os lados chamam esta MESMA função
    para que a lógica de protonação seja idêntica — é a mesma garantia de
    paridade treino/inferência que motivou os canais 3-4 serem
    site-derived em vez de vir do ligante (ver BLOCO 3B).

    IMPORTANTE: cada `residue` deve conter o conjunto COMPLETO de átomos do
    resíduo (backbone + cadeia lateral), não um subconjunto escolhido por
    proximidade atômica ao ligante — do contrário, protonate_residue
    interpretaria átomos ausentes no MEIO do resíduo (não apenas nas bordas
    da cadeia) como valência cortada e adicionaria H espúrios. A seleção de
    QUAIS resíduos incluir pode ser por proximidade (cutoff), mas uma vez
    selecionado, o resíduo entra inteiro.

    Resíduos-padrão (STANDARD_AMINO_ACIDS) são reconstruídos com H
    explícitos e cargas Gasteiger REAIS (RDKit). Íons/metais/água/
    heteroátomos não passam pela protonação e usam o fallback de carga
    aproximada por elemento (_ELEM_CHARGE_APPROX). Se a protonação de um
    resíduo-padrão falhar por qualquer motivo, esse resíduo específico
    também cai no fallback — nunca derruba o par/sítio inteiro.

    Returns:
        coords, elements, charges — arrays combinados de todos os átomos
            (heavy + H, na ordem dos resíduos de entrada).
        fragments — lista de dicts {"resname", "start", "end", "mol"}, um
            por resíduo — usada pela etapa de QM fragmentado
            (_fragment_qm_weights). "mol" é None quando caiu no fallback.
    """
    coords_all: list = []
    elements_all: list = []
    charges_all: list = []
    fragments: list = []

    for residue in residues:
        resname = residue.get_resname().upper()
        is_standard = residue.id[0] == " " and resname in STANDARD_AMINO_ACIDS
        mol = protonate_residue(residue, ph) if is_standard else None

        start = len(coords_all)
        if mol is not None:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coords_all.append([pos.x, pos.y, pos.z])
                elements_all.append(atom.GetSymbol().upper())
                q = float(atom.GetDoubleProp("_GasteigerCharge")
                          if atom.HasProp("_GasteigerCharge") else 0.0)
                charges_all.append(q if np.isfinite(q) else 0.0)
        else:
            for atom in residue.get_atoms():
                elem = (atom.element or atom.get_name()[0]).strip().upper()
                if elem in ("H", "D", ""):
                    continue
                coords_all.append(atom.get_vector().get_array())
                elements_all.append(elem)
                charges_all.append(_ELEM_CHARGE_APPROX.get(elem, 0.0))
        end = len(coords_all)
        if end > start:
            fragments.append({
                "resname": resname, "start": start, "end": end, "mol": mol,
            })

    coords_arr  = np.array(coords_all, dtype=np.float32)
    charges_arr = np.nan_to_num(np.array(charges_all, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
    return coords_arr, elements_all, charges_arr, fragments


def _classify_fragment_backbone(mol: "Chem.Mol") -> list[bool]:
    """Para um fragmento residual protonado (ver protonate_residue), retorna
    um bool por átomo: True = backbone (N, Cα, C, O), False = cadeia
    lateral. Átomos sem informação de resíduo (hidrogênios adicionados por
    Chem.AddHs — nunca herdam PDBResidueInfo) herdam a classificação do
    único átomo pesado ao qual estão ligados.
    """
    flags = [False] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None:
            flags[atom.GetIdx()] = info.GetName().strip() in BACKBONE_ATOM_NAMES
    for atom in mol.GetAtoms():
        if atom.GetPDBResidueInfo() is None:   # hidrogênio sem nome de átomo
            nbrs = atom.GetNeighbors()
            if len(nbrs) == 1:
                flags[atom.GetIdx()] = flags[nbrs[0].GetIdx()]
    return flags


def build_combined_site_mol(
    coords: np.ndarray,
    elements: list[str],
    fragments: list[dict],
) -> tuple["Chem.Mol", list[bool]]:
    """Combina os fragmentos residuais protonados (ver
    protonate_site_residues) num único Mol do sítio, preservando as
    ligações REAIS intra-resíduo — necessário para o modo "induced-fit" da
    minimização final: se os átomos de cadeia lateral vão poder se mover,
    o campo de força precisa das ligações entre eles para manter a
    geometria interna do resíduo sã durante a minimização (sem isso, nada
    impede um comprimento de ligação absurdo dentro da própria cadeia
    lateral). No modo travado (sítio 100% rígido), as ligações não fazem
    diferença — mas mantê-las sempre presentes evita ter dois caminhos de
    construção do Mol do sítio para manter em sincronia.

    Fragmentos sem protonação bem-sucedida (mol=None — íons, água,
    heteroátomos, ou falha pontual de protonação) entram como átomos
    isolados, sem ligações — tratados como sempre-backbone (sempre
    travados no modo induced-fit também, já que não há como classificá-los
    nem geometria interna a preservar).

    Returns:
        combined_mol — Mol único, MESMA ORDEM de átomos que `elements`
            (e portanto de site_coords, ver pdb_to_site_tensor).
        is_backbone — lista de bool, mesma ordem/tamanho.
    """
    pieces: list["Chem.Mol"] = []
    is_backbone: list[bool] = []

    for frag in fragments:
        mol = frag["mol"]
        if mol is not None:
            pieces.append(mol)
            is_backbone.extend(_classify_fragment_backbone(mol))
        else:
            n = frag["end"] - frag["start"]
            rw = Chem.RWMol()
            conf = Chem.Conformer(n)
            for k in range(n):
                idx = frag["start"] + k
                try:
                    atom = Chem.Atom(elements[idx].capitalize())
                except Exception:
                    atom = Chem.Atom(6)
                rw.AddAtom(atom)
            rw.AddConformer(conf, assignId=True)   # posições reais setadas depois
            pieces.append(rw.GetMol())
            is_backbone.extend([True] * n)

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = Chem.CombineMols(combined, piece)
    Chem.SanitizeMol(combined, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS)

    # Re-atribui as posições explicitamente a partir de `coords` (autoridade
    # única) — evita depender de como Chem.CombineMols funde conformers
    # entre peças reais (protonate_residue) e peças falsas sem posição
    # (fragmentos fallback, criados acima com um conformer vazio).
    if combined.GetNumConformers() == 0:
        combined.AddConformer(Chem.Conformer(combined.GetNumAtoms()), assignId=True)
    conf = combined.GetConformer()
    for i in range(combined.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(x) for x in coords[i]])

    return combined, is_backbone


def load_site_atoms_protonated(
    pdb_path: Path, ph: float = 7.4,
) -> tuple[np.ndarray, list[str], np.ndarray, list[dict]]:
    """Lê TODO o binding_site.pdb e protona por pH — wrapper de
    protonate_site_residues (ver lá para os detalhes). Usada no treino,
    onde binding_site.pdb já contém exatamente os resíduos do bolso
    (selecionados por pipeline_curate.get_binding_site, que sempre inclui
    o resíduo inteiro, nunca átomos parciais).

    Restrito ao modelo 0 (struct[0]) — defesa equivalente à de
    pipeline_curate.py::save_pdb: arquivos curados antes dessa correção
    (estruturas de RMN com vários MODELOS) ainda têm N cópias
    concatenadas em disco até serem re-curados; ler todos os modelos aqui
    multiplicaria o sítio de novo mesmo já tendo corrigido a curadoria.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("site", str(pdb_path))
    residues = [r for chain in struct[0] for r in chain]
    return protonate_site_residues(residues, ph)


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
    """Vetor de features de uma ligação RDKit (dimensão = 7).

    Layout: one-hot de 5 posições — [SINGLE, DOUBLE, TRIPLE, AROMATIC,
    other] — seguido de conjugada e em-anel.

    A versão anterior passava BOND_TYPES[:-1], excluindo AROMATIC da lista
    de escolhas: ligações aromáticas caíam na posição "other" e o vetor
    tinha 6 posições, divergindo dos 7 dos dados já gravados. A exclusão só
    fazia sentido quando BOND_TYPES terminava num marcador genérico, o que
    não é mais o caso.

    pipeline_train._build_adjacency lê as BOND_TYPE_DIM (=4) primeiras
    posições, obtendo a classe entre os quatro tipos reais. Uma ligação
    exótica marcada em "other" tem essas quatro zeradas e acaba lida como
    SINGLE — aceitável, por ser rara em ligantes de estruturas depositadas.
    """
    feat = one_hot(bond.GetBondType(), BOND_TYPES)   # 5 posições (4 + other)
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


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4B — ORDEM DE CONSTRUÇÃO ANCORADA NO SÍTIO (para o decoder autorregressivo)
# ══════════════════════════════════════════════════════════════════════════════

def compute_site_anchored_atom_order(
    ch3: np.ndarray,
    ch4: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    ligand_pos: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Ordem canônica de construção do ligante para teacher forcing do
    decoder autorregressivo — ancorada no sítio, ao estilo Pocket2Mol
    (Peng et al., 2022): a molécula "cresce" a partir do ponto do bolso com
    maior sinal doador/aceptor de elétrons (canais 3-4, ver
    channel_site_orbital_field), em vez de uma ordem arbitrária (canônica
    RDKit) desconectada da geometria do encaixe.

    Algoritmo:
      1. Âncora = voxel de maior (canal 3 + canal 4) → coordenada 3D.
      2. Átomo semente = átomo real do ligante mais próximo da âncora.
      3. Busca "melhor-primeiro" pelas ligações reais: a cada passo, expande
         o átomo não-visitado mais próximo da âncora entre os vizinhos já
         alcançados (não é BFS por profundidade — é por proximidade
         espacial à âncora, gerando uma ordem que cresce de dentro do bolso
         para fora).
      4. Se o grafo for desconexo (ligantes multi-fragmento, comuns em
         alguns datasets), ao esgotar um fragmento reinicia a busca a
         partir do átomo não-visitado mais próximo da âncora entre os
         fragmentos restantes — garante que TODOS os átomos entrem na
         ordem, sem exigir conectividade global.

    Limitação assumida: ordem única e determinística por ligante (calculada
    uma vez aqui, cacheada no grafo). Pocket2Mol usa seleção de foco
    ligeiramente randomizada durante o treino para robustez extra — não
    implementado aqui por simplicidade; possível refinamento futuro.

    Returns:
        order — array (n_atoms,) com os índices originais dos átomos na
            ordem de construção (order[0] = primeiro átomo a colocar).
    """
    n_atoms = ligand_pos.shape[0]
    if n_atoms == 0:
        return np.array([], dtype=np.int64)
    if n_atoms == 1:
        return np.array([0], dtype=np.int64)

    combined = ch3 + ch4
    ijk = np.unravel_index(int(np.argmax(combined)), combined.shape)
    anchor_xyz = origin + voxel_size * np.array(ijk, dtype=np.float32)

    adjacency: dict = {a: set() for a in range(n_atoms)}
    for idx in range(edge_index.shape[1]):
        i, j = int(edge_index[0, idx]), int(edge_index[1, idx])
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)

    dists = np.linalg.norm(ligand_pos - anchor_xyz[None, :], axis=1)

    visited = np.zeros(n_atoms, dtype=bool)
    order: list = []
    while len(order) < n_atoms:
        remaining = np.where(~visited)[0]
        seed = int(remaining[np.argmin(dists[remaining])])
        frontier = [seed]
        while frontier:
            frontier.sort(key=lambda a: (dists[a], a))   # mais próximo da âncora primeiro
            atom = frontier.pop(0)
            if visited[atom]:
                continue
            visited[atom] = True
            order.append(atom)
            for nb in adjacency[atom]:
                if not visited[nb] and nb not in frontier:
                    frontier.append(nb)

    return np.array(order, dtype=np.int64)


CCD_CACHE_DIR = BASE_DIR / "ccd_templates"
CCD_URL_TMPL  = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"


def ccd_template(code: str, cache_dir: "Path | None" = None) -> Optional[object]:
    """Template do Chemical Component Dictionary do RCSB para um código de 3
    letras, com cache local.

    O CCD é a fonte canônica da estrutura química de cada ligante depositado
    no PDB: traz as ordens de ligação e a aromaticidade que o formato PDB
    não carrega. O download é feito uma vez por código e reaproveitado.

    Devolve None se o código não existir no CCD ou se não houver rede — o
    chamador decide o que fazer, e nunca há substituição silenciosa por uma
    molécula errada.
    """
    if not code:
        return None
    code = code.strip().upper()
    cache = Path(cache_dir) if cache_dir else CCD_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    sdf = cache / f"{code}_ideal.sdf"

    if not sdf.exists():
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(CCD_URL_TMPL.format(code=code),
                                        timeout=30) as r:
                sdf.write_bytes(r.read())
        except (urllib.error.URLError, OSError) as exc:
            log.warning(f"  Template CCD indisponível para {code}: {exc}")
            return None

    try:
        tmpl = next(iter(Chem.SDMolSupplier(str(sdf), sanitize=True)), None)
        return Chem.RemoveHs(tmpl) if tmpl is not None else None
    except Exception as exc:
        log.warning(f"  Falha ao ler template CCD de {code}: {exc}")
        return None


def _transfere_ordens_parcial(tmpl, lig):
    """Copia as ordens de ligação do template para um ligante INCOMPLETO.

    AssignBondOrdersFromTemplate exige que o ligante contenha o template
    inteiro como subestrutura. Isso falha sempre que a estrutura depositada
    tem átomos não resolvidos — situação comum, e responsável por 22% dos
    pares desta base. Aqui o casamento é feito na direção inversa: o
    ligante, com as ordens apagadas, é procurado DENTRO do template, e as
    ordens são copiadas par a par.

    Devolve None se não houver casamento.
    """
    q = Chem.Mol(lig)
    for b in q.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in q.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)

    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.makeDummiesQueries = True
    match = tmpl.GetSubstructMatch(Chem.AdjustQueryProperties(q, params))
    if not match:
        return None

    out = Chem.RWMol(lig)
    for b in out.GetBonds():
        tb = tmpl.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()],
                                      match[b.GetEndAtomIdx()])
        if tb is not None:
            b.SetBondType(tb.GetBondType())
            b.SetIsAromatic(tb.GetIsAromatic())
    for a in out.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)

    mol = out.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def load_ligand_mol(pdb_path: Path,
                    ccd_code: "str | None" = None,
                    cache_dir: "Path | None" = None) -> Optional[object]:
    """Lê o ligand.pdb com RDKit, recuperando as ordens de ligação.

    O formato PDB não armazena ordem de ligação nem aromaticidade. Lido
    diretamente, o RDKit infere a conectividade por distância e marca TODAS
    as ligações como simples — foi o que produziu um conjunto de treino com
    zero ligações duplas em 33.090 ligantes, tornando constante o alvo da
    cabeça de tipo de ligação e impedindo que o modelo pudesse gerar anéis
    aromáticos.

    A correção usa o template do CCD (ver ccd_template) com
    AssignBondOrdersFromTemplate, que transfere as ordens preservando as
    coordenadas cristalográficas do arquivo.

    Se o código não for informado, é inferido do nome do resíduo no próprio
    PDB. Falhando a recuperação, a molécula é devolvida como estava, com
    aviso explícito — perder o par seria pior que treinar com ele, mas o
    usuário precisa saber quantos ficaram assim.
    """
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

    # Código do componente: o informado ou o resName do primeiro átomo
    if not ccd_code:
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        ccd_code = info.GetResidueName().strip() if info else None

    tmpl = ccd_template(ccd_code, cache_dir) if ccd_code else None
    if tmpl is not None:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(tmpl, mol)
        except Exception:
            # Ligante incompleto: tenta o casamento parcial, que cobre 54 de
            # cada 55 desses casos (ver _transfere_ordens_parcial).
            try:
                parcial = _transfere_ordens_parcial(tmpl, mol)
            except Exception:
                parcial = None
            if parcial is not None:
                mol = parcial
            else:
                log.warning(f"  {pdb_path.parent.name}: template {ccd_code} não "
                            "casou nem parcialmente; ordens de ligação "
                            "permanecem como lidas do PDB (todas simples)")
    else:
        log.warning(f"  {pdb_path.parent.name}: sem template CCD para "
                    f"{ccd_code!r}; ordens de ligação permanecem como lidas "
                    "do PDB (todas simples)")

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

    # ── 1. Carrega átomos do sítio (protonados por pH, se config permitir) ────
    try:
        if config.get("protonate_site", True):
            site_coords, site_elements, site_charges, site_fragments = \
                load_site_atoms_protonated(site_pdb, ph=config.get("ph", 7.4))
        else:
            site_coords, site_elements, site_charges = load_site_atoms(site_pdb)
            site_fragments = []
    except Exception as e:
        return {"pair_id": pair_id, "status": "error_site_load", "msg": str(e)}

    if len(site_coords) == 0:
        return {"pair_id": pair_id, "status": "empty_site"}

    # ── 2. Constrói grade 3D ──────────────────────────────────────────────────
    grid_coords, origin, voxel_size = build_grid(site_coords, config)
    n = config["grid_size"]

    # ── 3. Carrega ligante ────────────────────────────────────────────────────
    # Código do componente vindo do pair_id (<pdb>_<lig>_<cadeia><resseq>) —
    # mais confiável que o resName do arquivo, que pode vir truncado.
    _partes = pair_id.split("_")
    mol = load_ligand_mol(lig_pdb,
                          ccd_code=_partes[1] if len(_partes) > 1 else None)

    # ── 4. Calcula canais 0-2 (vetorizado, GPU se CuPy disponível) ───────────
    sigma = max(config["gaussian_sigma"], voxel_size * 1.0)
    w_vdw  = np.array([VDW_RADII.get(e.upper(), VDW_DEFAULT)
                       for e in site_elements], dtype=np.float32)
    gf     = grid_coords.reshape(-1, 3)

    ch0, ch1, ch2 = _voxelize_atoms(
        gf, site_coords, w_vdw, site_charges, sigma, (n, n, n),
        batch=config.get("vox_batch", 500 if HAS_CUPY else 100),
    )

    # Canais 3-4: campo doador/aceptor de elétrons do SÍTIO (HOMO/LUMO-like)
    # — computados a partir dos átomos do próprio bolso de ligação, nunca do
    # ligante (elimina o vazamento treino↔inferência que existia quando
    # esses canais vinham do QM do ligante co-cristalizado). Independem de
    # `mol`. Nível de teoria conforme config["site_qm_method"] — ver
    # channel_site_orbital_field.
    sigma_da = max(config["gaussian_sigma"] * 1.5, voxel_size)
    ch3, ch4, orbital_method_used = channel_site_orbital_field(
        gf, site_coords, site_elements, site_charges, site_fragments,
        sigma_da, (n, n, n), config,
    )

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
                # Ordem de construção ancorada no sítio (ver
                # compute_site_anchored_atom_order) — usada pelo decoder
                # autorregressivo para teacher forcing. ch3/ch4 aqui ainda
                # são os canais RAW (antes da normalização por canal em
                # `tensor`), o que não afeta a posição do argmax.
                order = compute_site_anchored_atom_order(
                    ch3, ch4, origin, voxel_size,
                    graph.pos.numpy(), graph.edge_index.numpy(),
                )
                graph.atom_order = torch.tensor(order, dtype=torch.long)
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
    # Canais 3-4 são sempre derivados do SÍTIO, nunca do ligante — o método
    # exato (QM fragmentado, xTB ou heurística) depende de site_qm_method e
    # da disponibilidade de fallback por resíduo (ver channel_site_orbital_field).
    meta = {
        "pair_id":         pair_id,
        "grid_size":       n,
        "voxel_size_A":    round(voxel_size, 4),
        "origin":          origin.tolist(),
        "n_site_atoms":    int(len(site_coords)),
        "n_ligand_atoms":  int(mol.GetNumAtoms()) if mol else 0,
        "ligand_smiles":   Chem.MolToSmiles(mol) if mol else "",
        # Método que REALMENTE gerou os canais 3-4 (não o configurado —
        # ver channel_site_orbital_field). config_site_qm_method fica
        # como referência do que foi pedido, para detectar divergência
        # (ex.: gfn_xtb_fragments pedido mas xtb ausente/falhou -> heuristic).
        "orbital_channels_source": orbital_method_used,
        "config_site_qm_method": config.get("site_qm_method", "rhf_sto3g_fragments"),
        "protonate_site":  bool(config.get("protonate_site", True)),
        "site_ph":         float(config.get("ph", 7.4)),
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
        f"| Vox: {meta['vox_backend']} "
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

    # Grava a config efetiva ANTES de processar (não só em caso de sucesso
    # completo) — mesmo uma rodada interrompida deixa registrado com que
    # config os tensores já gerados até então foram construídos.
    save_voxel_config_snapshot(tensor_dir)

    log.info("=" * 65)
    log.info("PIPELINE ETAPA 4: Voxelização 3D + Grafos Moleculares")
    log.info("=" * 65)
    log.info(f"  Hardware : {GPU_DESC}")
    log.info(f"  Curated  : {curated_dir}")
    log.info(f"  Output   : {tensor_dir}")

    log.info(
        f"  Grid     : {VOXEL_CONFIG['grid_size']}³ voxels  |  "
        f"Canais: {VOXEL_CONFIG['n_site_channels']} "
        f"(0-2 vdW/carga/MEP do sítio, 3-4 doador/aceptor semi-empírico do sítio)"
    )
    log.info(
        f"  Grafos   : {'PyTorch Geometric (.pt)' if HAS_TORCH else 'NumPy (.npz)'}"
    )
    # PySCF/GPU4PySCF permanecem disponíveis para channel_homo_lumo_qm
    # (HOMO/LUMO do ligante), mas essa função não é mais chamada na
    # construção do tensor do sítio — ver nota em channel_homo_lumo_qm.

    if args.pair:
        pair_dirs = [curated_dir / args.pair]
        if not pair_dirs[0].exists():
            log.error(f"Par não encontrado: {pair_dirs[0]}")
            sys.exit(1)
    else:
        if not curated_dir.exists():
            raise FileNotFoundError(f"Diretório curado não encontrado: {curated_dir}")
        no_disco = sorted([d for d in curated_dir.iterdir() if d.is_dir()])

        # A fonte da verdade é o metadata da curadoria, NÃO o que sobrou no
        # disco — ver _approved_pair_ids.
        aprovados = _approved_pair_ids(curated_dir)
        if aprovados is None:
            log.warning(
                "  Metadata da curadoria não encontrado — processando TODOS os "
                "diretórios de %s. Se os filtros da Etapa 2 mudaram desde a "
                "curadoria anterior, pares já rejeitados podem entrar na base.",
                curated_dir,
            )
            pair_dirs = no_disco
        else:
            pair_dirs = [d for d in no_disco if d.name in aprovados]
            obsoletos = len(no_disco) - len(pair_dirs)
            if obsoletos:
                log.info(
                    f"  Diretórios obsoletos ignorados: {obsoletos} "
                    f"(existem em {curated_dir.name}/ mas foram rejeitados "
                    f"pela curadoria atual)"
                )
            faltando = len(aprovados) - len(pair_dirs)
            if faltando > 0:
                log.warning(
                    f"  {faltando} pares aprovados pela curadoria não têm "
                    f"diretório em {curated_dir} — rode a Etapa 2 novamente."
                )

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
        # Paraleliza PARES (ThreadPoolExecutor, não ProcessPoolExecutor —
        # ver detect_optimal_workers: CUDA não suporta múltiplos contextos
        # de processo de forma proveitosa aqui, e a etapa que domina o
        # tempo por par (GFN-xTB fragmentado, ver channel_site_orbital_field)
        # já é subprocess-based, não presa ao GIL). O ganho real não vem de
        # rodar o QM mais rápido (ele já usa quase todos os núcleos por
        # par) — vem de sobrepor os ~25% restantes do tempo (protonação +
        # voxelização GPU, que hoje rodam 100% em série) com o QM de OUTROS
        # pares simultâneos. Ganho esperado: modesto (validado ~15-25% em
        # análise, não testado em lote grande) — por isso fica atrás de
        # config["parallel_pairs"], desligado por padrão.
        #
        # Divide o orçamento de threads do xtb fragmentado (ver
        # _pick_fragment_parallelism) por effective_workers, para pares
        # simultâneos não disputarem os mesmos núcleos SEM controle
        # (oversubscription) — cada par simultâneo fica com uma fração
        # dos núcleos totais em vez de tentar usar todos sozinho.
        cpu_total = os.cpu_count() or 1
        VOXEL_CONFIG["xtb_cpu_total"] = max(1, cpu_total // effective_workers)
        log.info(
            f"  ℹ Paralelizando {effective_workers} pares simultâneos "
            f"(cada um com até {VOXEL_CONFIG['xtb_cpu_total']} threads para o "
            f"GFN-xTB fragmentado, de {cpu_total} núcleos totais)."
        )
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            futures = {
                ex.submit(process_pair, d, tensor_dir, VOXEL_CONFIG): d
                for d in pair_dirs
            }
            with tqdm(total=len(pair_dirs), desc="Voxelizando") as pbar:
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
                    if r["status"] == "ok":
                        ok += 1
                    elif "skip" in r["status"]:
                        skipped += 1
                    else:
                        fail += 1
                        log.warning(f"  ✗ {r['pair_id']}: {r.get('msg', r['status'])}")
                    pbar.update(1)
    else:
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

def _approved_pair_ids(curated_dir: Path) -> "set[str] | None":
    """pair_ids que a curadoria APROVOU, lidos do metadata dela.

    A voxelização enumerava `curated_dir.iterdir()`, ou seja, o que está no
    disco. Isso diverge do que a curadoria decidiu sempre que os filtros
    mudam: a curadoria deixa de LISTAR um par rejeitado, mas não apaga o
    diretório da rodada anterior. Na prática, ao endurecer a exclusão de
    ligantes (resíduos modificados, açúcares, detergentes, tampões), a
    curadoria passou de 33.212 para 22.293 pares e a voxelização
    reprocessou os 11.471 diretórios obsoletos assim mesmo — 34% da base
    de treino era exatamente o material que se queria remover.

    Caminhos do metadata, conforme a fase (ver CODEEP.py, run_fn da
    curadoria):
        fase 1 : <...>/data/curated_metadata.csv      (irmão de curated/)
        fase 2 : <...>/curated/phase2_metadata.csv    (irmão de phase2/)
        fase 3 : <...>/curated/phase3_metadata.csv

    Retorna None quando nenhum metadata é encontrado — aí o chamador cai
    no comportamento antigo, com aviso.
    """
    candidatos = [
        curated_dir.parent / f"{curated_dir.name}_metadata.csv",
        curated_dir.parent / "curated_metadata.csv",
    ]
    for csv_path in candidatos:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            log.warning(f"  Metadata da curadoria ilegível ({csv_path}): {e}")
            continue
        if "pair_id" not in df.columns:
            log.warning(f"  {csv_path} sem coluna pair_id — ignorando.")
            continue
        aprovados = set(df["pair_id"].astype(str))
        log.info(f"  Metadata da curadoria: {csv_path.name} "
                 f"({len(aprovados)} pares aprovados)")
        return aprovados
    return None



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
