"""
Pipeline - Etapa 7: Geração de Ligantes para Proteína Externa
=============================================================
Recebe o PDB de uma proteína nova (não vista no treino), extrai o sítio
de ligação, voxeliza e usa o modelo CVAE treinado para gerar um conjunto
de ligantes candidatos ordenados por drug-likeness.

Uso básico:
    python 07_generate.py --pdb minha_proteina.pdb --ckpt checkpoints_phase3_target_specific/

Opções completas:
    python 07_generate.py
        --pdb         proteina.pdb          # arquivo PDB de entrada (obrigatório)
        --ckpt        checkpoints/           # diretório do checkpoint treinado
        --n-samples   200                   # número de ligantes a gerar
        --temperature 1.0                   # diversidade (>1 = mais diverso)
        --cutoff      6.0                   # raio do sítio em Å
        --min-atoms   5                     # mínimo de átomos pesados no sítio
        --top-n       20                    # top N candidatos no relatório
        --out         resultados/           # diretório de saída
        --ligand-ref  ligante_ref.sdf       # ligante de referência (opcional)
          # se fornecido, calcula similaridade de Tanimoto com os candidatos

Saídas em --out/:
    candidates.csv          tabela completa com todas as métricas
    top{N}_candidates.sdf   estruturas 3D dos top N candidatos
    top{N}_candidates.png   grade de estruturas 2D
    report.html             relatório interativo completo
    site_tensor.npz         tensor do sítio gerado

Requisitos:
    pip install torch torch-geometric rdkit biopython numpy pandas tqdm
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Imports condicionais ───────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    raise ImportError("PyTorch obrigatório: pip install torch")

try:
    from rdkit import Chem
    from rdkit.Chem import (AllChem, Descriptors, Draw, QED,
                             rdMolDescriptors, rdPartialCharges)
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdchem import BondType
    from rdkit.DataStructs import TanimotoSimilarity
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

PROJECT_DIR = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — EXTRAÇÃO DO SÍTIO DE LIGAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _infer_element(atom) -> str:
    """Infere elemento do átomo mesmo quando o campo element está vazio."""
    import re
    elem = (atom.element or "").strip().upper()
    if elem and elem not in ("", "X"):
        return elem
    name = atom.get_name().strip().upper()
    if not name:
        return "X"
    two = re.sub(r"[^A-Z]", "", name)[:2]
    one = two[:1]
    two_char = {"FE","CL","BR","MG","ZN","MN","CU","CO","NI","CD","HG",
                "PB","SE","SI","NA","AL","CR","MO","RU","RH","PD","AG",
                "PT","AU","SN","AS","TE","AT"}
    if two == "CA" and name.startswith("CA"):
        return "C"
    if two in two_char:
        return two
    if one in ("C","N","O","S","P","F","H","I","K","V","W","U","B"):
        return one
    alpha = re.sub(r"[^A-Z]", "", name)
    return alpha[0] if alpha else "X"


def extract_site_from_pdb(
    pdb_path: Path,
    cutoff_A: float = 6.0,
    min_site_residues: int = 5,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extrai o sítio de ligação de um PDB com ligante co-cristalizado.

    Estratégia:
      1. Procura ligantes HETATM orgânicos (com carbono, ≥7 átomos pesados)
      2. Para cada ligante, identifica resíduos proteicos a ≤ cutoff_A Å
      3. Retorna coordenadas, elementos e cargas do sítio

    Returns:
        coords   : (N, 3) float32 — coordenadas dos átomos do sítio
        elements : [N] str        — símbolos dos elementos
        charges  : (N,) float32   — cargas parciais aproximadas por elemento
    """
    from scipy.spatial import cKDTree

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("query", str(pdb_path))

    EXCLUDED = {
        "HOH","DOD","GOL","EDO","PEG","DMS","MPD","BME","DTT","EPE",
        "MES","TRS","FMT","ACT","ACE","EOH","IPA",
        "NA","K","CA","MG","ZN","FE","MN","CU","CO","NI","CD","HG",
        "PB","CL","BR","IOD","F","PO4","SO4","NO3","NO2",
    }
    ELEM_CHARGE = {
        "C": -0.05, "N": -0.30, "O": -0.35, "S": -0.15,
        "P": +0.40, "FE": +0.30, "ZN": +0.40, "MG": +0.50,
    }

    # Coleta todos os átomos proteicos
    prot_coords, prot_elems = [], []
    for model in struct:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                for atom in residue.get_atoms():
                    e = _infer_element(atom)
                    if e not in ("H", "D", "X"):
                        prot_coords.append(atom.get_vector().get_array())
                        prot_elems.append(e)

    if not prot_coords:
        raise ValueError(f"Nenhum átomo proteico encontrado em {pdb_path}")

    prot_arr  = np.array(prot_coords, dtype=np.float32)
    prot_tree = cKDTree(prot_arr)

    # Procura ligantes orgânicos
    best_ligand_coords = None
    best_n_heavy = 0

    for model in struct:
        for chain in model:
            for residue in chain:
                hetfield = residue.get_id()[0]
                resname  = residue.get_resname().strip()
                if not hetfield.startswith("H_") or resname in EXCLUDED:
                    continue
                atoms = list(residue.get_atoms())
                elems = [_infer_element(a) for a in atoms]
                heavy = [(a, e) for a, e in zip(atoms, elems)
                         if e not in ("H","D","X","")]
                n_heavy    = len(heavy)
                has_carbon = any(e == "C" for _, e in heavy)
                if n_heavy >= 7 and has_carbon and n_heavy > best_n_heavy:
                    best_ligand_coords = np.array(
                        [a.get_vector().get_array() for a, _ in heavy],
                        dtype=np.float32)
                    best_n_heavy = n_heavy

    if best_ligand_coords is None:
        log.warning("Nenhum ligante orgânico co-cristalizado encontrado. "
                    "Usando o centro geométrico da proteína como sítio.")
        center = prot_arr.mean(axis=0)
        # Pega os 80 átomos mais próximos do centro como sítio aproximado
        dists  = np.linalg.norm(prot_arr - center, axis=1)
        idx    = np.argsort(dists)[:80]
        site_coords = prot_arr[idx]
        site_elems  = [prot_elems[i] for i in idx]
    else:
        # Identifica resíduos do sítio via KDTree
        indices = prot_tree.query_ball_point(best_ligand_coords, r=cutoff_A)
        site_idx = sorted(set(i for sublist in indices for i in sublist))
        if len(site_idx) < min_site_residues:
            raise ValueError(
                f"Sítio muito pequeno ({len(site_idx)} átomos). "
                f"Tente aumentar --cutoff (atual: {cutoff_A} Å)."
            )
        site_coords = prot_arr[site_idx]
        site_elems  = [prot_elems[i] for i in site_idx]

    charges = np.array(
        [ELEM_CHARGE.get(e, 0.0) for e in site_elems],
        dtype=np.float32
    )

    log.info(f"  Sítio extraído: {len(site_coords)} átomos pesados")
    return site_coords, site_elems, charges


def load_site_from_residues_pdb(
    site_pdb: Path,
) -> tuple:
    """Carrega sítio diretamente de um PDB com apenas os resíduos de interesse.

    Usado com --site-pdb quando o usuário já conhece o sítio de ligação e
    fornece um arquivo PDB contendo apenas os resíduos do bolso — sem
    necessidade de detecção automática por ligante co-cristalizado.
    Água (HOH, DOD, WAT) é ignorada automaticamente.
    """
    ELEM_CHARGE   = {"C":-0.05,"N":-0.30,"O":-0.35,"S":-0.15,
                     "P":+0.40,"FE":+0.30,"ZN":+0.40,"MG":+0.50}
    SKIP_RESNAMES = {"HOH","DOD","WAT"}

    parser_bio = PDBParser(QUIET=True)
    struct = parser_bio.get_structure("site", str(site_pdb))

    coords, elements = [], []
    for model in struct:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in SKIP_RESNAMES:
                    continue
                for atom in residue.get_atoms():
                    e = _infer_element(atom)
                    if e not in ("H","D","X",""):
                        coords.append(atom.get_vector().get_array())
                        elements.append(e)

    if not coords:
        raise ValueError(f"Nenhum átomo encontrado em {site_pdb}")

    coords_arr = np.array(coords, dtype=np.float32)
    charges    = np.array([ELEM_CHARGE.get(e, 0.0) for e in elements],
                          dtype=np.float32)
    log.info(f"  Sítio carregado de {site_pdb.name}: "
             f"{len(coords_arr)} átomos pesados")
    return coords_arr, elements, charges

def load_reference_ligand(ref_path: Path, out_dir: Path) -> tuple["Chem.Mol | None", "Path | None"]:
    """Carrega ligante de referência em SDF/PDB/MOL2 e normaliza para SDF.

    Quando a entrada não é SDF, grava uma cópia convertida em `out_dir`
    para que o restante do pipeline opere com um formato único.
    """
    ref_path = Path(ref_path)
    suffix = ref_path.suffix.lower()
    mol = None

    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(ref_path), removeHs=True)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            return None, None
        return mol, ref_path

    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ref_path), removeHs=True, sanitize=True)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(ref_path), removeHs=True, sanitize=True)
        if mol is None:
            mol = Chem.MolFromPDBFile(str(ref_path), removeHs=True, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
    else:
        raise ValueError(
            f"Formato de ligante de referência não suportado: {ref_path.suffix}. "
            "Use .sdf, .pdb ou .mol2."
        )

    if mol is None:
        return None, None

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)

    sdf_path = out_dir / f"{ref_path.stem}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    try:
        writer.write(mol)
    finally:
        writer.close()

    log.info(f"  Ligante de referência convertido para SDF: {sdf_path.name}")
    return mol, sdf_path


def pdb_to_site_tensor(
    pdb_path: Path,
    cutoff_A: float = 6.0,
    min_site_residues: int = 5,
    site_pdb: "Path | None" = None,
    project_dir: "Path | None" = None,
) -> np.ndarray:
    """Pipeline completo: PDB → tensor do sítio (5, 32, 32, 32).

    Se site_pdb for fornecido, carrega o sítio diretamente desse arquivo
    (sem detecção automática). Caso contrário, extrai automaticamente a
    partir do ligante co-cristalizado em pdb_path.
    """
    # Carrega módulo de voxelização
    _proj = Path(project_dir) if project_dir else PROJECT_DIR
    spec = importlib.util.spec_from_file_location(
        "pipeline_voxelize", _proj / "MODULES" / "pipeline_voxelize.py")
    vox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vox)

    config = vox.VOXEL_CONFIG.copy()
    n      = config["grid_size"]  # 32

    # Extrai sítio — modo manual (--site-pdb) ou automático
    if site_pdb is not None:
        site_coords, site_elements, site_charges = \
            load_site_from_residues_pdb(site_pdb)
    else:
        site_coords, site_elements, site_charges = extract_site_from_pdb(
            pdb_path, cutoff_A, min_site_residues
        )

    # Grade 3D
    grid_coords, origin, voxel_size = vox.build_grid(site_coords, config)

    # Canais 0-2: voxelização vetorizada (NumPy ou CuPy)
    w_vdw = np.array(
        [vox.VDW_RADII.get(e.upper(), vox.VDW_DEFAULT) for e in site_elements],
        dtype=np.float32
    )
    gf = grid_coords.reshape(-1, 3)
    ch0, ch1, ch2 = vox._voxelize_atoms(
        gf, site_coords, w_vdw, site_charges,
        config["gaussian_sigma"], (n, n, n)
    )

    # Canais 3-4: HOMO/LUMO → zeros (sem ligante de referência)
    # O modelo lerá o padrão do espaço latente para completar
    ch3 = np.zeros((n, n, n), dtype=np.float32)
    ch4 = np.zeros((n, n, n), dtype=np.float32)

    # Monta e normaliza tensor
    tensor = np.stack([ch0, ch1, ch2, ch3, ch4], axis=-1)  # (n,n,n,5)
    for c in range(tensor.shape[-1]):
        sl  = tensor[..., c]
        med = np.median(sl)
        mad = np.median(np.abs(sl - med)) + 1e-8
        tensor[..., c] = np.clip(
            (sl - med) / (1.4826 * mad), -5, 5
        ).astype(np.float32)

    log.info(f"  Tensor gerado: {tensor.shape}  voxel={voxel_size:.2f} Å")
    return tensor, origin, voxel_size


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — CARREGAMENTO DO MODELO
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    ckpt_dir: Path,
    project_dir: "Path | None" = None,
):
    """Carrega o modelo CVAE do melhor checkpoint."""
    best_ckpt = ckpt_dir / "best_model.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {best_ckpt}\n"
            "Train the model before generating candidates."
        )

    _proj = Path(project_dir) if project_dir else PROJECT_DIR
    spec = importlib.util.spec_from_file_location(
        "pipeline_train", _proj / "MODULES" / "pipeline_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    cfg  = ckpt.get("config", {})

    model = mod.LigandCVAE(
        in_channels  = 5,
        latent_dim   = cfg.get("latent_dim",   128),
        hidden_dim   = cfg.get("hidden_dim",   128),
        n_gnn_layers = cfg.get("n_gnn_layers",   3),
    ).to(DEVICE)

    encoder_fc_weight = ckpt["model_state"].get("encoder.fc.0.weight")
    if encoder_fc_weight is not None:
        ckpt_in_features = int(encoder_fc_weight.shape[1])
        current_fc = model.encoder.fc[0]
        if current_fc.in_features != ckpt_in_features:
            model.encoder.fc[0] = torch.nn.Linear(
                ckpt_in_features,
                current_fc.out_features,
                bias=current_fc.bias is not None,
            ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    log.info(f"  Modelo: epoch {ckpt.get('epoch','?')} | "
             f"val_loss={ckpt.get('val_loss',0):.4f} | "
             f"latent={cfg.get('latent_dim',128)}-D | "
             f"device={DEVICE}")

    return model, mod


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — GERAÇÃO E DECODIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

ATOM_TYPES_GEN = ["C","N","O","S","P","F","Cl","Br","I","Fe","Zn","Mg","C","C"]

MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "F": 1,
    "CL": 1, "BR": 1, "I": 1, "FE": 6, "ZN": 4, "MG": 2,
}

BOND_TYPE_MAP = {
    0: BondType.SINGLE,
    1: BondType.DOUBLE,
    2: BondType.TRIPLE,
    3: BondType.AROMATIC,
}

BOND_ORDER_BUDGET = {
    BondType.SINGLE: 1.0,
    BondType.DOUBLE: 2.0,
    BondType.TRIPLE: 3.0,
    BondType.AROMATIC: 1.5,
}


def _bond_distance_ok(sym_i: str, sym_j: str, bond_type: BondType, dist: float) -> bool:
    if dist <= 0.9:
        return False
    limit = {
        BondType.SINGLE: 2.15,
        BondType.DOUBLE: 1.90,
        BondType.TRIPLE: 1.45,
        BondType.AROMATIC: 1.65,
    }.get(bond_type, 2.15)
    if {sym_i.upper(), sym_j.upper()} & {"S", "P", "FE", "ZN", "MG", "CL", "BR", "I"}:
        limit += 0.25
    return dist <= limit


def _sanitize_generated_mol(mol: "Chem.Mol") -> "Chem.Mol | None":
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass

    aromatic_bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetBondType() == BondType.AROMATIC
    ]
    if aromatic_bonds:
        downgraded = Chem.RWMol(mol)
        for i, j in aromatic_bonds:
            bond = downgraded.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bond.SetBondType(BondType.SINGLE)
            bond.SetIsAromatic(False)
        downgraded_mol = downgraded.GetMol()
        try:
            Chem.SanitizeMol(downgraded_mol)
            return downgraded_mol
        except Exception:
            pass

    return None


def _bond_components(n_atoms: int, added_bonds: set[tuple[int, int]]) -> list[list[int]]:
    neighbors = [[] for _ in range(n_atoms)]
    for i, j in added_bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    seen = [False] * n_atoms
    components = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in neighbors[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(comp)
    return components


def _connect_components(
    rwmol: "Chem.RWMol",
    atom_types: np.ndarray,
    positions: np.ndarray,
    valence_used: list[float],
    added_bonds: set[tuple[int, int]],
) -> bool:
    """Tenta conectar componentes desconexos usando a ligação simples válida mais próxima."""
    n_atoms = len(atom_types)
    while True:
        components = _bond_components(n_atoms, added_bonds)
        if len(components) <= 1:
            return True

        best = None
        for left_idx in range(len(components) - 1):
            left = components[left_idx]
            for right in components[left_idx + 1:]:
                for i in left:
                    sym_i = ATOM_TYPES_GEN[min(int(atom_types[i]), len(ATOM_TYPES_GEN) - 1)].upper()
                    if valence_used[i] + 1.0 > MAX_VALENCE.get(sym_i, 4) + 1e-6:
                        continue
                    for j in right:
                        pair = tuple(sorted((i, j)))
                        if pair in added_bonds:
                            continue
                        sym_j = ATOM_TYPES_GEN[min(int(atom_types[j]), len(ATOM_TYPES_GEN) - 1)].upper()
                        if valence_used[j] + 1.0 > MAX_VALENCE.get(sym_j, 4) + 1e-6:
                            continue
                        dist = float(np.linalg.norm(positions[i] - positions[j]))
                        if not _bond_distance_ok(sym_i, sym_j, BondType.SINGLE, dist):
                            continue
                        if best is None or dist < best[0]:
                            best = (dist, i, j)

        if best is None:
            return False

        _, i, j = best
        rwmol.AddBond(i, j, BondType.SINGLE)
        valence_used[i] += 1.0
        valence_used[j] += 1.0
        added_bonds.add(tuple(sorted((i, j))))


def _relax_generated_geometry(mol: "Chem.Mol") -> "Chem.Mol":
    if mol is None or mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return mol
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol

def preds_to_mol(
    preds: dict,
    template_graph=None,
    bond_threshold: float = 0.35,
    cutoff_dist: float = 1.9,
) -> "Chem.Mol | None":
    """Converte predições do decoder em molécula RDKit.

    Estratégia:
      1. Tipo de átomo → argmax dos logits
      2. Posição 3D    → predição direta
      3. Ligações      → usa o edge_index do template graph, bond_exist e
         bond_logits para reconstruir a conectividade; se isso falhar, cai
         para a heurística de distância como último recurso.

    Isso preserva melhor a topologia 3D prevista pelo decoder e reduz a
    geração de estruturas quimicamente irreais.
    """
    try:
        atom_types = preds["atom_logits"].argmax(dim=-1).cpu().numpy()
        positions  = preds["positions"].detach().cpu().numpy()
        n_atoms    = len(atom_types)

        rwmol = Chem.RWMol()
        conf  = Chem.Conformer(n_atoms)

        for i, at_idx in enumerate(atom_types):
            sym = ATOM_TYPES_GEN[min(int(at_idx), len(ATOM_TYPES_GEN)-1)]
            rwmol.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(i, positions[i].tolist())

        rwmol.AddConformer(conf, assignId=True)

        valence_used = [0.0] * n_atoms
        added = set()
        edge_candidates = []

        if template_graph is not None and hasattr(template_graph, "edge_index"):
            edge_index = template_graph.edge_index.detach().cpu().numpy()
            bond_probs = torch.sigmoid(preds["bond_exist"]).detach().cpu().numpy()
            bond_types = preds["bond_logits"].argmax(dim=-1).detach().cpu().numpy()
            merged_edges = {}

            for idx in range(edge_index.shape[1]):
                i = int(edge_index[0, idx])
                j = int(edge_index[1, idx])
                if i == j or i >= n_atoms or j >= n_atoms:
                    continue
                a, b = sorted((i, j))
                item = merged_edges.setdefault((a, b), {"prob": [], "type": []})
                item["prob"].append(float(bond_probs[idx]))
                item["type"].append(int(bond_types[idx]))

            for (i, j), item in merged_edges.items():
                prob = max(item["prob"]) if item["prob"] else 0.0
                type_idx = max(set(item["type"]), key=item["type"].count) if item["type"] else 0
                bond_type = BOND_TYPE_MAP.get(type_idx, BondType.SINGLE)
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                edge_candidates.append((prob, dist, i, j, bond_type))

            edge_candidates.sort(key=lambda entry: (-entry[0], entry[1]))

        for prob, dist, i, j, bond_type in edge_candidates:
            if prob < bond_threshold or (i, j) in added:
                continue
            sym_i = ATOM_TYPES_GEN[min(int(atom_types[i]), len(ATOM_TYPES_GEN) - 1)].upper()
            sym_j = ATOM_TYPES_GEN[min(int(atom_types[j]), len(ATOM_TYPES_GEN) - 1)].upper()
            budget = BOND_ORDER_BUDGET.get(bond_type, 1.0)
            if not _bond_distance_ok(sym_i, sym_j, bond_type, dist):
                continue
            if valence_used[i] + budget > MAX_VALENCE.get(sym_i, 4) + 1e-6:
                continue
            if valence_used[j] + budget > MAX_VALENCE.get(sym_j, 4) + 1e-6:
                continue
            rwmol.AddBond(i, j, bond_type)
            valence_used[i] += budget
            valence_used[j] += budget
            added.add((i, j))

        if not added:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    d = float(np.linalg.norm(positions[i] - positions[j]))
                    if 1.0 < d < cutoff_dist and (i, j) not in added:
                        sym_i = ATOM_TYPES_GEN[min(int(atom_types[i]), len(ATOM_TYPES_GEN) - 1)].upper()
                        sym_j = ATOM_TYPES_GEN[min(int(atom_types[j]), len(ATOM_TYPES_GEN) - 1)].upper()
                        if valence_used[i] >= MAX_VALENCE.get(sym_i, 4):
                            continue
                        if valence_used[j] >= MAX_VALENCE.get(sym_j, 4):
                            continue
                        rwmol.AddBond(i, j, BondType.SINGLE)
                        valence_used[i] += 1.0
                        valence_used[j] += 1.0
                        added.add((i, j))

        if added and not _connect_components(rwmol, atom_types, positions, valence_used, added):
            return None

        mol = _sanitize_generated_mol(rwmol.GetMol())
        if mol is None:
            return None
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        return _relax_generated_geometry(mol)

    except Exception:
        return None

def get_template_graph(mod, tensor_dir: Path) -> "PyGData | None":
    """Obtém um grafo template do conjunto de validação para guiar o decoder."""
    try:
        ds = mod.SiteLigandDataset(tensor_dir=tensor_dir, split="val", max_atoms=50)
        if len(ds) == 0:
            ds = mod.SiteLigandDataset(tensor_dir=tensor_dir, split="train", max_atoms=50)
        item = ds[0]
        if item is None:
            return None
        from torch_geometric.data import Batch
        return Batch.from_data_list([item["graph"]]).to(DEVICE)
    except Exception as e:
        log.warning(f"  Template graph: {e}")
        return None


def get_reference_template_graph(ref_mol) -> "PyGData | None":
    """Converte o ligante de referência em grafo template compatível com o treino."""
    if ref_mol is None:
        return None
    try:
        import MODULES.pipeline_voxelize as vox
        from torch_geometric.data import Batch

        mol = Chem.Mol(ref_mol)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        graph = vox.mol_to_graph(mol, pair_id="reference_ligand")
        if graph is None:
            return None
        return Batch.from_data_list([graph]).to(DEVICE)
    except Exception as e:
        log.warning(f"  Reference ligand template: {e}")
        return None


def generate_candidates(
    model,
    mod,
    site_tensor: np.ndarray,
    tensor_dir: Path,
    n_samples: int = 200,
    temperature: float = 1.0,
    voxel_size: float = 1.0,
    ref_mol = None,
) -> list[dict]:
    """Gera N candidatos a partir do tensor do sítio.

    Para cada amostra:
      1. Codifica o sítio → z_mean via encoder
      2. Amostra z = z_mean + temperatura × N(0,I)
      3. Decodifica z + template → predições de átomos/ligações
      4. Converte predições em molécula RDKit
      5. Calcula métricas de drug-likeness

    Returns lista de dicts com mol, smiles e métricas.
    """
    # Prepara tensor
    arr    = np.transpose(site_tensor, (3, 0, 1, 2))         # (C,X,Y,Z)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Template para o decoder: prioriza o ligante de referência do sítio.
    template = get_reference_template_graph(ref_mol)
    if template is None:
        template = get_template_graph(mod, tensor_dir)
    if template is None:
        log.warning("  Sem template graph — geração pode ser limitada")

    # Obtém z_mean do encoder (ponto central do espaço latente para este sítio)
    with torch.no_grad():
        voxel_sizes_gen = torch.tensor([voxel_size],
                                        dtype=torch.float32).to(DEVICE)
        z_mean, _ = model.encoder(tensor, voxel_sizes_gen)

    candidates = []
    failed     = 0

    for i in tqdm(range(n_samples), desc="  Gerando candidatos", leave=False):
        try:
            with torch.no_grad():
                # Amostragem com temperatura
                z = z_mean + temperature * torch.randn_like(z_mean)

                if template is not None:
                    preds = model.decoder(z, template)
                else:
                    continue

            mol = preds_to_mol(preds, template_graph=template)
            if mol is None or mol.GetNumAtoms() < 3:
                failed += 1
                continue

            smiles = Chem.MolToSmiles(mol)
            if not smiles or smiles == "":
                failed += 1
                continue

            metrics = compute_drug_metrics(mol, smiles)
            if metrics is None:
                failed += 1
                continue

            candidates.append({
                "idx":    i,
                "mol":    mol,
                "smiles": smiles,
                **metrics,
            })

        except Exception:
            failed += 1
            continue

    log.info(f"  Gerados: {len(candidates)} válidos | {failed} inválidos "
             f"de {n_samples} amostras "
             f"({100*len(candidates)/max(n_samples,1):.1f}% válidos)")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — MÉTRICAS DE DRUG-LIKENESS
# ══════════════════════════════════════════════════════════════════════════════

def compute_drug_metrics(mol, smiles: str) -> dict | None:
    """Calcula métricas de qualidade para um candidato.

    Métricas calculadas:
      QED         — Drug-likeness geral (0-1, Bickerton et al. 2012)
      MW          — Peso molecular (Da)
      LogP        — Coeficiente de partição octanol/água (lipofilia)
      HBD / HBA   — Doadores e aceptores de H
      TPSA        — Área de superfície polar topológica (Å²)
      RotBonds    — Ligações rotacionáveis
      Lipinski    — Aprovado na regra dos 5 (bool)
      Veber       — Aprovado nos critérios de Veber (oral bioavailability)
      SA_score    — Sintetizabilidade (1=fácil, 10=difícil; Ertl & Schuffenhauer)
      n_rings     — Número de anéis
      n_arom_rings— Número de anéis aromáticos
    """
    try:
        mw      = Descriptors.MolWt(mol)
        logp    = Descriptors.MolLogP(mol)
        hbd     = rdMolDescriptors.CalcNumHBD(mol)
        hba     = rdMolDescriptors.CalcNumHBA(mol)
        tpsa    = rdMolDescriptors.CalcTPSA(mol)
        rot     = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
        qed_val = QED.qed(mol)

        lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        veber    = (rot <= 10 and tpsa <= 140)

        # SA Score via função de Ertl (implementação via rdkit contrib)
        try:
            from rdkit.Chem import RDConfig
            sa_path = os.path.join(RDConfig.RDContribDir,
                                   "SA_Score", "sascorer.py")
            if os.path.exists(sa_path):
                spec = importlib.util.spec_from_file_location("sascorer", sa_path)
                sa_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sa_mod)
                sa_score = round(sa_mod.calculateScore(mol), 3)
            else:
                sa_score = None
        except Exception:
            sa_score = None

        return {
            "qed":        round(qed_val, 4),
            "mw":         round(mw, 2),
            "logp":       round(logp, 3),
            "hbd":        hbd,
            "hba":        hba,
            "tpsa":       round(tpsa, 2),
            "rot_bonds":  rot,
            "n_rings":    n_rings,
            "n_arom":     n_arom,
            "lipinski":   lipinski,
            "veber":      veber,
            "sa_score":   sa_score,
            "n_atoms":    mol.GetNumAtoms(),
        }
    except Exception:
        return None


def compute_tanimoto(mol, ref_mol) -> float | None:
    """Calcula similaridade de Tanimoto com Morgan fingerprints (r=2)."""
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = generator.GetFingerprint(mol)
        fp2 = generator.GetFingerprint(ref_mol)
        return round(TanimotoSimilarity(fp1, fp2), 4)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — FILTRAGEM E RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_candidates(
    candidates: list[dict],
    ref_mol=None,
) -> pd.DataFrame:
    """Filtra duplicatas, calcula Tanimoto com referência (se disponível)
    e ordena por score composto.

    Score composto = 0.4×QED + 0.2×(1-SA/10) + 0.2×Lipinski + 0.1×Veber
                   + 0.1×Tanimoto(ref)

    Penaliza: MW>600, LogP>5, TPSA>140, RotBonds>10
    """
    if not candidates:
        return pd.DataFrame()

    # Remove duplicatas por SMILES canônico
    seen   = {}
    unique = []
    for c in candidates:
        smi = c["smiles"]
        if smi not in seen:
            seen[smi] = True
            unique.append(c)

    log.info(f"  Candidatos únicos: {len(unique)} "
             f"(de {len(candidates)} gerados)")

    rows = []
    for c in unique:
        mol = c["mol"]

        # Tanimoto com referência
        tanimoto = compute_tanimoto(mol, ref_mol) if ref_mol else None

        # Penalidades
        mw_ok   = c["mw"]       <= 600
        logp_ok = c["logp"]     <= 5.0
        tpsa_ok = c["tpsa"]     <= 140
        rot_ok  = c["rot_bonds"] <= 10

        # Score composto (0-1)
        sa_norm = (10 - c["sa_score"]) / 9 if c.get("sa_score") else 0.5
        score   = (0.4 * c["qed"]
                 + 0.2 * sa_norm
                 + 0.15 * float(c["lipinski"])
                 + 0.10 * float(c["veber"])
                 + 0.05 * float(mw_ok and logp_ok and tpsa_ok and rot_ok)
                 + (0.10 * tanimoto if tanimoto else 0.0))

        rows.append({
            "smiles":    c["smiles"],
            "score":     round(score, 4),
            "qed":       c["qed"],
            "mw":        c["mw"],
            "logp":      c["logp"],
            "hbd":       c["hbd"],
            "hba":       c["hba"],
            "tpsa":      c["tpsa"],
            "rot_bonds": c["rot_bonds"],
            "n_rings":   c["n_rings"],
            "n_arom":    c["n_arom"],
            "lipinski":  c["lipinski"],
            "veber":     c["veber"],
            "sa_score":  c.get("sa_score"),
            "n_atoms":   c["n_atoms"],
            "tanimoto_ref": tanimoto,
            "_mol":      mol,  # removido antes de salvar CSV
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1  # ranking começa em 1
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — RELATÓRIO E SAÍDAS
# ══════════════════════════════════════════════════════════════════════════════

def save_sdf(df: pd.DataFrame, out_path: Path, top_n: int):
    """Salva os top N candidatos em formato SDF com propriedades."""
    writer = Chem.SDWriter(str(out_path))
    for rank, row in df.head(top_n).iterrows():
        mol = row["_mol"]
        if mol is None:
            continue
        mol.SetProp("_Name",        f"candidate_{rank}")
        mol.SetProp("SMILES",       row["smiles"])
        mol.SetProp("Score",        str(row["score"]))
        mol.SetProp("QED",          str(row["qed"]))
        mol.SetProp("MW",           str(row["mw"]))
        mol.SetProp("LogP",         str(row["logp"]))
        mol.SetProp("TPSA",         str(row["tpsa"]))
        mol.SetProp("Lipinski",     str(row["lipinski"]))
        mol.SetProp("SA_Score",     str(row.get("sa_score", "N/A")))
        if row.get("tanimoto_ref") is not None:
            mol.SetProp("Tanimoto_Ref", str(row["tanimoto_ref"]))
        writer.write(mol)
    writer.close()
    log.info(f"  SDF salvo: {out_path}")


def save_grid_image(df: pd.DataFrame, out_path: Path, top_n: int):
    """Salva grade de estruturas 2D dos top N candidatos."""
    try:
        mols   = []
        labels = []
        for rank, row in df.head(top_n).iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)
                labels.append(
                    f"#{rank} QED={row['qed']:.2f} "
                    f"MW={row['mw']:.0f} Score={row['score']:.3f}"
                )
        if not mols:
            return
        cols  = min(4, len(mols))
        rows  = (len(mols) + cols - 1) // cols
        img   = Draw.MolsToGridImage(
            mols, molsPerRow=cols,
            subImgSize=(320, 260),
            legends=labels,
        )
        img.save(str(out_path))
        log.info(f"  Imagem salva: {out_path}")
    except Exception as e:
        log.warning(f"  Erro ao gerar imagem: {e}")


def save_html_report(
    df: pd.DataFrame,
    out_path: Path,
    pdb_name: str,
    top_n: int,
    n_generated: int,
    generation_time: float,
    ckpt_info: dict,
):
    """Gera relatório HTML interativo com tabela ordenável."""
    top_df = df.head(top_n).copy()
    top_df = top_df.drop(columns=["_mol"], errors="ignore")

    # Formata colunas booleanas
    for col in ["lipinski", "veber"]:
        if col in top_df.columns:
            top_df[col] = top_df[col].map({True: "✓", False: "✗"})

    table_html = top_df.to_html(
        classes="table", border=0, index=True,
        na_rep="N/A", float_format=lambda x: f"{x:.3f}"
    )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>LigGenIA — Candidatos para {pdb_name}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #2c3e50; }}
  h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
  h2   {{ color: #3498db; margin-top: 2rem; }}
  .meta {{ background: #f8f9fa; border-left: 4px solid #3498db;
           padding: 1rem; margin: 1rem 0; border-radius: 4px; }}
  .meta p {{ margin: 0.3rem 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  th {{ background: #2c3e50; color: white; padding: 8px 12px;
        cursor: pointer; user-select: none; }}
  th:hover {{ background: #3498db; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #dee2e6; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #d6eaf8; }}
  .score-high {{ color: #27ae60; font-weight: bold; }}
  .score-mid  {{ color: #e67e22; }}
  .score-low  {{ color: #e74c3c; }}
  .smiles {{ font-family: monospace; font-size: 0.78rem; max-width: 300px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style>
</head>
<body>
<h1>🧬 LigGenIA — Candidatos a Ligantes</h1>
<div class="meta">
  <p><strong>Proteína:</strong> {pdb_name}</p>
  <p><strong>Modelo:</strong> epoch {ckpt_info.get('epoch','?')} |
     val_loss={ckpt_info.get('val_loss',0):.4f} |
     latent={ckpt_info.get('latent_dim',128)}-D</p>
  <p><strong>Candidatos gerados:</strong> {n_generated} amostras →
     {len(df)} válidos únicos</p>
  <p><strong>Tempo de geração:</strong> {generation_time:.1f}s</p>
  <p><strong>Data:</strong> {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
</div>

<h2>Top {top_n} Candidatos (ordenados por Score composto)</h2>
<p>
  <strong>Score</strong> = 0.4×QED + 0.2×SAscore + 0.15×Lipinski +
  0.10×Veber + 0.05×filtros ADMET + 0.10×Tanimoto(ref)<br>
  <strong>Lipinski:</strong> MW≤500, LogP≤5, HBD≤5, HBA≤10 |
  <strong>Veber:</strong> RotBonds≤10, TPSA≤140
</p>

{table_html}

<script>
// Ordenação de tabela por clique no cabeçalho
document.querySelectorAll('th').forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const rows  = Array.from(table.querySelectorAll('tr')).slice(1);
    const asc   = th.dataset.asc !== 'true';
    th.dataset.asc = asc;
    rows.sort((a, b) => {{
      const va = a.cells[idx]?.textContent.trim() || '';
      const vb = b.cells[idx]?.textContent.trim() || '';
      const na = parseFloat(va), nb = parseFloat(vb);
      return isNaN(na) || isNaN(nb)
        ? (asc ? va.localeCompare(vb) : vb.localeCompare(va))
        : (asc ? na - nb : nb - na);
    }});
    rows.forEach(r => table.tBodies[0].appendChild(r));
  }});
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info(f"  Relatório HTML: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gera ligantes candidatos para uma proteína externa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Geração básica
  python 07_generate.py --pdb minha_proteina.pdb

  # Com ligante de referência para cálculo de similaridade
  python 07_generate.py --pdb alvo.pdb --ligand-ref ligante_conhecido.sdf

  # Alta diversidade, muitos candidatos
  python 07_generate.py --pdb alvo.pdb --n-samples 500 --temperature 1.2 --top-n 50

  # Checkpoint específico
  python 07_generate.py --pdb alvo.pdb --ckpt checkpoints_phase3_mtb/
        """
    )
    parser.add_argument("--pdb",         required=True,
                        help="Arquivo PDB da proteína alvo")
    parser.add_argument(
        "--ckpt",
        default=str(PROJECT_DIR / "checkpoints" / "phase1"),
        help="diretório do checkpoint treinado",
    )
    parser.add_argument("--n-samples",   type=int,   default=200,
                        help="Número de amostras a gerar (default: 200)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperatura de amostragem — >1 mais diverso (default: 1.0)")
    parser.add_argument("--cutoff",      type=float, default=6.0,
                        help="Raio do sítio em Å (default: 6.0)")
    parser.add_argument("--min-atoms",   type=int,   default=5,
                        help="Mínimo de átomos no sítio (default: 5)")
    parser.add_argument("--top-n",       type=int,   default=20,
                        help="Top N candidatos no relatório (default: 20)")
    parser.add_argument("--out",         default=None,
                        help="Diretório de saída (default: resultados/{pdb_stem}/)")
    parser.add_argument("--ligand-ref",  default=None,
                        help="SDF do ligante de referência para cálculo de Tanimoto")
    parser.add_argument("--tensor-dir",  default=None,
                        help="Diretório de tensores da fase usada para obter o template graph")
    parser.add_argument("--site-pdb",    default=None,
                        help="PDB contendo apenas os resíduos do sítio de ligação")
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        log.error(f"Arquivo PDB não encontrado: {pdb_path}")
        sys.exit(1)

    ckpt_dir = Path(args.ckpt)
    tensor_dir = Path(args.tensor_dir) if args.tensor_dir else (PROJECT_DIR / "data" / "tensors")
    site_pdb_path = Path(args.site_pdb) if args.site_pdb else None
    out_dir = Path(args.out) if args.out else PROJECT_DIR / "results" / pdb_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("LIGENIA — Geração de Candidatos a Ligantes")
    log.info("=" * 65)
    log.info(f"  Proteína  : {pdb_path.name}")
    log.info(f"  Checkpoint: {ckpt_dir}")
    log.info(f"  Amostras  : {args.n_samples}  |  Temperatura: {args.temperature}")
    log.info(f"  Saída     : {out_dir}")
    log.info(f"  Device    : {DEVICE}")

    t_total = time.perf_counter()

    # ── 1. Extrai e voxeliza o sítio ──────────────────────────────────────────
    log.info("\n[1/5] Extraindo sítio de ligação...")
    if site_pdb_path:
        if not site_pdb_path.exists():
            log.error(f"--site-pdb não encontrado: {site_pdb_path}")
            sys.exit(1)
        log.info(f"  Modo: sítio fornecido ({site_pdb_path.name})")
    else:
        log.info("  Modo: detecção automática via ligante co-cristalizado")

    site_tensor, origin, voxel_size = pdb_to_site_tensor(
        pdb_path, args.cutoff, args.min_atoms,
        site_pdb=site_pdb_path,
    )
    np.savez_compressed(
        out_dir / "site_tensor.npz",
        tensor=site_tensor, origin=origin,
        voxel_size=np.array([voxel_size]),
    )

    # ── 2. Carrega modelo ──────────────────────────────────────────────────────
    log.info("\n[2/5] Carregando modelo CVAE...")
    model, mod = load_model(ckpt_dir)

    ckpt_info = torch.load(
        ckpt_dir / "best_model.pt",
        map_location="cpu", weights_only=False
    )
    ckpt_meta = {
        "epoch":      ckpt_info.get("epoch", "?"),
        "val_loss":   ckpt_info.get("val_loss", 0),
        "latent_dim": ckpt_info.get("config", {}).get("latent_dim", 128),
    }

    # ── 3. Ligante de referência (opcional) ───────────────────────────────────
    ref_mol = None
    if args.ligand_ref:
        ref_path = Path(args.ligand_ref)
        if ref_path.exists():
            suppl = Chem.SDMolSupplier(str(ref_path), removeHs=True)
            ref_mol = next((m for m in suppl if m is not None), None)
            if ref_mol:
                log.info(f"\n  Referência: {ref_path.name} "
                         f"({ref_mol.GetNumAtoms()} átomos)")
        else:
            log.warning(f"Ligante de referência não encontrado: {ref_path}")

    # ── 4. Gera candidatos ────────────────────────────────────────────────────
    log.info(f"\n[3/5] Gerando {args.n_samples} candidatos...")
    t_gen = time.perf_counter()
    candidates = generate_candidates(
        model,
        mod,
        site_tensor,
        tensor_dir=tensor_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
        voxel_size=voxel_size,              # ← passar o valor real do sítio
    )
    gen_time = time.perf_counter() - t_gen

    if not candidates:
        log.error("Nenhum candidato válido gerado. "
                  "Tente aumentar --n-samples ou --temperature.")
        sys.exit(1)

    # ── 5. Ranking e métricas ─────────────────────────────────────────────────
    log.info("\n[4/5] Ranqueando candidatos...")
    df = rank_candidates(candidates, ref_mol=ref_mol)

    # ── 6. Salva resultados ───────────────────────────────────────────────────
    log.info("\n[5/5] Salvando resultados...")

    # CSV completo
    df_csv = df.drop(columns=["_mol"], errors="ignore")
    df_csv.to_csv(out_dir / "candidates.csv")
    log.info(f"  CSV completo: {out_dir / 'candidates.csv'}")

    # SDF top N
    save_sdf(df, out_dir / f"top{args.top_n}_candidates.sdf", args.top_n)

    # Imagem grade 2D
    save_grid_image(df, out_dir / f"top{args.top_n}_candidates.png", args.top_n)

    # Relatório HTML
    save_html_report(
        df, out_dir / "report.html",
        pdb_name=pdb_path.name,
        top_n=args.top_n,
        n_generated=args.n_samples,
        generation_time=gen_time,
        ckpt_info=ckpt_meta,
    )

    # ── Resumo final ──────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    top5 = df.head(5)

    log.info("\n" + "=" * 65)
    log.info("RESUMO")
    log.info("=" * 65)
    log.info(f"  Candidatos gerados   : {args.n_samples}")
    log.info(f"  Válidos únicos       : {len(df)}")
    log.info(f"  Tempo total          : {total_time:.1f}s")
    log.info(f"  Saída                : {out_dir.resolve()}")
    log.info(f"\n  TOP 5 CANDIDATOS:")
    log.info(f"  {'#':>3}  {'Score':>6}  {'QED':>5}  {'MW':>6}  "
             f"{'LogP':>5}  {'Lip':>4}  SMILES")
    log.info("  " + "-"*65)
    for rank, row in top5.iterrows():
        smi_short = row["smiles"][:45] + "..." if len(row["smiles"]) > 45 else row["smiles"]
        log.info(f"  {rank:>3}  {row['score']:>6.3f}  {row['qed']:>5.3f}  "
                 f"{row['mw']:>6.1f}  {row['logp']:>5.2f}  "
                 f"{'✓' if row['lipinski'] else '✗':>4}  {smi_short}")


if __name__ == "__main__":
    main()
