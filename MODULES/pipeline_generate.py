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
import shutil
import subprocess
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

# GROMACS + AmberTools (antechamber/parmchk2/tleap, via acpype): opcionais,
# usados apenas no caminho de minimização "amber_ff14sb"/"charmm36" da
# geometria final (ver _relax_generated_geometry / FINAL_FF_METHODS). Nenhum
# dos dois é necessário para o caminho padrão (RDKit MMFF/UFF).
#
# Escolhido no lugar de openmm+openmmforcefields+openff-toolkit (tentativa
# anterior, nunca validada — openff-toolkit só existe via conda-forge, não
# pip, para nenhuma versão de Python) porque GROMACS é uma ferramenta de
# linha de comando compartilhada com o restante do conjunto CODOC/CODYN/
# CODRUG (o CODYN já depende dele para dinâmica molecular) e é chamada via
# subprocess — nunca precisa ser importada dentro do venv do CODEEP,
# elimina o problema de distribuição do openff-toolkit por completo.
#
# acpype (pip-instalável) orquestra antechamber/parmchk2/tleap (AmberTools)
# e escreve .itp/.gro prontos para GROMACS — mas os binários do AmberTools
# em si continuam sendo instalados via conda-forge tipicamente
# (`conda install -c conda-forge ambertools`) e precisam estar no PATH (ou
# em config["ambertools_bin_dir"], prepend ao PATH da chamada) quando o
# acpype for executado.
HAS_GROMACS    = shutil.which("gmx") is not None
HAS_ACPYPE     = shutil.which("acpype") is not None
HAS_AMBERTOOLS = shutil.which("antechamber") is not None
HAS_GROMACS_STACK = HAS_GROMACS and HAS_ACPYPE

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
) -> tuple[np.ndarray, list[str], np.ndarray, list]:
    """Extrai o sítio de ligação de um PDB com ligante co-cristalizado.

    Estratégia:
      1. Procura ligantes HETATM orgânicos (com carbono, ≥7 átomos pesados)
      2. Para cada ligante, identifica RESÍDUOS proteicos com pelo menos um
         átomo a ≤ cutoff_A Å — e inclui o resíduo INTEIRO (backbone +
         cadeia lateral), nunca um subconjunto parcial de átomos. Mesma
         regra de seleção usada no treino (pipeline_curate.get_binding_site)
         — necessário para load_site_atoms_protonated/protonate_residue
         funcionarem corretamente (um resíduo com átomos faltando NO MEIO,
         não só nas bordas da cadeia, confundiria a lógica de H-capping).
      3. Retorna coordenadas, elementos, cargas aproximadas E a lista de
         objetos de resíduo Bio.PDB (para protonação — ver
         pipeline_voxelize.protonate_site_residues).

    Returns:
        coords   : (N, 3) float32 — coordenadas dos átomos do sítio
        elements : [N] str        — símbolos dos elementos
        charges  : (N,) float32   — cargas parciais aproximadas por elemento
        residues : list           — objetos Bio.PDB Residue selecionados
                                     (resíduos completos), na mesma ordem
                                     em que seus átomos aparecem em coords
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

    # Coleta todos os átomos proteicos, com o resíduo de origem de cada um
    prot_coords, prot_elems, prot_residues = [], [], []
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
                        prot_residues.append(residue)

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
        # Pega os resíduos cujo átomo mais próximo está entre os 80
        # mais próximos do centro geométrico (sítio aproximado)
        dists  = np.linalg.norm(prot_arr - center, axis=1)
        idx    = np.argsort(dists)[:80]
        site_residue_ids = {id(prot_residues[i]) for i in idx}
    else:
        # Resíduos com ao menos um átomo a ≤ cutoff_A do ligante
        indices = prot_tree.query_ball_point(best_ligand_coords, r=cutoff_A)
        atom_idx = sorted(set(i for sublist in indices for i in sublist))
        if len(atom_idx) < min_site_residues:
            raise ValueError(
                f"Sítio muito pequeno ({len(atom_idx)} átomos). "
                f"Tente aumentar --cutoff (atual: {cutoff_A} Å)."
            )
        site_residue_ids = {id(prot_residues[i]) for i in atom_idx}

    # Expande para o resíduo INTEIRO (todos os átomos, não só os que caíram
    # dentro do cutoff) — mesma regra de pipeline_curate.get_binding_site.
    seen = set()
    residues, site_coords, site_elems = [], [], []
    for coord, elem, residue in zip(prot_coords, prot_elems, prot_residues):
        if id(residue) not in site_residue_ids:
            continue
        site_coords.append(coord)
        site_elems.append(elem)
        if id(residue) not in seen:
            seen.add(id(residue))
            residues.append(residue)

    site_coords = np.array(site_coords, dtype=np.float32)
    charges = np.array(
        [ELEM_CHARGE.get(e, 0.0) for e in site_elems],
        dtype=np.float32
    )

    log.info(f"  Sítio extraído: {len(site_coords)} átomos pesados "
             f"({len(residues)} resíduos completos)")
    return site_coords, site_elems, charges, residues


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

    coords, elements, residues = [], [], []
    for model in struct:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in SKIP_RESNAMES:
                    continue
                added = False
                for atom in residue.get_atoms():
                    e = _infer_element(atom)
                    if e not in ("H","D","X",""):
                        coords.append(atom.get_vector().get_array())
                        elements.append(e)
                        added = True
                if added:
                    residues.append(residue)

    if not coords:
        raise ValueError(f"Nenhum átomo encontrado em {site_pdb}")

    coords_arr = np.array(coords, dtype=np.float32)
    charges    = np.array([ELEM_CHARGE.get(e, 0.0) for e in elements],
                          dtype=np.float32)
    log.info(f"  Sítio carregado de {site_pdb.name}: "
             f"{len(coords_arr)} átomos pesados ({len(residues)} resíduos)")
    return coords_arr, elements, charges, residues

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
    config_overrides: "dict | None" = None,
) -> np.ndarray:
    """Pipeline completo: PDB → tensor do sítio (5, 32, 32, 32).

    Se site_pdb for fornecido, carrega o sítio diretamente desse arquivo
    (sem detecção automática). Caso contrário, extrai automaticamente a
    partir do ligante co-cristalizado em pdb_path.

    config_overrides permite sobrescrever ph/protonate_site/site_qm_method
    (mesmas chaves de pipeline_voxelize.VOXEL_CONFIG) — por padrão usa os
    valores default do módulo, que DEVEM ser os mesmos usados para gerar
    os tensores de treino (ver nota de paridade abaixo).

    Returns:
        tensor, origin, voxel_size, site_coords, site_elements, site_mol,
        site_is_backbone — os quatro últimos descrevem os átomos
        protonados do sítio: coords (N,3)/elementos [N] em arrays soltos
        (para a checagem de colisão estérica), site_mol (RDKit Mol único
        com as ligações intra-resíduo reais preservadas, mesma ordem de
        átomos) e site_is_backbone (bool por átomo — True para N/Cα/C/O,
        usados pela minimização final em modo "induced-fit", ver
        _minimize_complex_rdkit/_minimize_complex_gromacs).
    """
    # Carrega módulo de voxelização
    _proj = Path(project_dir) if project_dir else PROJECT_DIR
    spec = importlib.util.spec_from_file_location(
        "pipeline_voxelize", _proj / "MODULES" / "pipeline_voxelize.py")
    vox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vox)

    config = vox.VOXEL_CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)
    n = config["grid_size"]  # 32

    # Extrai sítio — modo manual (--site-pdb) ou automático. Em ambos os
    # casos recebe também a lista de resíduos Bio.PDB (resíduos completos)
    # para a etapa de protonação abaixo.
    if site_pdb is not None:
        site_coords, site_elements, site_charges, site_residues = \
            load_site_from_residues_pdb(site_pdb)
    else:
        site_coords, site_elements, site_charges, site_residues = \
            extract_site_from_pdb(pdb_path, cutoff_A, min_site_residues)

    # Protonação por pH (ver pipeline_voxelize.protonate_site_residues) —
    # MESMA função usada em pipeline_voxelize.process_pair (treino), com o
    # MESMO pH default (config["ph"]). Isso é o que garante que o tensor de
    # inferência tenha a mesma distribuição do tensor de treino: qualquer
    # divergência aqui (pH diferente, protonate_site desligado só de um
    # lado) reabriria um mismatch treino↔inferência como o que motivou os
    # canais 3-4 serem site-derived (ver pipeline_voxelize BLOCO 3B).
    if config.get("protonate_site", True):
        site_coords, site_elements, site_charges, site_fragments = \
            vox.protonate_site_residues(site_residues, ph=config.get("ph", 7.4))
    else:
        site_fragments = []

    # Mol combinado do sítio (ligações intra-resíduo reais preservadas) +
    # máscara backbone/cadeia-lateral por átomo — usados pela minimização
    # final em modo "induced-fit" (ver pipeline_generate._minimize_complex_*).
    # Sem protonação (protonate_site=False), não há nomes de átomo/
    # conectividade — o sítio inteiro entra como sempre-backbone (sempre
    # travado; induced-fit não tem efeito nesse caso).
    site_fragments_for_mol = site_fragments if site_fragments else [
        {"resname": "UNK", "start": 0, "end": len(site_coords), "mol": None}
    ]
    site_mol, site_is_backbone = vox.build_combined_site_mol(
        site_coords, site_elements, site_fragments_for_mol
    )

    # Grade 3D
    grid_coords, origin, voxel_size = vox.build_grid(site_coords, config)

    # Canais 0-2: voxelização vetorizada (NumPy ou CuPy)
    # sigma segue a MESMA fórmula usada em pipeline_voxelize.process_pair
    # (piso de voxel_size) — necessário para que o tensor de inferência seja
    # idêntico em distribuição ao tensor de treino.
    sigma = max(config["gaussian_sigma"], voxel_size * 1.0)
    w_vdw = np.array(
        [vox.VDW_RADII.get(e.upper(), vox.VDW_DEFAULT) for e in site_elements],
        dtype=np.float32
    )
    gf = grid_coords.reshape(-1, 3)
    ch0, ch1, ch2 = vox._voxelize_atoms(
        gf, site_coords, w_vdw, site_charges, sigma, (n, n, n)
    )

    # Canais 3-4: campo doador/aceptor de elétrons do SÍTIO (HOMO/LUMO-like)
    # — mesmo dispatcher usado em pipeline_voxelize.process_pair
    # (channel_site_orbital_field: QM fragmentado / xTB / heurística
    # conforme config["site_qm_method"]), calculado apenas a partir dos
    # átomos do sítio. Não depende de nenhum ligante de referência.
    sigma_da = max(config["gaussian_sigma"] * 1.5, voxel_size)
    ch3, ch4 = vox.channel_site_orbital_field(
        gf, site_coords, site_elements, site_charges, site_fragments,
        sigma_da, (n, n, n), config,
    )

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
    # site_coords/site_elements/site_mol/site_is_backbone (protonados, com H
    # explícitos) também são retornados — usados na minimização conjunta
    # ligante+bolso (rígida ou induced-fit) e na checagem de colisão
    # estérica pós-geração (ver _relax_generated_geometry/_check_pocket_clashes).
    return tensor, origin, voxel_size, site_coords, site_elements, site_mol, site_is_backbone


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — CARREGAMENTO DO MODELO
# ══════════════════════════════════════════════════════════════════════════════

def resolve_site_config_overrides(
    ckpt_voxel_config: "dict | None",
    ui_overrides: "dict | None" = None,
) -> dict:
    """Resolve a config_overrides final para pdb_to_site_tensor na geração,
    dando PRIORIDADE ABSOLUTA ao que está gravado no checkpoint (ver
    load_model) sobre qualquer valor escolhido manualmente na interface —
    fecha o mismatch treino↔inferência que motivou site_qm_method/qm_basis/
    ph/protonate_site existirem como CONFIG DE UM CHECKPOINT, não como
    escolha livre a cada rodada de geração (um sítio novo tem que ser
    revoxelizado com a MESMA config usada para treinar o encoder que vai
    processá-lo).

    ui_overrides só prevalece (a) integralmente, quando o checkpoint não
    tem voxel_config gravada (checkpoints treinados antes desta
    funcionalidade existir — fallback para o comportamento manual legado);
    (b) chave a chave, para qualquer chave que o checkpoint não tenha
    gravado. Loga um aviso sempre que um valor manual conflita com o do
    checkpoint, para deixar claro por que a escolha da interface foi
    ignorada em vez de falhar silenciosamente.
    """
    ui_overrides = dict(ui_overrides or {})
    if not ckpt_voxel_config:
        return ui_overrides
    resolved = dict(ui_overrides)
    for key, ckpt_value in ckpt_voxel_config.items():
        if key in ui_overrides and ui_overrides[key] != ckpt_value:
            log.warning(
                f"  {key}: valor manual da interface ({ui_overrides[key]!r}) "
                f"ignorado — usando o valor gravado no checkpoint "
                f"({ckpt_value!r}) para manter a paridade com o tensor de treino."
            )
        resolved[key] = ckpt_value
    return resolved

def load_model(
    ckpt_dir: Path,
    project_dir: "Path | None" = None,
):
    """Carrega o modelo CVAE do melhor checkpoint.

    Returns (model, mod, voxel_config): voxel_config é o retrato da config
    de voxelização (ver pipeline_voxelize.VOXEL_CONFIG_TENSOR_KEYS /
    save_voxel_config_snapshot) embutido no checkpoint pelo treino que o
    gerou — {} para checkpoints treinados antes desta funcionalidade
    existir. Usado pelo chamador (pdb_to_site_tensor) para reconstruir o
    tensor de um sítio NOVO com a MESMA config usada no treino, em vez de
    depender de o usuário reproduzir manualmente esses valores na interface
    (ver nota de paridade treino/inferência em pdb_to_site_tensor)."""
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
    voxel_config = cfg.get("voxel_config", {}) or {}

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
    if voxel_config:
        log.info(f"  Voxel config herdada do checkpoint: {voxel_config}")
    else:
        log.warning(
            "  Checkpoint sem voxel_config embutida (treinado antes desta "
            "funcionalidade existir) — a geração usará a config manual fornecida "
            "pelo chamador (Etapa 8 da interface, ou defaults de VOXEL_CONFIG)."
        )

    return model, mod, voxel_config


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — GERAÇÃO E DECODIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

ATOM_TYPES_GEN = ["C","N","O","S","P","F","Cl","Br","I","Fe","Zn","Mg","C","C"]

MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "F": 1,
    "CL": 1, "BR": 1, "I": 1, "FE": 6, "ZN": 4, "MG": 2,
}

# Raios de van der Waals por elemento (Å) — Bondi, 1964. Mesma tabela usada
# em pipeline_voxelize.VDW_RADII; duplicada aqui (pequena, chamada por
# candidato) em vez de importar o módulo inteiro dinamicamente no caminho
# quente da checagem de colisão.
VDW_RADII_CLASH = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 2.05, "ZN": 1.39, "MG": 1.73, "CA": 2.31, "MN": 1.97,
    "CU": 1.40, "NI": 1.63, "CO": 1.70, "SE": 1.90,
}
VDW_RADIUS_DEFAULT = 1.70

# Fatores de tolerância de colisão/geometria — mesmos limiares documentados
# pelo PoseBusters (Buttenschoen, Morris & Deane, 2024): pares atômicos
# mais próximos que CLASH_VDW_FACTOR × soma dos raios de vdW são
# considerados colisão estérica; ligações fora de
# [BOND_LENGTH_TOL_LOW, BOND_LENGTH_TOL_HIGH] × comprimento canônico são
# consideradas geometricamente implausíveis.
CLASH_VDW_FACTOR = 0.75
BOND_LENGTH_TOL_LOW = 0.75
BOND_LENGTH_TOL_HIGH = 1.25

# Comprimento canônico de referência por tipo de ligação (Å) — usado só
# para a checagem de plausibilidade PÓS-minimização (não confundir com
# _bond_distance_ok, que é o limiar usado na CONSTRUÇÃO da molécula).
_CANONICAL_BOND_LENGTH = {
    BondType.SINGLE: 1.50, BondType.DOUBLE: 1.32,
    BondType.TRIPLE: 1.18, BondType.AROMATIC: 1.39,
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


FINAL_FF_METHODS = ("mmff_uff", "amber_ff14sb", "charmm36")

# Nome do campo de força GROMACS (pasta <nome>.ff) para cada opção exposta
# na interface. GROMACS não traz amber14sb/charmm36 nativamente (só até
# amber99sb-ildn/charmm27) — se uma pasta <nome_real>.ff estiver presente em
# GROMACS_FF_EXTRA_DIR, ela é usada (ver _resolve_gromacs_ff); caso
# contrário, cai no equivalente já empacotado com o GROMACS, com aviso no
# log. amber99sb-ildn e charmm27 são amplamente usados e validados — uma
# aproximação razoável, não os arquivos mais recentes publicados.
GROMACS_FF_PREFERRED = {"amber_ff14sb": "amber14sb", "charmm36": "charmm36"}
GROMACS_FF_FALLBACK  = {"amber_ff14sb": "amber99sb-ildn", "charmm36": "charmm27"}

# Pasta onde o CODEEP procura versões mais recentes dos campos de força
# (amber14sb.ff/charmm36.ff) antes de cair no fallback acima. Baixe uma vez
# de uma fonte confiável (ex. mirror oficial AMBER/CHARMM-GUI) e coloque
# aqui — nunca no diretório compartilhado da instalação do GROMACS.
GROMACS_FF_EXTRA_DIR = Path(
    globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent)
) / "MODULES" / "forcefields"

# Nomes de átomo do backbone (convenção PDB/AMBER) — mesma definição de
# pipeline_voxelize.BACKBONE_ATOM_NAMES, duplicada aqui porque este módulo
# só carrega pipeline_voxelize dinamicamente dentro de funções (nunca em
# nível de módulo, ver pdb_to_site_tensor), então não há import direto a
# reaproveitar. Usada para derivar o backbone flag NA ORDEM DE ÁTOMOS QUE O
# PRÓPRIO pdb2gmx PRODUZ (ver _build_gromacs_site_topology) — não podemos
# reaproveitar a lista site_is_backbone original (ordem/índices do site_mol
# protonado pelo RDKit): com -ignh o pdb2gmx descarta nossos H e reconstrói
# os dele, em posições/ordem própria do campo de força, então o mapeamento
# índice-a-índice com a lista original ficaria errado (grupo [Frozen]
# apontando para átomos trocados, gerando índice fora do intervalo do
# sítio ou travando/liberando o átomo errado).
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}


def _build_ligand_site_complex(ligand_mol: "Chem.Mol", site_mol: "Chem.Mol") -> tuple:
    """Combina o ligante com o Mol do sítio (já com as ligações intra-
    resíduo REAIS preservadas, ver
    pipeline_voxelize.build_combined_site_mol) num único Mol.

    Ter as ligações reais do sítio presentes é o que torna o modo
    "induced-fit" (Seção _minimize_complex_rdkit) fisicamente sensato: se
    átomos de cadeia lateral vão poder se mover, o campo de força precisa
    dessas ligações para manter a geometria interna do resíduo (ângulos,
    comprimentos de ligação) sã durante a minimização — sem elas, nada
    impediria a cadeia lateral de colapsar sobre si mesma.

    Returns: (combined_mol, n_ligand_atoms) — os primeiros n_ligand_atoms
    átomos de combined_mol são o ligante, na mesma ordem do mol de entrada;
    o resto é o sítio, na mesma ordem de site_mol (e portanto de
    site_is_backbone, ver pdb_to_site_tensor).
    """
    combined = Chem.CombineMols(ligand_mol, site_mol)
    Chem.SanitizeMol(combined, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS)
    return combined, ligand_mol.GetNumAtoms()


def _minimize_complex_rdkit(
    combined: "Chem.Mol",
    n_ligand_atoms: int,
    site_is_backbone: list,
    induced_fit: bool = False,
    max_iters: int = 500,
) -> bool:
    """Minimiza os átomos do ligante (índices 0..n_ligand_atoms-1) dentro
    do campo de força combinado. Os átomos do sítio contribuem termos
    não-ligados à energia sempre; ficam travados (AddFixedPoint) ou não
    conforme `induced_fit`:

      induced_fit=False (padrão): TODOS os átomos do sítio travados —
        bolso rígido, só o ligante se move.
      induced_fit=True: só os átomos de BACKBONE (N, Cα, C, O — ver
        site_is_backbone) ficam travados; os de cadeia lateral se movem
        livremente, ajustando-se ao ligante durante a mesma minimização
        (não é uma dinâmica térmica separada — é o mesmo passo de
        minimização de energia, só que agora com mais graus de liberdade
        livres). Backbone travado representa o fato de que só temos o
        bolso recortado, não a proteína inteira — sem o resto da cadeia
        seguraria essa porção no lugar; deixá-lo livre deixaria os
        resíduos "flutuarem" sem o suporte estrutural real.

    Tenta MMFF94s primeiro (mesma preferência usada para o ligante
    isolado); cai para UFF se o MMFF não reconhecer algum átomo do sítio
    — comum para metais (Zn, Fe, Mg, frequentes em cofatores de sítios
    catalíticos), já que o MMFF94 foi parametrizado majoritariamente para
    moléculas orgânicas drug-like, não para ambientes proteicos com
    metais. UFF cobre a tabela periódica de forma mais genérica, à custa
    de precisão — validado empiricamente: constrói o campo de força e
    minimiza mesmo com um átomo de Zn²⁺ não reconhecido pelo tipador.

    Retorna (sucesso, site_coords_finais). site_coords_finais é None
    quando induced_fit=False (nada no sítio se move — a chamadora já tem
    as coordenadas originais) ou em caso de falha; quando induced_fit=True
    e a minimização roda, traz as posições PÓS-minimização de TODOS os
    átomos do sítio (não só o backbone travado) — necessário para a
    checagem de colisão (_check_pocket_clashes) comparar contra a
    geometria real após o ajuste, não contra a posição original da cadeia
    lateral que acabou de se mover.
    """
    ff = None
    try:
        if AllChem.MMFFHasAllMoleculeParams(combined):
            props = AllChem.MMFFGetMoleculeProperties(combined)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(combined, props)
    except Exception:
        ff = None
    if ff is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(combined)
        except Exception:
            ff = None
    if ff is None:
        return False, None

    for offset, is_bb in enumerate(site_is_backbone):
        if (not induced_fit) or is_bb:
            ff.AddFixedPoint(n_ligand_atoms + offset)
    try:
        ff.Minimize(maxIts=max_iters)
    except Exception:
        return False, None

    site_coords_final = None
    if induced_fit:
        conf = combined.GetConformer()
        site_coords_final = np.array([
            [conf.GetAtomPosition(n_ligand_atoms + k).x,
             conf.GetAtomPosition(n_ligand_atoms + k).y,
             conf.GetAtomPosition(n_ligand_atoms + k).z]
            for k in range(len(site_is_backbone))
        ], dtype=np.float32)
    return True, site_coords_final


def _resolve_gromacs_ff(protein_ff: str) -> str:
    """Resolve final_ff ('amber_ff14sb'/'charmm36') para o nome de pasta
    <nome>.ff que o `gmx pdb2gmx` deve usar: prefere a versão mais recente
    se presente em GROMACS_FF_EXTRA_DIR, senão cai no equivalente já
    empacotado com o GROMACS (aviso no log — ver GROMACS_FF_FALLBACK)."""
    preferred = GROMACS_FF_PREFERRED.get(protein_ff, protein_ff)
    fallback  = GROMACS_FF_FALLBACK.get(protein_ff, protein_ff)
    if (GROMACS_FF_EXTRA_DIR / f"{preferred}.ff").is_dir():
        return preferred
    log.info(
        f"  {preferred}.ff não encontrado em {GROMACS_FF_EXTRA_DIR} — usando "
        f"{fallback} (já empacotado com o GROMACS). Para usar {preferred} de "
        f"verdade, baixe a pasta {preferred}.ff de uma fonte confiável e "
        f"coloque em {GROMACS_FF_EXTRA_DIR}/ (nunca no diretório compartilhado "
        f"da instalação do GROMACS)."
    )
    return fallback


def _infer_amber_resname(residue_atoms: list, resname: str) -> str:
    """Lê o estado de protonação REALMENTE aplicado a um resíduo do sítio
    (ver pipeline_voxelize.protonate_residue) a partir da geometria já
    construída — não re-deriva da regra de pH uma segunda vez — e retorna
    o nome de resíduo AMBER correspondente (ASH/GLH/HID/HIE/HIP/LYN quando
    aplicável). Garante que a topologia GROMACS reflita EXATAMENTE a mesma
    decisão de protonação usada nos canais 3-4 do tensor do sítio (Seção 3
    da metodologia) — sem isso, pdb2gmx assumiria estados padrão fixos,
    sem noção nenhuma do pH configurado.

    residue_atoms: átomos RDKit (de site_mol) pertencentes a UM resíduo —
    GetNeighbors() continua refletindo as ligações reais dentro de
    site_mol (preservadas por Chem.CombineMols), então funciona igual a
    operar sobre o Mol isolado do resíduo.
    """
    def _n_h(atom_name):
        for atom in residue_atoms:
            info = atom.GetPDBResidueInfo()
            if info is not None and info.GetName().strip() == atom_name:
                return sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "H")
        return None

    if resname == "ASP":
        return "ASH" if _n_h("OD2") == 1 else "ASP"
    if resname == "GLU":
        return "GLH" if _n_h("OE2") == 1 else "GLU"
    if resname == "LYS":
        return "LYS" if _n_h("NZ") == 3 else "LYN"
    if resname == "HIS":
        nd1_h, ne2_h = _n_h("ND1"), _n_h("NE2")
        if nd1_h == 1 and ne2_h == 1:
            return "HIP"
        if nd1_h == 1:
            return "HID"
        return "HIE"
    return resname


def _build_gromacs_site_topology(
    site_mol: "Chem.Mol",
    protein_ff: str,
    workdir: Path,
) -> "dict | None":
    """Constrói a topologia GROMACS do sítio UMA VEZ por rodada de geração
    (reutilizável por todos os N candidatos — o sítio não muda entre
    candidatos, só o ligante varia). Escreve os átomos pesados do sítio
    (nomes de resíduo já ajustados ao estado de protonação real, ver
    _infer_amber_resname) num PDB e roda `gmx pdb2gmx -ignh` (reconstrói
    os hidrogênios do zero segundo as regras do próprio campo de força,
    consistente com o nome de resíduo já ajustado — mais robusto do que
    tentar casar nossos H adicionados via RDKit com a convenção de nomes
    esperada pelo GROMACS).

    site_mol: Mol combinado do sítio (ver
    pipeline_voxelize.build_combined_site_mol) — os átomos são agrupados
    por resíduo aqui mesmo, a partir do PDBResidueInfo de cada um (nome +
    número do resíduo), sem precisar da lista de fragments separada.

    Termini (N/C-terminal) tratados com a opção padrão do pdb2gmx — o
    sítio recortado não tem N/C-terminal reais (é um pedaço da cadeia,
    não a proteína inteira), então essa escolha é uma aproximação, no
    mesmo espírito do "capping" já aceito na protonação por resíduo (ver
    pipeline_voxelize.protonate_residue).

    Returns dict {"top", "gro", "n_atoms", "ff"} em sucesso, None em
    qualquer falha (o chamador cai para RDKit).
    """
    if not (HAS_GROMACS and HAS_ACPYPE):
        return None
    gmx = shutil.which("gmx")
    try:
        # Agrupa átomos de site_mol por resíduo (nome+número+cadeia do
        # PDBResidueInfo), preservando a ordem de aparição. Átomos sem
        # PDBResidueInfo (fragmentos fallback, sem protonação bem-sucedida)
        # formam grupos de 1 átomo cada — entram com o nome de elemento
        # como nome de átomo/resíduo, quase certamente sem template
        # correspondente (falha adiante, tratada com try/except).
        residues: dict = {}
        order: list = []
        for atom in site_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None:
                key = (info.GetResidueName(), info.GetResidueNumber(), info.GetChainId())
            else:
                key = ("UNK", atom.GetIdx(), "")
            if key not in residues:
                residues[key] = []
                order.append(key)
            residues[key].append(atom)

        pdb_path = workdir / "site.pdb"
        lines, serial = [], 1
        conf = site_mol.GetConformer()
        for key in order:
            residue_atoms = residues[key]
            resname = key[0]
            amber_resname = _infer_amber_resname(residue_atoms, resname)
            resnum = key[1]
            for atom in residue_atoms:
                if atom.GetSymbol() == "H":
                    continue   # -ignh reconstrói do zero
                info = atom.GetPDBResidueInfo()
                name = info.GetName().strip() if info else atom.GetSymbol()
                p = conf.GetAtomPosition(atom.GetIdx())
                name_f = f" {name:<3s}" if len(name) < 4 else name[:4]
                lines.append(
                    f"ATOM  {serial:5d} {name_f:<4s} {amber_resname:<3s} "
                    f"A{resnum:4d}    {p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00"
                )
                serial += 1
        if serial == 1:
            return None   # nenhum átomo pesado — nada a fazer
        lines += ["TER", "END"]
        pdb_path.write_text("\n".join(lines))

        ff_name  = _resolve_gromacs_ff(protein_ff)
        top_path = workdir / "site.top"
        gro_path = workdir / "site.gro"
        result = subprocess.run(
            [gmx, "pdb2gmx", "-f", str(pdb_path), "-o", str(gro_path),
             "-p", str(top_path), "-i", str(workdir / "posre.itp"),
             "-ff", ff_name, "-water", "none", "-ignh", "-ter"],
            input="1\n1\n", capture_output=True, text=True,
            cwd=str(workdir), timeout=120,
        )
        if result.returncode != 0 or not top_path.exists() or not gro_path.exists():
            log.debug(f"  pdb2gmx falhou: {result.stderr[-800:]}")
            return None

        gro_lines = gro_path.read_text().splitlines()
        n_atoms = int(gro_lines[1].strip())
        atom_names = [line[10:15].strip() for line in gro_lines[2:2 + n_atoms]]
        # Backbone flag e elemento NA ORDEM/ÍNDICES do .gro que o pdb2gmx
        # efetivamente produziu (ver comentário de BACKBONE_ATOM_NAMES
        # acima) — não da lista site_is_backbone/site_elements original
        # (RDKit), que não corresponde mais 1:1 depois do -ignh reconstruir
        # os hidrogênios do zero (conta e ordem de átomos mudam). Elemento
        # aproximado pela 1a letra do nome (convenção AMBER/GROMACS: nomes
        # de H sempre começam com "H", pesados com o símbolo do elemento —
        # válido para os elementos possíveis num resíduo de aminoácido
        # padrão: C/H/N/O/S; site_mol só contém fragmentos de aminoácido,
        # ver build_combined_site_mol).
        is_backbone = [name in BACKBONE_ATOM_NAMES for name in atom_names]
        elements = [name[0] for name in atom_names]
        return {
            "top": top_path, "gro": gro_path, "n_atoms": n_atoms, "ff": ff_name,
            "is_backbone": is_backbone, "elements": elements,
        }
    except Exception as e:
        log.debug(f"  _build_gromacs_site_topology falhou: {type(e).__name__}: {e}")
        return None


def _minimize_complex_gromacs(
    ligand_mol: "Chem.Mol",
    site_topology: "dict | None",
    induced_fit: bool = False,
    ligand_charge_method: str = "gas",
) -> tuple:
    """Minimiza o complexo via GROMACS + AmberTools (antechamber/parmchk2/
    tleap, orquestrados pelo `acpype`, chamados por subprocess — nunca
    importados dentro do venv do CODEEP) — proteína com campo de força
    real (AMBER/CHARMM, ver site_topology), ligante parametrizado como
    GAFF2. Reaproveita site_topology (ver _build_gromacs_site_topology,
    construída UMA VEZ por sítio) — só o ligante é reparametrizado a cada
    chamada.

    Ao contrário de _minimize_complex_rdkit, NÃO recebe site_is_backbone
    como parâmetro — o backbone flag usado para o grupo [Frozen] vem de
    site_topology["is_backbone"] (derivado da ordem de átomos que o
    próprio pdb2gmx produziu, ver _build_gromacs_site_topology), porque
    -ignh reconstrói os hidrogênios do zero e invalida qualquer
    correspondência índice-a-índice com a lista original do site_mol
    RDKit.

    ligand_charge_method: "gas" (Gasteiger, padrão) ou "bcc" (AM1-BCC via
        sqm). Medido empiricamente: "gas" leva ~0.02-0.4s por ligante;
        "bcc" leva ~2.5s+ MESMO para uma molécula pequena (21 átomos) —
        e esse custo é do MÉTODO DE CARGA em si (também presente em
        SMIRNOFF/AM1BCC-ELF10, não é uma particularidade do GROMACS), não
        do subprocess. Para geração em lote de centenas de candidatos,
        "gas" é o padrão sensato — o ganho de rigor do AMBER/CHARMM aqui
        vem principalmente da PROTEÍNA (campo de força real), não da
        carga do ligante.

    Retorna (sucesso, site_coords_finais). DIFERENTE de
    _minimize_complex_rdkit aqui: site_coords_finais nunca é None em caso
    de sucesso, mesmo com induced_fit=False — o `gmx editconf -c` (passo 4)
    recentra o sistema inteiro num frame novo (translação rígida aplicada
    a TODOS os átomos, sítio e ligante, travados ou não), então o chamador
    SEMPRE precisa da geometria do sítio nesse novo frame para comparar
    contra a posição final do ligante (que também sai desse mesmo frame) —
    devolver None faria a checagem de colisão comparar coordenadas em dois
    frames diferentes, com um offset de dezenas de Å (bug encontrado
    empiricamente). Só None em falha.
    """
    if not (HAS_GROMACS and HAS_ACPYPE) or site_topology is None:
        return False, None
    gmx    = shutil.which("gmx")
    acpype = shutil.which("acpype")
    try:
        with tempfile.TemporaryDirectory(prefix="codeep_gmx_") as tmp_str:
            tmp = Path(tmp_str)

            # 1) Parametriza o ligante — acpype -> GAFF2 .itp/.gro prontos p/ GROMACS
            lig_pdb = tmp / "ligand.pdb"
            Chem.MolToPDBFile(ligand_mol, str(lig_pdb))
            input_names = [a.GetSymbol() for a in ligand_mol.GetAtoms()]  # validação de ordem abaixo

            charge = Chem.GetFormalCharge(ligand_mol)
            result = subprocess.run(
                [acpype, "-i", str(lig_pdb), "-c", ligand_charge_method,
                 "-n", str(charge), "-b", "LIG", "-o", "gmx"],
                capture_output=True, text=True, cwd=str(tmp), timeout=180,
            )
            lig_itp = tmp / "LIG.acpype" / "LIG_GMX.itp"
            lig_gro = tmp / "LIG.acpype" / "LIG_GMX.gro"
            if result.returncode != 0 or not lig_itp.exists() or not lig_gro.exists():
                log.debug(f"  acpype falhou (carga={ligand_charge_method}): "
                          f"{result.stderr[-800:]}")
                return False, None

            lig_gro_lines = lig_gro.read_text().splitlines()
            n_lig = int(lig_gro_lines[1].strip())
            # Validação de ordem: acpype passa o ligante por vários conversores
            # internamente (OpenBabel -> antechamber -> parmchk2 -> tleap) antes
            # de escrever o .gro — confirmado empiricamente que a ordem de
            # átomos de entrada é preservada, mas isso é checado aqui a cada
            # chamada em vez de assumido cegamente: um mapeamento posicional
            # errado corromperia geometria silenciosamente, sem erro nenhum.
            if n_lig != ligand_mol.GetNumAtoms():
                log.debug(f"  acpype: nº de átomos do ligante mudou "
                          f"({ligand_mol.GetNumAtoms()} -> {n_lig}) — abortando.")
                return False, None
            out_symbols = [line[10:15].strip()[0] for line in lig_gro_lines[2:2 + n_lig]]
            if out_symbols != input_names:
                log.debug("  acpype: ordem dos átomos do ligante mudou — abortando "
                          "(mapeamento de posições não seria confiável).")
                return False, None

            # 2) Une a topologia do ligante com a do sítio (cacheada)
            top_text = site_topology["top"].read_text()
            out_lines, inserted = [], False
            for line in top_text.split("\n"):
                out_lines.append(line)
                if not inserted and line.strip().startswith("#include") and "forcefield.itp" in line:
                    out_lines.append(f'#include "{lig_itp}"')
                    inserted = True
            merged_top = "\n".join(out_lines).rstrip() + "\nLIG                 1\n"
            complex_top = tmp / "complex.top"
            complex_top.write_text(merged_top)

            # 3) Une as coordenadas (site.gro + ligante.gro)
            site_lines = site_topology["gro"].read_text().splitlines()
            n_site = site_topology["n_atoms"]
            complex_gro_raw = tmp / "complex_raw.gro"
            complex_gro_raw.write_text("\n".join(
                ["complexo ligante+sitio", str(n_site + n_lig)]
                + site_lines[2:2 + n_site]
                + lig_gro_lines[2:2 + n_lig]
                + [site_lines[2 + n_site]]   # vetor de caixa (do sítio)
            ) + "\n")

            # 4) Caixa de vácuo com padding — sistema pequeno e isolado, sem
            # relevância física de periodicidade real
            complex_box = tmp / "complex_box.gro"
            subprocess.run(
                [gmx, "editconf", "-f", str(complex_gro_raw), "-o", str(complex_box),
                 "-c", "-d", "1.2", "-bt", "cubic"],
                capture_output=True, text=True, cwd=str(tmp), timeout=60,
            )
            if not complex_box.exists():
                return False, None

            # 5) Grupo de índice travado: tudo (rígido) ou só backbone (induced-fit).
            # Usa o backbone flag NA ORDEM DO PRÓPRIO site_topology (derivado dos
            # nomes de átomo do .gro que o pdb2gmx gerou, ver
            # _build_gromacs_site_topology) — NUNCA o site_is_backbone recebido
            # como parâmetro (ordem/índices do site_mol original do RDKit, que
            # -ignh invalida ao reconstruir os hidrogênios do zero). Se a
            # topologia (cache antigo) não tiver "is_backbone", trata tudo como
            # backbone — mais seguro (sítio rígido) do que travar/liberar o
            # átomo errado por um mapeamento de índice inválido.
            top_is_backbone = site_topology.get("is_backbone")
            if top_is_backbone is None or len(top_is_backbone) != n_site:
                log.debug("  site_topology sem 'is_backbone' compatível — "
                          "tratando sítio inteiro como backbone (rígido).")
                top_is_backbone = [True] * n_site
            frozen_idx = (
                [i + 1 for i, is_bb in enumerate(top_is_backbone) if is_bb]
                if induced_fit else list(range(1, n_site + 1))
            )
            ndx_lines = ["[ Frozen ]"]
            for i in range(0, len(frozen_idx), 15):
                ndx_lines.append(" ".join(str(x) for x in frozen_idx[i:i + 15]))
            (tmp / "index.ndx").write_text("\n".join(ndx_lines) + "\n")

            (tmp / "min.mdp").write_text(
                "integrator = steep\nnsteps = 500\nemtol = 100\nemstep = 0.01\n"
                "cutoff-scheme = Verlet\ncoulombtype = cutoff\nrcoulomb = 1.2\n"
                "rvdw = 1.2\npbc = xyz\nfreezegrps = Frozen\nfreezedim = Y Y Y\n"
            )

            tpr_path = tmp / "min.tpr"
            r = subprocess.run(
                [gmx, "grompp", "-f", str(tmp / "min.mdp"), "-c", str(complex_box),
                 "-p", str(complex_top), "-n", str(tmp / "index.ndx"),
                 "-o", str(tpr_path), "-maxwarn", "10"],
                capture_output=True, text=True, cwd=str(tmp), timeout=60,
            )
            if r.returncode != 0 or not tpr_path.exists():
                log.debug(f"  grompp falhou: {r.stderr[-800:]}")
                return False, None

            r = subprocess.run(
                [gmx, "mdrun", "-deffnm", "min", "-nt", "1", "-ntmpi", "1"],
                capture_output=True, text=True, cwd=str(tmp), timeout=120,
            )
            min_gro = tmp / "min.gro"
            if r.returncode != 0 or not min_gro.exists():
                log.debug(f"  mdrun falhou: {r.stderr[-800:]}")
                return False, None

            # 6) Extrai posições finais (nm -> Å)
            final_lines = min_gro.read_text().splitlines()

            def _coords(lines, start, n):
                return np.array([
                    [float(l[20:28]), float(l[28:36]), float(l[36:44])]
                    for l in lines[start:start + n]
                ], dtype=np.float32) * 10.0

            site_final = _coords(final_lines, 2, n_site)
            lig_final  = _coords(final_lines, 2 + n_site, n_lig)

            lig_conf = ligand_mol.GetConformer()
            for i in range(n_lig):
                lig_conf.SetAtomPosition(i, lig_final[i].tolist())

            # site_final é retornado SEMPRE (rígido ou induced-fit), não só
            # quando a cadeia lateral se move: o `gmx editconf -c` (passo 4)
            # recentra o sistema inteiro num frame novo, deslocando TODOS os
            # átomos (ligante incluso) por uma translação rígida — mesmo com
            # o sítio 100% travado, ele se move nesse frame junto com o
            # ligante. Devolver None no caso rígido faria o chamador comparar
            # o ligante (já na nova posição) contra o site_coords ORIGINAL
            # (frame antigo), produzindo colisões completamente erradas —
            # bug encontrado empiricamente (min_clash_dist de ~41 Å num
            # ligante posicionado propositalmente dentro do bolso).
            return True, site_final
    except Exception as e:
        log.debug(f"  Minimização GROMACS falhou: {type(e).__name__}: {e}")
        return False, None


def _check_pocket_clashes(mol: "Chem.Mol", site_coords: np.ndarray,
                           site_elements: list) -> dict:
    """Checagem de plausibilidade geométrica PÓS-minimização, estilo
    PoseBusters (Buttenschoen, Morris & Deane, 2024):

      - colisão estérica: algum par (átomo do ligante, átomo do sítio)
        mais próximo que CLASH_VDW_FACTOR (0.75) × soma dos raios de
        van der Waals?
      - comprimento de ligação: alguma ligação do ligante fora de
        [0.75, 1.25] × comprimento canônico do seu tipo (indício de
        geometria não relaxada corretamente / conformação implausível)?

    Não substitui uma checagem completa (planaridade de anel, ângulos de
    ligação, energia relativa a conformações de referência — ver
    PoseBusters para o conjunto completo); cobre os dois critérios mais
    diretamente relevantes para "esse candidato colide com a proteína".
    """
    result = {
        "has_clash": False, "n_clashes": 0, "min_clash_dist": None,
        "bond_length_ok": True, "max_bond_deviation": None,
    }
    if mol is None or mol.GetNumConformers() == 0:
        return result

    conf = mol.GetConformer()
    lig_pos = conf.GetPositions()
    lig_elems = [a.GetSymbol().upper() for a in mol.GetAtoms()]

    if site_coords is not None and len(site_coords) > 0:
        site_pos = np.asarray(site_coords, dtype=np.float32)
        site_elems_up = [e.upper() for e in site_elements]
        lig_radii  = np.array([VDW_RADII_CLASH.get(e, VDW_RADIUS_DEFAULT) for e in lig_elems])
        site_radii = np.array([VDW_RADII_CLASH.get(e, VDW_RADIUS_DEFAULT) for e in site_elems_up])

        diff = lig_pos[:, None, :] - site_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        threshold = CLASH_VDW_FACTOR * (lig_radii[:, None] + site_radii[None, :])
        clash_mask = dist < threshold

        result["n_clashes"] = int(clash_mask.sum())
        result["has_clash"] = result["n_clashes"] > 0
        if dist.size > 0:
            result["min_clash_dist"] = round(float(dist.min()), 3)

    max_dev = None
    bond_length_ok = True
    for bond in mol.GetBonds():
        canon = _CANONICAL_BOND_LENGTH.get(bond.GetBondType())
        if canon is None:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d = float(np.linalg.norm(lig_pos[i] - lig_pos[j]))
        ratio = d / canon
        dev = abs(ratio - 1.0)
        if max_dev is None or dev > max_dev:
            max_dev = dev
        if not (BOND_LENGTH_TOL_LOW <= ratio <= BOND_LENGTH_TOL_HIGH):
            bond_length_ok = False
    result["bond_length_ok"] = bond_length_ok
    result["max_bond_deviation"] = round(max_dev, 3) if max_dev is not None else None

    return result


def _relax_generated_geometry(
    mol: "Chem.Mol",
    site_coords: "np.ndarray | None" = None,
    site_elements: "list | None" = None,
    site_mol: "Chem.Mol | None" = None,
    site_is_backbone: "list | None" = None,
    final_ff: str = "mmff_uff",
    induced_fit: bool = False,
    gromacs_site_topology: "dict | None" = None,
    ligand_charge_method: str = "gas",
) -> "Chem.Mol":
    """Relaxa a geometria 3D do candidato gerado.

    Sem site_coords/site_elements (comportamento original): minimiza o
    ligante ISOLADO (MMFF94s, fallback UFF) — não sabe nada sobre a
    proteína.

    Com site_coords/site_elements/site_mol/site_is_backbone (ver
    pdb_to_site_tensor): minimiza o ligante DENTRO do campo de força do
    complexo (_minimize_complex_rdkit/_minimize_complex_gromacs), e anexa
    ao mol o resultado da checagem de colisão estérica pós-minimização
    (_check_pocket_clashes) como propriedades RDKit:
      mol.GetBoolProp("has_pocket_clash"), GetIntProp("n_pocket_clashes"),
      GetDoubleProp("min_clash_dist") (se houver), GetBoolProp("bond_length_ok").

    final_ff: "mmff_uff" (padrão, RDKit, sem dependência extra) |
              "amber_ff14sb" | "charmm36" (via GROMACS+AmberTools — cai
              para "mmff_uff" automaticamente se a pilha GROMACS/acpype
              não estiver instalada, ou se gromacs_site_topology não foi
              construída, com aviso no log).
    gromacs_site_topology: dict retornado por
              _build_gromacs_site_topology — construído UMA VEZ por sítio
              pelo chamador (generate_candidates), não aqui (custaria caro
              refazer a cada candidato). Necessário só quando
              final_ff é "amber_ff14sb"/"charmm36".
    ligand_charge_method: "gas" (Gasteiger, padrão, rápido) | "bcc"
              (AM1-BCC, muito mais lento — ver _minimize_complex_gromacs).
    induced_fit: False (padrão) — sítio 100% rígido durante a
              minimização. True — só o backbone (N,Cα,C,O) fica travado;
              a cadeia lateral se move livremente, ajustando-se ao
              ligante (ver _minimize_complex_rdkit/_gromacs). Requer
              site_mol/site_is_backbone; sem eles, cai silenciosamente
              para rígido.
    """
    if mol is None or mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return mol

    have_site = site_coords is not None and site_elements is not None and len(site_elements) > 0
    have_site_mol = site_mol is not None and site_is_backbone is not None

    if not have_site:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return mol

    if induced_fit and not have_site_mol:
        log.warning("  induced_fit=True requer site_mol/site_is_backbone "
                    "(indisponíveis — provavelmente protonate_site=False) "
                    "— usando sítio rígido.")
        induced_fit = False

    # effective_site_coords/effective_site_elements: começam como os dados
    # originais; se a minimização em modo induced-fit realmente mover a
    # cadeia lateral, são substituídos pela geometria PÓS-minimização — a
    # checagem de colisão (abaixo) tem que comparar contra a posição REAL
    # final do sítio, não contra a posição de antes do ajuste (bug
    # corrigido: sem isso, um candidato cuja cadeia lateral se afastou pra
    # acomodá-lo aparecia com colisão pior do que no modo rígido, porque
    # comparava contra onde a cadeia lateral estava ANTES de sair do
    # caminho). No caminho GROMACS, "updated" vem na ordem/contagem de
    # átomos do site_topology (pdb2gmx -ignh reconstrói os hidrogênios do
    # zero — conta e ordem diferem do site_mol original do RDKit), então
    # site_elements TEM que trocar junto (gromacs_site_topology["elements"]),
    # nunca ficar comparando coords novas contra elementos da lista antiga
    # (bug corrigido: causava ValueError de shape em _check_pocket_clashes,
    # engolido pelo try/except de preds_to_mol — candidato descartado
    # silenciosamente sempre que induced_fit=True tinha sucesso via GROMACS).
    effective_site_coords = site_coords
    effective_site_elements = site_elements
    minimized = False
    if final_ff in ("amber_ff14sb", "charmm36"):
        if HAS_GROMACS and HAS_ACPYPE and gromacs_site_topology is not None and have_site_mol:
            minimized, updated = _minimize_complex_gromacs(
                mol, gromacs_site_topology, induced_fit, ligand_charge_method,
            )
            if updated is not None:
                effective_site_coords = updated
                effective_site_elements = gromacs_site_topology.get(
                    "elements", site_elements
                )
        elif not (HAS_GROMACS and HAS_ACPYPE):
            log.warning(
                f"  final_ff={final_ff!r} requer GROMACS (gmx) + acpype "
                f"(AmberTools: antechamber/parmchk2/tleap no PATH) — "
                f"não encontrados. Usando RDKit MMFF/UFF (mmff_uff) como fallback."
            )
        elif gromacs_site_topology is None:
            log.warning(
                f"  final_ff={final_ff!r} requer a topologia GROMACS do sítio "
                f"(gromacs_site_topology) — não construída (ver "
                f"_build_gromacs_site_topology). Usando RDKit MMFF/UFF como fallback."
            )
    if not minimized and have_site_mol:
        try:
            combined, n_lig = _build_ligand_site_complex(mol, site_mol)
            minimized, updated = _minimize_complex_rdkit(
                combined, n_lig, site_is_backbone, induced_fit
            )
            if minimized:
                combined_conf = combined.GetConformer()
                mol_conf = mol.GetConformer()
                for i in range(n_lig):
                    p = combined_conf.GetAtomPosition(i)
                    mol_conf.SetAtomPosition(i, p)
                if updated is not None:
                    effective_site_coords = updated
        except Exception as e:
            log.debug(f"  Minimização conjunta RDKit falhou: {type(e).__name__}: {e}")
            minimized = False

    if not minimized:
        # Nenhum campo de força pôde ser construído para o complexo —
        # cai para a relaxação isolada (melhor que geometria não relaxada).
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    clash = _check_pocket_clashes(mol, effective_site_coords, effective_site_elements)
    mol.SetBoolProp("has_pocket_clash", bool(clash["has_clash"]))
    mol.SetIntProp("n_pocket_clashes", int(clash["n_clashes"]))
    mol.SetBoolProp("bond_length_ok", bool(clash["bond_length_ok"]))
    if clash["min_clash_dist"] is not None:
        mol.SetDoubleProp("min_clash_dist", float(clash["min_clash_dist"]))
    if clash["max_bond_deviation"] is not None:
        mol.SetDoubleProp("max_bond_deviation", float(clash["max_bond_deviation"]))

    return mol

def preds_to_mol(
    candidate: dict,
    cutoff_dist: float = 1.9,
    site_coords: "np.ndarray | None" = None,
    site_elements: "list | None" = None,
    site_mol: "Chem.Mol | None" = None,
    site_is_backbone: "list | None" = None,
    final_ff: str = "mmff_uff",
    induced_fit: bool = False,
    gromacs_site_topology: "dict | None" = None,
    ligand_charge_method: str = "gas",
) -> "Chem.Mol | None":
    """Converte a saída de LigandCVAE.sample() (um candidato) em molécula
    RDKit.

    candidate: {"atom_types": LongTensor(n,), "charges": FloatTensor(n,),
                "positions": FloatTensor(n,3), "bonds": list[(i,j,bond_type)]}
    — o decoder autorregressivo já decidiu quais ligações existem durante
    a própria geração (ver GNNDecoder/LigandCVAE.sample em
    pipeline_train.py); esta função só valida quimicamente essas decisões
    (valência, distância) e monta o objeto RDKit — não reconstrói mais a
    topologia a partir de um "template graph" externo.

    Sanidade aplicada nas ligações já decididas pelo modelo:
      - distância átomo-átomo compatível com o tipo de ligação
        (_bond_distance_ok)
      - orçamento de valência não excedido (MAX_VALENCE)
    Ligações que violam essas checagens são descartadas. Se restarem
    fragmentos desconexos, _connect_components tenta uma reparação mínima
    (liga o par mais próximo entre fragmentos, respeitando valência) antes
    de desistir do candidato.

    site_coords/site_elements/site_mol/site_is_backbone (opcionais, ver
    pdb_to_site_tensor): se fornecidos, a relaxação final de geometria
    (_relax_generated_geometry) passa a minimizar o ligante DENTRO do
    campo de força do complexo e anexa ao mol resultante o resultado da
    checagem de colisão estérica (ver _check_pocket_clashes) — sem eles,
    cai no comportamento anterior de relaxar o ligante isolado, sem
    qualquer noção do sítio. induced_fit controla se o sítio fica 100%
    rígido (padrão) ou só o backbone travado, cadeia lateral livre (ver
    _minimize_complex_rdkit/_minimize_complex_gromacs).
    """
    try:
        atom_types = candidate["atom_types"].cpu().numpy()
        positions  = candidate["positions"].detach().cpu().numpy()
        n_atoms    = len(atom_types)
        if n_atoms < 1:
            return None

        rwmol = Chem.RWMol()
        conf  = Chem.Conformer(n_atoms)
        for i, at_idx in enumerate(atom_types):
            sym = ATOM_TYPES_GEN[min(int(at_idx), len(ATOM_TYPES_GEN) - 1)]
            rwmol.AddAtom(Chem.Atom(sym))
            conf.SetAtomPosition(i, positions[i].tolist())
        rwmol.AddConformer(conf, assignId=True)

        valence_used = [0.0] * n_atoms
        added = set()
        for (i, j, type_idx) in candidate["bonds"]:
            i, j = int(i), int(j)
            if i == j or i >= n_atoms or j >= n_atoms or (i, j) in added or (j, i) in added:
                continue
            bond_type = BOND_TYPE_MAP.get(int(type_idx), BondType.SINGLE)
            sym_i = ATOM_TYPES_GEN[min(int(atom_types[i]), len(ATOM_TYPES_GEN) - 1)].upper()
            sym_j = ATOM_TYPES_GEN[min(int(atom_types[j]), len(ATOM_TYPES_GEN) - 1)].upper()
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if not _bond_distance_ok(sym_i, sym_j, bond_type, dist):
                continue
            budget = BOND_ORDER_BUDGET.get(bond_type, 1.0)
            if valence_used[i] + budget > MAX_VALENCE.get(sym_i, 4) + 1e-6:
                continue
            if valence_used[j] + budget > MAX_VALENCE.get(sym_j, 4) + 1e-6:
                continue
            rwmol.AddBond(i, j, bond_type)
            valence_used[i] += budget
            valence_used[j] += budget
            added.add((i, j))

        # Fallback por distância só se o modelo não decidiu NENHUMA ligação
        # válida (ex.: candidato com 1-2 átomos onde a cabeça de ligação
        # nunca disparou) — último recurso, igual à versão anterior.
        if not added and n_atoms > 1:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    d = float(np.linalg.norm(positions[i] - positions[j]))
                    if 1.0 < d < cutoff_dist:
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
        return _relax_generated_geometry(
            mol, site_coords=site_coords, site_elements=site_elements,
            site_mol=site_mol, site_is_backbone=site_is_backbone,
            final_ff=final_ff, induced_fit=induced_fit,
            gromacs_site_topology=gromacs_site_topology,
            ligand_charge_method=ligand_charge_method,
        )

    except Exception:
        return None


def generate_candidates(
    model,
    mod,
    site_tensor: np.ndarray,
    tensor_dir: Path,
    n_samples: int = 200,
    temperature: float = 1.0,
    voxel_size: float = 1.0,
    ref_mol = None,   # não usado mais aqui (era o template) — mantido por
                       # compatibilidade de assinatura; a similaridade de
                       # Tanimoto com ref_mol é calculada depois, em
                       # rank_candidates.
    max_atoms: int = 60,
    site_coords: "np.ndarray | None" = None,
    site_elements: "list | None" = None,
    site_mol: "Chem.Mol | None" = None,
    site_is_backbone: "list | None" = None,
    final_ff: str = "mmff_uff",
    induced_fit: bool = False,
    ligand_charge_method: str = "gas",
) -> list[dict]:
    """Gera N candidatos a partir do tensor do sítio — construção
    SEQUENCIAL autorregressiva (um átomo por vez), com TODOS os n_samples
    candidatos avançando em lockstep numa única chamada a model.sample()
    (não mais um template graph reclassificado, nem um loop Python
    chamando o decoder uma vez por amostra — ver LigandCVAE.sample em
    pipeline_train.py para a estratégia de paralelismo).

    Para cada amostra:
      1. Codifica o sítio → z_mean via encoder (uma vez, compartilhado)
      2. Constrói o grafo átomo-a-átomo condicionado em z = z_mean +
         temperatura × N(0,I), decidindo tipo/posição/ligações a cada
         passo (bond_exist agora com amostragem negativa real no treino —
         não é mais poda de um template fixo)
      3. Converte o resultado em molécula RDKit (preds_to_mol)
      4. Calcula métricas de drug-likeness

    Returns lista de dicts com mol, smiles e métricas.
    """
    arr    = np.transpose(site_tensor, (3, 0, 1, 2))         # (C,X,Y,Z)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    voxel_sizes_gen = torch.tensor([voxel_size], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        samples = model.sample(
            tensor, n_samples=n_samples, temperature=temperature,
            voxel_sizes=voxel_sizes_gen, max_atoms=max_atoms,
        )

    candidates = []
    failed     = 0
    n_clashing = 0

    # Topologia GROMACS do sítio — construída UMA VEZ para toda a rodada
    # (o sítio não muda entre os N candidatos, só o ligante varia a cada
    # chamada de preds_to_mol/_minimize_complex_gromacs). Refazer isso por
    # candidato custaria uma chamada a pdb2gmx redundante N vezes à toa.
    gromacs_topology_dir = None
    gromacs_site_topology = None
    if final_ff in ("amber_ff14sb", "charmm36") and site_mol is not None:
        if HAS_GROMACS and HAS_ACPYPE:
            gromacs_topology_dir = tempfile.TemporaryDirectory(prefix="codeep_gmx_site_")
            gromacs_site_topology = _build_gromacs_site_topology(
                site_mol, final_ff, Path(gromacs_topology_dir.name)
            )
            if gromacs_site_topology is None:
                log.warning(
                    "  Não foi possível construir a topologia GROMACS do sítio "
                    f"({final_ff}) — candidatos cairão para RDKit MMFF/UFF."
                )
        else:
            log.warning(
                f"  final_ff={final_ff!r} requer GROMACS (gmx) + acpype "
                "(AmberTools no PATH) — não encontrados. Candidatos usarão "
                "RDKit MMFF/UFF (mmff_uff) como fallback."
            )

    try:
        for i, candidate in enumerate(tqdm(samples, desc="  Montando candidatos", leave=False)):
            try:
                mol = preds_to_mol(
                    candidate, site_coords=site_coords, site_elements=site_elements,
                    site_mol=site_mol, site_is_backbone=site_is_backbone,
                    final_ff=final_ff, induced_fit=induced_fit,
                    gromacs_site_topology=gromacs_site_topology,
                    ligand_charge_method=ligand_charge_method,
                )
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

                # Resultado da checagem de colisão estérica pós-minimização
                # (anexado ao mol como propriedade RDKit por _relax_generated_
                # geometry — ver _check_pocket_clashes). Ausente (None) quando
                # site_coords não foi passado (comportamento sem sítio).
                has_clash = mol.GetBoolProp("has_pocket_clash") if mol.HasProp("has_pocket_clash") else None
                n_clashes = mol.GetIntProp("n_pocket_clashes") if mol.HasProp("n_pocket_clashes") else None
                if has_clash:
                    n_clashing += 1

                candidates.append({
                    "idx":    i,
                    "mol":    mol,
                    "smiles": smiles,
                    "has_pocket_clash": has_clash,
                    "n_pocket_clashes": n_clashes,
                    **metrics,
                })

            except Exception:
                failed += 1
                continue
    finally:
        if gromacs_topology_dir is not None:
            gromacs_topology_dir.cleanup()

    clash_info = (f" | {n_clashing} com colisão estérica no sítio"
                  if site_coords is not None else "")
    log.info(f"  Gerados: {len(candidates)} válidos | {failed} inválidos "
             f"de {n_samples} amostras "
             f"({100*len(candidates)/max(n_samples,1):.1f}% válidos){clash_info}")
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

        # Colisão estérica com o sítio (ver _check_pocket_clashes) — None
        # quando site_coords não foi passado para generate_candidates
        # (sem informação de sítio, não penaliza).
        has_clash = c.get("has_pocket_clash")
        clash_ok  = True if has_clash is None else not has_clash

        # Score composto (0-1)
        sa_norm = (10 - c["sa_score"]) / 9 if c.get("sa_score") else 0.5
        score   = (0.4 * c["qed"]
                 + 0.2 * sa_norm
                 + 0.15 * float(c["lipinski"])
                 + 0.10 * float(c["veber"])
                 + 0.05 * float(mw_ok and logp_ok and tpsa_ok and rot_ok and clash_ok)
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
            "has_pocket_clash": has_clash,
            "n_pocket_clashes": c.get("n_pocket_clashes"),
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
    parser.add_argument("--final-ff",    default="mmff_uff", choices=list(FINAL_FF_METHODS),
                        help="Campo de força para a minimização final do complexo "
                             "ligante+sítio (padrão: mmff_uff — RDKit, sem dependência "
                             "extra). amber_ff14sb/charmm36 requerem GROMACS (gmx) + "
                             "acpype + AmberTools (antechamber/parmchk2/tleap) no PATH "
                             "— caem automaticamente para mmff_uff se ausentes.")
    parser.add_argument("--induced-fit", action="store_true",
                        help="Trava só o backbone do sítio (N,Cα,C,O) na minimização "
                             "final, deixando a cadeia lateral livre para se ajustar "
                             "ao ligante. Padrão: sítio 100%% rígido.")
    parser.add_argument("--ligand-charge-method", default="gas", choices=["gas", "bcc"],
                        help="Método de carga parcial do ligante no caminho GROMACS "
                             "(padrão: gas — Gasteiger, rápido. 'bcc' = AM1-BCC via "
                             "sqm, muito mais lento — ~2.5s+ por ligante mesmo para "
                             "moléculas pequenas — mas mais rigoroso).")
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

    # ── 1. Carrega modelo (ANTES do sítio — precisa da voxel_config gravada
    # no checkpoint, ver load_model/resolve_site_config_overrides, para
    # revoxelizar o sítio novo com a MESMA config usada no treino) ────────────
    log.info("\n[1/5] Carregando modelo CVAE...")
    model, mod, ckpt_voxel_config = load_model(ckpt_dir)

    ckpt_info = torch.load(
        ckpt_dir / "best_model.pt",
        map_location="cpu", weights_only=False
    )
    ckpt_meta = {
        "epoch":      ckpt_info.get("epoch", "?"),
        "val_loss":   ckpt_info.get("val_loss", 0),
        "latent_dim": ckpt_info.get("config", {}).get("latent_dim", 128),
    }

    # ── 2. Extrai e voxeliza o sítio ──────────────────────────────────────────
    log.info("\n[2/5] Extraindo sítio de ligação...")
    if site_pdb_path:
        if not site_pdb_path.exists():
            log.error(f"--site-pdb não encontrado: {site_pdb_path}")
            sys.exit(1)
        log.info(f"  Modo: sítio fornecido ({site_pdb_path.name})")
    else:
        log.info("  Modo: detecção automática via ligante co-cristalizado")

    (site_tensor, origin, voxel_size, site_coords, site_elements,
     site_mol, site_is_backbone) = pdb_to_site_tensor(
        pdb_path, args.cutoff, args.min_atoms,
        site_pdb=site_pdb_path,
        config_overrides=resolve_site_config_overrides(ckpt_voxel_config),
    )
    np.savez_compressed(
        out_dir / "site_tensor.npz",
        tensor=site_tensor, origin=origin,
        voxel_size=np.array([voxel_size]),
    )

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
        site_coords=site_coords, site_elements=site_elements,
        site_mol=site_mol, site_is_backbone=site_is_backbone,
        final_ff=args.final_ff, induced_fit=args.induced_fit,
        ligand_charge_method=args.ligand_charge_method,
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
