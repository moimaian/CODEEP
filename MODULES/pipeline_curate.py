"""
Pipeline - Etapa 2: Validação e Curadoria da Base de Dados
==========================================================
Este script lê os arquivos PDB baixados, valida a qualidade das estruturas,
extrai informações sobre os sítios de ligação e gera um dataset curado
pronto para uso no pipeline de visão computacional.

Requisitos:
    pip install biopython rdkit pandas tqdm numpy

Uso:
    python 02_validate_and_curate.py

Saída:
    data/
    ├── curated/
    │   ├── {pdb_id}/
    │   │   ├── receptor.pdb   ← só a proteína, sem ligante e sem água
    │   │   ├── ligand.sdf     ← ligante extraído na posição do cristal
    │   │   └── binding_site.pdb ← resíduos do sítio (distância ≤ 6Å)
    ├── curated_metadata.csv
    └── quality_report.csv
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP

warnings.filterwarnings("ignore")

# ── Configuração ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR     = PROJECT_ROOT / "data"
RAW_PDB_DIR  = BASE_DIR / "raw" / "pdb"
CUR_DIR      = BASE_DIR / "curated"
LOG_DIR      = BASE_DIR / "logs"
CUR_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "curate.log", encoding="utf-8"),
        logging.StreamHandler(stream=None),
    ],
)
log = logging.getLogger(__name__)

CURATE_CONFIG = {
    # Distância máxima (Å) para considerar resíduo parte do sítio de ligação
    "binding_site_cutoff_A": 6.0,

    # Número mínimo de resíduos no sítio para aceitar a estrutura
    "min_site_residues": 5,

    # Peso molecular máximo do ligante (Da)
    "max_ligand_mw": 800.0,

    # Número mínimo de átomos pesados (não-H) no arquivo PDB do ligante.
    # Deve ser consistente com min_heavy_atoms do script 01.
    "min_heavy_atoms": 7,

    # Solventes/íons residuais que podem aparecer como HETATM no PDB
    # mesmo após filtragem pelo script 01 (ex: íons de empacotamento cristalino)
    "excluded_ligands": {
        "HOH", "DOD", "GOL", "EDO", "PEG", "DMS", "MPD",
        "BME", "DTT", "EPE", "MES", "TRS", "FMT", "ACT", "ACE",
        "NA", "K", "CA", "MG", "ZN", "FE", "MN", "CU", "CO",
        "NI", "CD", "HG", "PB", "CL", "BR", "IOD", "F",
        "PO4", "SO4", "NO3", "NO2",
    },
}

# ── Seletores BioPython ────────────────────────────────────────────────────────

class ProteinSelect(Select):
    """Seleciona apenas resíduos aminoácidos padrão (exclui ligante e água)."""
    def accept_residue(self, residue):
        return residue.get_id()[0] == " "  # hetfield vazio = aminoácido


class LigandSelect(Select):
    """Seleciona exatamente uma cópia do ligante: mesmo nome, cadeia e resseq."""
    def __init__(self, ligand_name: str, chain_id: str, resseq: int):
        self.ligand_name = ligand_name
        self.chain_id    = chain_id
        self.resseq      = resseq

    def accept_residue(self, residue):
        hetfield = residue.get_id()[0]
        return (
            hetfield.startswith("H_")
            and residue.get_resname().strip() == self.ligand_name
            and residue.get_parent().get_id()  == self.chain_id
            and residue.get_id()[1]            == self.resseq
        )

class BindingSiteSelect(Select):
    """Seleciona resíduos proteicos dentro de um raio do ligante.

    site_residue_ids contém pares (chain_id, res_id) — NUNCA res_id sozinho:
    numeração de resíduo se repete entre cadeias (ex.: toda cadeia de um
    complexo com N cópias biológicas tendo resíduos 1..300), então um
    conjunto sem a cadeia casaria residuos de cadeias inteiramente
    diferentes que só compartilham o mesmo número — ver get_binding_site.
    """
    def __init__(self, site_residue_ids: set):
        self.site_residue_ids = site_residue_ids

    def accept_residue(self, residue):
        chain_id = residue.get_parent().get_id()
        return (chain_id, residue.get_id()) in self.site_residue_ids


# ── Funções de análise ─────────────────────────────────────────────────────────

def _infer_element(atom) -> str:
    """Infere o símbolo do elemento de um átomo BioPython.

    O campo atom.element está frequentemente vazio em arquivos PDB do RCSB.
    Neste caso, inferimos o elemento a partir do nome do átomo (atom.name):
      - Colunas 13-14 do PDB contêm o símbolo do elemento para átomos pesados
      - Nomes como "C1", "CA", "CB"  → carbono
      - Nomes como "N1", "NZ"        → nitrogênio
      - Nomes como "O1", "OXT"       → oxigênio
      - Nomes como "FE", "FE2"       → ferro
      etc.
    """
    # Primeiro tenta o campo element (confiável quando presente)
    elem = (atom.element or "").strip().upper()
    if elem and elem not in ("", "X"):
        return elem

    # Infere do nome do átomo
    name = atom.get_name().strip().upper()
    if not name:
        return "X"

    # Remove dígitos do final e espaços: "C1'" → "C", "FE2" → "FE"
    import re
    # Tenta primeiros 2 caracteres como elemento (ex: FE, CL, BR, MG)
    two = re.sub(r"[^A-Z]", "", name)[:2]
    one = two[:1]

    # Elementos de dois caracteres comuns em ligantes/proteínas
    # ATENÇÃO: "CA" é ambíguo — pode ser alfa-carbono (C) ou íon cálcio (CA).
    # Em nomes de átomo PDB, CA sem dígitos seguintes é quase sempre alfa-C.
    # Íon cálcio aparece como resíduo "CA" ou como átomo único sem contexto.
    # Aqui tratamos CA no nome do átomo como carbono para ser conservador.
    two_char = {"FE","CL","BR","MG","ZN","MN","CU","CO","NI",
                "CD","HG","PB","SE","SI","NA","AL","CR","MO","RU",
                "RH","PD","AG","PT","AU","SN","AS","TE","AT"}
    # CA como nome de átomo = alfa-carbono (C), não cálcio
    if two == "CA" and name.startswith("CA"):
        return "C"
    if two in two_char:
        return two

    # Elementos de um caractere
    if one in ("C","N","O","S","P","F","H","I","K","V","W","U","B"):
        return one

    # Fallback: primeiro caractere alfabético
    alpha = re.sub(r"[^A-Z]", "", name)
    return alpha[0] if alpha else "X"


def get_ligands(structure, config: dict) -> list[dict]:
    """Retorna ligantes orgânicos válidos presentes na estrutura PDB.

    Filtros aplicados diretamente sobre os átomos do arquivo PDB:
    1. Deve ser HETATM (hetfield começa com "H_")
    2. Não pode estar na excluded_ligands
    3. Deve ter >= min_heavy_atoms átomos pesados (não-H)
    4. Deve conter pelo menos 1 átomo de carbono
       → rejeita íons isolados (FE2, CUA, ZN2) e complexos puramente
         inorgânicos mesmo que tenham vários átomos (ex: clusters de enxofre
         sem carbono)
       → aceita metaloporfirinas e complexos organometálicos porque têm C

    NOTA: o campo atom.element frequentemente vem vazio em arquivos PDB do
    RCSB. Por isso usamos _infer_element() que combina o campo element com
    inferência a partir do nome do átomo, garantindo detecção correta de
    carbonos mesmo quando o campo está ausente.
    """
    ligands = []
    excluded  = config["excluded_ligands"]
    min_heavy = config["min_heavy_atoms"]

    for chain in structure[0]:
        for residue in chain:
            hetfield = residue.get_id()[0]
            resname  = residue.get_resname().strip()

            if not hetfield.startswith("H_"):
                continue
            if resname in excluded:
                log.debug(f"  {resname}: excluído pela excluded_ligands")
                continue

            atoms = list(residue.get_atoms())

            # Infere elemento de cada átomo (robusto a campos element vazios)
            elem_list  = [_infer_element(a) for a in atoms]
            heavy_atoms = [(a, e) for a, e in zip(atoms, elem_list)
                            if e not in ("H", "D", "X", "")]
            n_heavy    = len(heavy_atoms)
            has_carbon = any(e == "C" for _, e in heavy_atoms)

            if n_heavy < min_heavy:
                log.debug(f"  {resname}: rejeitado — n_heavy={n_heavy} < {min_heavy}")
                continue

            if not has_carbon:
                log.debug(f"  {resname}: rejeitado — sem carbono "
                            f"(elementos: {set(e for _,e in heavy_atoms)})")
                continue

            ligands.append({
                "chain":      chain.get_id(),
                "resname":    resname,
                "resseq":     residue.get_id()[1],
                "n_atoms":    len(atoms),
                "n_heavy":    n_heavy,
                "has_carbon": has_carbon,
                "residue":    residue,
            })

    return ligands


def get_binding_site(structure, ligand_residue, cutoff: float) -> set:
    """Retorna (chain_id, res_id) dos resíduos proteicos dentro de
    `cutoff` Å do ligante — sempre restrito ao modelo 0 e à cadeia de cada
    resíduo (ver BindingSiteSelect: numeração de resíduo se repete entre
    cadeias, então retornar só res_id casaria resíduos de cadeias
    inteiramente diferentes que só compartilham o número, o que já causou
    binding_site.pdb com dezenas de milhares de átomos em complexos
    multi-cadeia — ver BENCHMARK_CROSSDOCKED2020.md).

    Usa cKDTree para busca espacial O(n log n) — ~50-100x mais rápido que
    loops Python para proteínas grandes (>3.000 átomos).
    """
    lig_coords = np.array([a.get_vector().get_array()
                           for a in ligand_residue.get_atoms()],
                          dtype=np.float32)
    if len(lig_coords) == 0:
        return set()

    # Coleta todos os átomos proteicos com seus IDs (cadeia, resíduo) —
    # sempre só do modelo 0 (estruturas de RMN têm vários MODELOS
    # representando o mesmo conjunto de átomos em conformações diferentes;
    # usar todos multiplicaria o sítio por N sem nenhum ganho).
    prot_coords  = []
    prot_res_ids = []
    model0 = structure[0]
    for chain in model0:
        chain_id = chain.get_id()
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue
            res_id = (chain_id, residue.get_id())
            for atom in residue.get_atoms():
                prot_coords.append(atom.get_vector().get_array())
                prot_res_ids.append(res_id)

    if not prot_coords:
        return set()

    prot_arr = np.array(prot_coords, dtype=np.float32)
    tree     = cKDTree(prot_arr)
    indices  = tree.query_ball_point(lig_coords, r=cutoff)

    site_residue_ids = set()
    for idx_list in indices:
        for idx in idx_list:
            site_residue_ids.add(prot_res_ids[idx])

    return site_residue_ids


def compute_ligand_properties(residue) -> dict:
    """Calcula propriedades básicas do ligante a partir dos átomos PDB."""
    atoms = list(residue.get_atoms())

    # Peso atômico aproximado por elemento
    atomic_weights = {
        "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065,
        "P": 30.974, "F": 18.998, "CL": 35.453, "BR": 79.904,
        "I": 126.90, "H": 1.008,
    }
    mw = sum(atomic_weights.get(a.element.upper() if a.element else "C", 12.0)
             for a in atoms)

    coords = np.array([a.get_vector().get_array() for a in atoms])
    centroid = coords.mean(axis=0)

    return {
        "n_heavy_atoms": len(atoms),
        "approx_mw": round(mw, 2),
        "centroid_x": round(centroid[0], 3),
        "centroid_y": round(centroid[1], 3),
        "centroid_z": round(centroid[2], 3),
    }


# ── Funções de salvamento ──────────────────────────────────────────────────────

def save_pdb(structure, selector, out_path: Path):
    """Salva uma estrutura ou seleção em formato PDB.

    Restrito ao modelo 0: estruturas de RMN trazem vários MODELOS (mesmos
    átomos, conformações diferentes) e o BioPython PDBIO grava todos por
    padrão se receber a Structure inteira — sem essa restrição,
    receptor.pdb/ligand.pdb/binding_site.pdb saem com N cópias
    concatenadas (uma por modelo), o que já causou sítios com dezenas de
    milhares de átomos e estourou memória na voxelização. Aceita tanto
    Structure (tem .get_models(), pega o modelo 0) quanto um Model já
    isolado (chamador já restringiu — usa como está).
    """
    model0 = structure[0] if hasattr(structure, "get_models") else structure
    io = PDBIO()
    io.set_structure(model0)
    io.save(str(out_path), selector)



# Código de 1 letra para aminoácidos padrão
_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","HSD":"H","HSE":"H","HSP":"H","SEC":"C","PYL":"K",
}


def site_fingerprint(structure, site_residue_ids: set) -> str:
    """Gera uma assinatura do sítio de ligação baseada na sequência de resíduos.

    A assinatura é a sequência de aminoácidos do sítio em ordem canônica
    (por número de resíduo) — não leva a cadeia em conta NA STRING final
    (só resname+resseq), então sítios equivalentes em cadeias simétricas
    ainda produzem a mesma assinatura e continuam sendo deduplicados pelo
    chamador (curate_structure: cada cópia simétrica do ligante gera seu
    próprio site_residue_ids, corretamente restrito à SUA cadeia por
    get_binding_site, mas o texto resultante coincide entre cópias
    equivalentes). site_residue_ids em si é (chain_id, res_id) — ver
    get_binding_site/BindingSiteSelect — daí o `residue.get_parent()`
    abaixo para casar corretamente.

    Exemplo: sítio com ALA-5, GLY-8, PHE-12 → "AGF"
    """
    residues = []
    model0 = structure[0]
    for chain in model0:
        chain_id = chain.get_id()
        for residue in chain:
            if (chain_id, residue.get_id()) in site_residue_ids:
                resname = residue.get_resname().strip()
                aa1     = _AA1.get(resname, "X")
                resseq  = residue.get_id()[1]
                residues.append((resseq, aa1))

    # Ordena por número de sequência e concatena
    residues.sort(key=lambda x: x[0])
    return "".join(aa for _, aa in residues)


# ── Utilitários de paralelismo ─────────────────────────────────────────────────

def detect_optimal_workers() -> int:
    """Detecta o número ótimo de workers com base nos CPUs e RAM disponíveis.

    Estratégia:
      - Usa todos os CPUs lógicos menos 1 (reserva 1 para o SO).
      - Limita pelo número de workers que cabem na RAM disponível,
        assumindo ~200 MB por worker (estimativa conservadora para
        processar estruturas PDB grandes com BioPython/NumPy).
      - Garante pelo menos 1 worker.
    """
    import os
    cpu_count = os.cpu_count() or 1
    workers_by_cpu = max(1, cpu_count - 1)

    try:
        import shutil
        # shutil.disk_usage não serve aqui; tenta psutil se disponível
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
        workers_by_ram = max(1, int(available_mb // 200))
    except ImportError:
        # psutil não instalado — usa apenas CPU como critério
        workers_by_ram = workers_by_cpu

    optimal = min(workers_by_cpu, workers_by_ram)
    return optimal


# ── Pipeline principal ─────────────────────────────────────────────────────────

def process_entry(
    pdb_path: Path,
    config: dict,
    output_curated_dir: "Path | str",
) -> tuple[list[dict], int]:
    """Processa uma única entrada PDB e salva os arquivos curados.

    Recebe o Path completo do arquivo para suportar PDBs em subdiretórios
    (phase2/, phase3/, pdbbind_opt/, etc.).
    Retorna:
      - lista de pares realmente processados nesta execução
      - número de pares já existentes que foram pulados
    """
    output_curated_dir = Path(output_curated_dir)

    pdb_id = pdb_path.stem
    if not pdb_path.exists():
        log.debug(f"{pdb_id}: arquivo PDB não encontrado, pulando.")
        return [], 0

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        log.warning(f"{pdb_id}: erro ao parsear PDB — {e}")
        return [], 0

    ligands = get_ligands(structure, config)
    if not ligands:
        log.debug(f"{pdb_id}: nenhum ligante válido encontrado.")
        return [], 0

    records = []
    seen_fingerprints: set[str] = set()
    n_duplicates = 0
    n_skipped_existing = 0

    for lig in ligands:
        lig_props = compute_ligand_properties(lig["residue"])

        if lig_props["approx_mw"] > config["max_ligand_mw"]:
            log.debug(
                f"{pdb_id}/{lig['resname']}: ligante muito pesado "
                f"({lig_props['approx_mw']:.0f} Da), pulando."
            )
            continue

        site_ids = get_binding_site(
            structure,
            lig["residue"],
            config["binding_site_cutoff_A"],
        )
        if len(site_ids) < config["min_site_residues"]:
            log.debug(
                f"{pdb_id}/{lig['resname']}: sítio muito pequeno "
                f"({len(site_ids)} resíduos), pulando."
            )
            continue

        fp = lig["resname"] + ":" + site_fingerprint(structure, site_ids)

        if fp in seen_fingerprints:
            n_duplicates += 1
            log.debug(
                f"{pdb_id}/{lig['resname']} cadeia {lig['chain']}: "
                "sítio equivalente já registrado — pulando duplicata."
            )
            continue
        seen_fingerprints.add(fp)

        pair_id = f"{pdb_id}_{lig['resname']}_{lig['chain']}{lig['resseq']}"
        out_dir = output_curated_dir / pair_id

        receptor_file = out_dir / "receptor.pdb"
        ligand_file = out_dir / "ligand.pdb"
        binding_site_file = out_dir / "binding_site.pdb"
        pair_info_file = out_dir / "pair_info.json"

        if (
            receptor_file.exists()
            and ligand_file.exists()
            and binding_site_file.exists()
            and pair_info_file.exists()
        ):
            try:
                with open(pair_info_file, "r") as f:
                    pair_meta = json.load(f)

                records.append(pair_meta)
                n_skipped_existing += 1
                log.debug(f"{pair_id}: já curado, metadata reaproveitada, pulando.")
                continue

            except Exception as e:
                log.warning(
                    f"{pair_id}: arquivos curados existem, mas não foi possível "
                    f"ler {pair_info_file.name} ({e}). Reprocessando."
                )

        out_dir.mkdir(parents=True, exist_ok=True)

        save_pdb(structure, ProteinSelect(), receptor_file)
        save_pdb(structure, LigandSelect(lig["resname"], lig["chain"], lig["resseq"]), ligand_file,)
        save_pdb(structure, BindingSiteSelect(site_ids), binding_site_file)

        pair_meta = {
            "pair_id":               pair_id,
            "pdb_id":                pdb_id,
            "ligand_name":           lig["resname"],
            "ligand_chain":          lig["chain"],
            "ligand_resseq":         lig["resseq"],
            "binding_site_residues": len(site_ids),
            "site_fingerprint":      fp,
            **lig_props,
        }
        with open(out_dir / "pair_info.json", "w") as f:
            json.dump(pair_meta, f, indent=2)

        records.append(pair_meta)
        log.debug(
            f"✓ {pair_id}: sítio={len(site_ids)} res., "
            f"MW={lig_props['approx_mw']:.0f}"
        )

    if n_duplicates:
        log.debug(f"{pdb_id}: {n_duplicates} sítio(s) equivalente(s) removido(s)")

    return records, n_skipped_existing


def _process_entry_worker(args):
    """Wrapper para uso com executores paralelos."""
    pdb_path, config, output_curated_dir = args
    try:
        return process_entry(pdb_path, config, output_curated_dir)
    except Exception as e:
        log.error(f"{pdb_path.stem}: erro — {e}")
        return [], 0


def run_curation(
    config: dict,
    input_pdb_dir: "Path | str",
    output_curated_dir: "Path | str",
    output_metadata_csv: "Path | str",
    output_report_csv: "Path | str",
    workers: int | None = 4,
    executor_mode: str = "thread",
):
    input_pdb_dir = Path(input_pdb_dir)
    output_curated_dir = Path(output_curated_dir)
    output_metadata_csv = Path(output_metadata_csv)
    output_report_csv = Path(output_report_csv)

    output_curated_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report_csv.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PIPELINE: Validação e Curadoria")
    log.info("=" * 60)
    log.info(f"Input PDB dir        : {input_pdb_dir}")
    log.info(f"Output curated dir   : {output_curated_dir}")
    log.info(f"Metadata CSV         : {output_metadata_csv}")
    log.info(f"Quality report CSV   : {output_report_csv}")

    if not input_pdb_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_pdb_dir}")

    pdb_files = sorted(input_pdb_dir.rglob("*.pdb"))

    sources: dict[str, int] = {}
    for p in pdb_files:
        try:
            rel = p.relative_to(input_pdb_dir)
            src = rel.parts[0] if len(rel.parts) > 1 else input_pdb_dir.name
        except ValueError:
            src = p.parent.name
        sources[src] = sources.get(src, 0) + 1

    log.info(f"PDB files found: {len(pdb_files)} in {len(sources)} sources")
    for src, n in sorted(sources.items()):
        log.info(f"  {src:<30}: {n:>6} PDBs")

    if not pdb_files:
        log.warning("Nenhum arquivo PDB encontrado para curadoria.")
        df = pd.DataFrame()
        df.to_csv(output_metadata_csv, index=False)
        pd.DataFrame([{
            "total_pdb_files": 0,
            "total_valid_pairs": 0,
            "skipped_pairs": 0,
            "unique_pdbs_with_pairs": 0,
            "unique_ligands": 0,
            "unique_site_fingerprints": 0,
            "failed_entries": 0,
        }]).to_csv(output_report_csv, index=False)
        return

    if workers is None:
        workers = detect_optimal_workers()
        log.info(f"Workers: {workers} (auto-detected)")
    else:
        log.info(f"Workers: {workers} (user-defined)")

    executor_mode = (executor_mode or "thread").strip().lower()
    if executor_mode not in {"thread", "process"}:
        raise ValueError(f"executor_mode inválido: {executor_mode}")

    if executor_mode == "process":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        executor_cls = ProcessPoolExecutor
        log.info("Executor mode: ProcessPoolExecutor")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        executor_cls = ThreadPoolExecutor
        log.info("Executor mode: ThreadPoolExecutor")

    all_records = []
    failed = []
    skipped_pairs = 0

    args_list = [(p, config, output_curated_dir) for p in pdb_files]

    with executor_cls(max_workers=workers) as executor:
        futures = {executor.submit(_process_entry_worker, a): a[0] for a in args_list}
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="Curating entries"):
            pdb_path = futures[future]
            try:
                records, skipped_count = future.result()
                all_records.extend(records)
                skipped_pairs += skipped_count
            except Exception as e:
                log.error(f"{pdb_path.stem}: unexpected error — {e}")
                failed.append(pdb_path.stem)

    df = pd.DataFrame(all_records)
    df.to_csv(output_metadata_csv, index=False)

    n_unique_fps = (
        df["site_fingerprint"].nunique()
        if len(df) and "site_fingerprint" in df.columns
        else 0
    )

    report = {
        "total_pdb_files":          len(pdb_files),
        "total_valid_pairs":        len(df),
        "skipped_pairs":            skipped_pairs,
        "unique_pdbs_with_pairs":   df["pdb_id"].nunique() if len(df) else 0,
        "unique_ligands":           df["ligand_name"].nunique() if len(df) else 0,
        "unique_site_fingerprints": n_unique_fps,
        "failed_entries":           len(failed),
    }
    pd.DataFrame([report]).to_csv(output_report_csv, index=False)

    log.info("\n" + "=" * 60)
    log.info("CURATION SUMMARY")
    log.info("=" * 60)
    for k, v in report.items():
        log.info(f"  {k:<35}: {v}")
    log.info(f"\nCurated pairs saved to: {output_curated_dir.resolve()}")
    log.info(f"Metadata saved to     : {output_metadata_csv.resolve()}")
    log.info(f"Report saved to       : {output_report_csv.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdb-dir", required=True,
                        help="Diretório de entrada com PDBs a curar")
    parser.add_argument("--output-curated-dir", required=True,
                        help="Diretório de saída com os pares curados")
    parser.add_argument("--output-metadata-csv", required=True,
                        help="CSV de metadados da curadoria")
    parser.add_argument("--output-report-csv", required=True,
                        help="CSV com relatório resumido de qualidade")
    parser.add_argument("--workers", type=int, default=None,
                        help="Número de workers; se omitido, detecta automaticamente")
    parser.add_argument(
        "--executor-mode",
        choices=["thread", "process"],
        default="thread",
        help="Tipo de executor paralelo a usar",
    )
    args = parser.parse_args()

    run_curation(
        config=CURATE_CONFIG,
        input_pdb_dir=args.input_pdb_dir,
        output_curated_dir=args.output_curated_dir,
        output_metadata_csv=args.output_metadata_csv,
        output_report_csv=args.output_report_csv,
        workers=args.workers,
        executor_mode=args.executor_mode,
    )
