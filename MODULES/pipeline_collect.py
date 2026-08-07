"""
Pipeline - Etapa 1b: Coleta Expandida para Pesquisa Orientada a Alvos
=====================================================================
Estratégia em três fases para construir um dataset robusto de interações
proteína-ligante para qualquer alvo definido pelo usuário.

DATASETS EXTERNOS DISPONÍVEIS (Fase 1 — pré-treino geral):
  lp-pdbbind   LP-PDBBind via GitHub (~17.000 pares curados, sem data leakage)
  rcsb-wide   Busca ampla no RCSB (complexos proteína-ligante com resolução <= 2,5 Å)  
  plinder      PLINDER via pip (~449.000 pares, maior dataset disponível)
  moad         Binding MOAD via bindingmoad.org (~41.000 pares, requer registro)

FASE 2 — Especialização por família / organismo:
  Busca configurável no PDB usando termos de organismo e/ou descrição
  proteica definidos pelo usuário.

FASE 3 — Especialização em alvos específicos:
  Coleta direcionada por lista explícita de PDB IDs e/ou metadados
  fornecidos pelo usuário.

Uso:
    # Fase 1 — escolha o dataset externo:    
    python pipeline_collect.py --dataset lp-pdbbind
    python pipeline_collect.py --dataset rcsb-wide    
    python pipeline_collect.py --dataset plinder
    python pipeline_collect.py --dataset moad

    # Fase 2:
    python pipeline_collect.py --phase 2 \
        --phase2-organism "Homo sapiens" \
        --phase2-description "protein kinase"

    # Fase 3:
    python pipeline_collect.py --phase 3 \
        --phase3-pdb-ids 1ATP 2SRC 4HJB

    # Tudo de uma vez:
    python pipeline_collect.py --dataset lp-pdbbind --phase all \
        --phase2-organism "Homo sapiens" \
        --phase2-description "protein kinase" \
        --phase3-pdb-ids 1ATP 2SRC 4HJB

    # Verificar status dos dados:
    python pipeline_collect.py --status
"""

import os
import json
import time
import shutil
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuração ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = BASE_DIR / "raw"
LOG_DIR      = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collect_expanded.log", encoding="utf-8"),
        logging.StreamHandler(stream=None),   # None = resolve sys.stderr em tempo de emit
    ],
)
log = logging.getLogger(__name__)

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA   = "https://data.rcsb.org/rest/v1/core"
RCSB_FILES  = "https://files.rcsb.org/download"

REQUEST_DELAY = 0.25  # segundos entre requisições

# ══════════════════════════════════════════════════════════════════════════════
# LISTA CURADA — ALVOS BASEADOS NOS TERMOS DE BUSCA
# ══════════════════════════════════════════════════════════════════════════════

# Filtros de qualidade para busca geral
QUALITY_FILTERS = {
    "max_resolution_A":    2.5,   # mais restrito que os 3.0 Å da coleta inicial
    "min_heavy_atoms":     7,
    "max_ligand_mw":       800,
    "min_ligand_mw":       20,   # exclui fragmentos muito pequenos
}

def _normalize_cli_list(values: "list[str] | None") -> "list[str] | None":
    """Normaliza listas vindas da CLI removendo entradas vazias."""
    if not values:
        return None
    cleaned = [v.strip() for v in values if v and v.strip()]
    return cleaned or None


def _normalize_pdb_ids(values: "list[str] | None") -> "list[str] | None":
    """Normaliza PDB IDs vindos da CLI."""
    items = _normalize_cli_list(values)
    if not items:
        return None
    return [item.upper() for item in items]


def _load_target_list_json(path: str = "") -> "dict | None":
    """Carrega um dicionário de alvos a partir de JSON opcional."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("phase3-target-json must contain a JSON object.")
    return data

# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE BUSCA NO PDB
# ══════════════════════════════════════════════════════════════════════════════

def search_pdb_by_organism_and_ligand(organism: str, max_resolution: float) -> list[str]:
    """Busca estruturas de um organismo com ligantes não-polímeros."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                        "operator": "exact_match",
                        "value": organism,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                      "direction": "asc"}],
        },
    }
    try:
        resp = requests.post(RCSB_SEARCH, json=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids = [e["identifier"] for e in data.get("result_set", [])]
        log.info(f"  '{organism}': {len(ids)} estruturas")
        return ids
    except Exception as e:
        log.warning(f"  Busca falhou para '{organism}': {e}")
        return []


def search_pdb_by_description(term: str, max_resolution: float) -> list[str]:
    """Busca por termo na descrição da entidade proteica."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.pdbx_description",
                        "operator": "contains_phrase",
                        "value": term,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                      "direction": "asc"}],
        },
    }
    try:
        resp = requests.post(RCSB_SEARCH, json=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids = [e["identifier"] for e in data.get("result_set", [])]
        log.info(f"  '{term}': {len(ids)} estruturas")
        return ids
    except Exception as e:
        log.warning(f"  Busca falhou para '{term}': {e}")
        return []


def download_pdb_file(pdb_id: str, dest_dir: Path) -> bool:
    """Baixa arquivo PDB se ainda não existir."""
    out = dest_dir / f"{pdb_id}.pdb"
    if out.exists():
        return True
    url = f"{RCSB_FILES}/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        time.sleep(REQUEST_DELAY)
        return True
    except Exception as e:
        log.debug(f"  {pdb_id}: falha no download — {e}")
        return False


def download_sdf_file(comp_id: str, dest_dir: Path) -> bool:
    """Baixa SDF do ligante se ainda não existir."""
    out = dest_dir / f"{comp_id}.sdf"
    if out.exists():
        return True
    urls = [
        f"https://files.rcsb.org/ligands/download/{comp_id}_ideal.sdf",
        f"https://files.rcsb.org/ligands/{comp_id[0]}/{comp_id}/{comp_id}_ideal.sdf",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code in (404, 500):
                continue
            r.raise_for_status()
            if len(r.text) < 10:
                continue
            out.write_text(r.text)
            time.sleep(REQUEST_DELAY)
            return True
        except Exception:
            continue
    return False


def _fetch_ligand_ids(pdb_id: str, excluded: set, min_heavy: int) -> list[dict]:
    """Consulta a API REST do RCSB para identificar ligantes orgânicos de um PDB.

    Usa o endpoint /core/entry para listar entidades não-poliméricas e o
    endpoint /core/chemcomp/{comp_id} para obter o número de átomos pesados.
    Filtra solventes, íons e fragmentos muito pequenos.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        entry = r.json()
    except Exception:
        return []

    ids = (entry.get("rcsb_entry_container_identifiers", {})
                .get("non_polymer_entity_ids") or [])
    if not ids:
        return []

    ligands = []
    seen_comp_ids = set()

    for eid in ids:
        try:
            url2 = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id.upper()}/{eid}"
            r2 = requests.get(url2, timeout=30)
            if r2.status_code != 200:
                continue
            ent = r2.json()

            comp_id = (ent.get("pdbx_entity_nonpoly", {})
                       .get("comp_id", "")).strip().upper()
            if not comp_id or comp_id in excluded or comp_id in seen_comp_ids:
                continue

            chem_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
            r3 = requests.get(chem_url, timeout=30)
            if r3.status_code != 200:
                continue
            chem = r3.json()

            n_atoms = (chem.get("rcsb_chem_comp_info", {})
                           .get("atom_count_heavy", 0))
            if n_atoms < min_heavy:
                continue

            ligands.append({"comp_id": comp_id})
            seen_comp_ids.add(comp_id)
            time.sleep(0.05)
        except Exception:
            continue

    return ligands


def _download_ligands_for_ids(pdb_ids: list[str], sdf_dir: Path,
                               label: str = "") -> dict:
    """Consulta a API do RCSB para cada PDB ID, identifica ligantes orgânicos
    e baixa os SDFs correspondentes.

    Autossuficiente — não depende de scripts externos.
    Retorna dict com contadores: {found, downloaded, failed, no_ligands}
    """
    # Solventes, tampões e íons a excluir
    EXCLUDED_LIGS = {
        "HOH","WAT","DOD","H2O",
        "SO4","PO4","GOL","EDO","PEG","DMS","MPD","TRS","MES","HEP",
        "FMT","ACT","ACE","BME","DTT","EPE",
        "CL","NA","MG","ZN","CA","FE","MN","CU","CO","NI","K",
        "CD","HG","PB","IOD","BR","F","I","LI","CS","RB","BA",
        "NO3","NO2","CO3","HCO",
    }
    MIN_HEAVY = QUALITY_FILTERS.get("min_heavy_atoms", 7)

    sdf_dir.mkdir(parents=True, exist_ok=True)
    counts = {"found": 0, "downloaded": 0, "failed": 0, "no_ligands": 0}
    all_lig_ids: set[str] = set()

    log.info(f"Identifying ligands via RCSB API for {len(pdb_ids)} PDBs {label}...")
    for pid in tqdm(pdb_ids, desc=f"Ligands {label}"):
        ligands = _fetch_ligand_ids(pid, EXCLUDED_LIGS, MIN_HEAVY)
        if not ligands:
            counts["no_ligands"] += 1
        else:
            counts["found"] += len(ligands)
            all_lig_ids.update(l["comp_id"] for l in ligands)
        time.sleep(REQUEST_DELAY)

    log.info(f"  Unique ligands found: {len(all_lig_ids)}")
    log.info(f"  Downloading SDFs...")

    for lig_id in tqdm(sorted(all_lig_ids), desc=f"SDF {label}"):
        if download_sdf_file(lig_id, sdf_dir):
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1

    log.info(f"  SDF: {counts['downloaded']} downloaded, {counts['failed']} failed")
    return counts

# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — BUSCA CONFIGURÁVEL POR FAMÍLIA / ORGANISMO
# ══════════════════════════════════════════════════════════════════════════════

def collect_phase2(
    job_dir: "Path | str | None" = None,
    organism_terms: "list[str] | None" = None,
    description_terms: "list[str] | None" = None,
):
    """Coleta por família/organismo configurável para o job atual.

    Parâmetros
    ----------
    job_dir : Path | str | None
        Diretório raiz do job (JOBS/<job_id>/). Se None usa BASE_DIR.
        PDBs vão para <job_dir>/data/raw/pdb/phase2/
        SDFs vão para <job_dir>/data/raw/sdf/
    organism_terms : list[str] | None
        Termos de organismo para busca no RCSB (ex: "Homo sapiens").
        Obrigatório se description_terms também for vazio.
    description_terms : list[str] | None
        Termos de descrição de entidade proteica (ex: "protein kinase").
        Obrigatório se organism_terms também for vazio.

    Raises
    ------
    ValueError
        Se nenhum termo de busca for fornecido. Fases 2/3 requerem
        configuração explícita no Step 1 — não há padrões embutidos.
    """
    # Exige ao menos um conjunto de termos — sem fallback silencioso
    if not organism_terms and not description_terms:
        raise ValueError(
            "Phase 2 requires at least one search term.\n"
            "Provide 'organism_terms' (e.g. ['Mycobacterium tuberculosis']) "
            "or 'description_terms' (e.g. ['kinase', 'transferase']).\n"
            "In the CODEEP interface, fill in the fields in Step 1 before "
            "running Phase 2."
        )

    # Determina diretórios de saída
    _base = Path(job_dir) if job_dir is not None else BASE_DIR

    pdb_dir = _base / "data" / "raw" / "pdb" / "phase2"
    sdf_dir = _base / "data" / "raw" / "sdf" / "phase2"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    # Arquivo sentinela — evita re-download se a fase já foi concluída
    done_flag = pdb_dir / ".phase2_complete"
    existing  = list(pdb_dir.glob("*.pdb"))
    if done_flag.exists():
        log.info(f"Phase 2 already collected ({len(existing)} PDBs). "
                 "Delete .phase2_complete to force re-download.")
        return sorted(p.stem for p in existing)

    log.info("\n" + "═"*60)
    log.info("PHASE 2 — Configurable protein family collection")
    log.info(f"  Output         : {pdb_dir}")
    log.info(f"  Organism terms : {organism_terms}")
    log.info(f"  Desc. terms    : {description_terms}")
    log.info("═"*60)

    all_ids = set()

    # Busca por termos de descrição da entidade
    if description_terms:
        log.info("\nSearching by entity description...")
        for term in description_terms:
            ids = search_pdb_by_description(term, QUALITY_FILTERS["max_resolution_A"])
            all_ids.update(ids)
            time.sleep(REQUEST_DELAY)

    # Busca por organismo
    if organism_terms:
        log.info("\nSearching by organism...")
        for organism in organism_terms:
            ids = search_pdb_by_organism_and_ligand(organism, QUALITY_FILTERS["max_resolution_A"])
            all_ids.update(ids)
            time.sleep(REQUEST_DELAY)

    # Remove PDBs já baixados
    existing_ids = {p.stem.lower() for p in pdb_dir.glob("*.pdb")}
    to_download  = sorted(all_ids - existing_ids)
    log.info(f"\nTotal unique: {len(all_ids)} | Already present: {len(existing_ids)} "
             f"| To download: {len(to_download)}")

    ok = fail = 0
    for pid in tqdm(to_download, desc="Download Phase 2"):
        if download_pdb_file(pid, pdb_dir):
            ok += 1
        else:
            fail += 1

    # Salva lista de IDs
    out = _base / "data" / "phase2_ids.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sorted(all_ids), f)

    log.info(f"Phase 2 — PDB: {ok} downloaded, {fail} failed. IDs saved to {out}")

    lig_counts = _download_ligands_for_ids(sorted(all_ids), sdf_dir, label="(Phase 2)")
    log.info(f"Phase 2 complete: {ok} PDBs, {lig_counts['downloaded']} unique SDFs")

    done_flag.touch()
    return sorted(all_ids)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — ALVOS ESPECÍFICOS CONFIGURÁVEIS
# ══════════════════════════════════════════════════════════════════════════════

def collect_phase3(
    job_dir: "Path | str | None" = None,
    target_list: "dict | None" = None,
    extra_pdb_ids: "list[str] | None" = None,
):
    """Coleta de alvos específicos para o job atual.

    Parâmetros
    ----------
    job_dir : Path | str | None
        Diretório raiz do job. Se None usa BASE_DIR.
        PDBs vão para <job_dir>/data/raw/pdb/phase3/
    target_list : dict | None
        Dicionário de alvos no formato:
          { "key": { "description": ..., "pdb_ids": [...], ... } }
        Se None e extra_pdb_ids também vazio → levanta ValueError.
    extra_pdb_ids : list[str] | None
        PDB IDs específicos a baixar (pode ser usado sem target_list).

    Raises
    ------
    ValueError
        Se nem target_list nem extra_pdb_ids forem fornecidos.
        Fases 2/3 requerem configuração explícita no Step 1.
    """
    # Exige configuração explícita — sem fallback para lista Mtb
    has_targets  = target_list and len(target_list) > 0
    has_ids      = extra_pdb_ids and len(extra_pdb_ids) > 0
    if not has_targets and not has_ids:
        raise ValueError(
            "Phase 3 requires either a target_list or a list of extra_pdb_ids.\n"
            "Provide specific PDB IDs (e.g. ['7W40', '6WVI']) or a target_list dict.\n"
            "In the CODEEP interface, fill in 'Phase 3 PDB IDs' in Step 1 before "
            "running Phase 3."
        )

    _base = Path(job_dir) if job_dir is not None else BASE_DIR

    pdb_dir = _base / "data" / "raw" / "pdb" / "phase3"
    sdf_dir = _base / "data" / "raw" / "sdf" / "phase3"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    # Arquivo sentinela
    done_flag = pdb_dir / ".phase3_complete"
    existing  = list(pdb_dir.glob("*.pdb"))
    if done_flag.exists():
        log.info(f"Phase 3 already collected ({len(existing)} PDBs). "
                 "Delete .phase3_complete to force re-download.")
        return sorted(p.stem for p in existing)

    log.info("\n" + "═"*60)
    log.info("PHASE 3 — Specific target collection")
    log.info(f"  Output  : {pdb_dir}")
    log.info(f"  Targets : {len(target_list or {})} entries")
    log.info(f"  Extra IDs: {len(extra_pdb_ids or [])}")
    log.info("═"*60)

    all_ids       = set(extra_pdb_ids or [])
    priority_meta = []

    for target_key, info in (target_list or {}).items():
        ids = info.get("pdb_ids", [])
        log.info(f"\n  {info.get('description', target_key)}")
        log.info(f"  Priority: {info.get('priority','?')} | IDs: {ids}")
        all_ids.update(ids)
        for pid in ids:
            priority_meta.append({
                "pdb_id":      pid,
                "target_key":  target_key,
                "description": info.get("description", ""),
                "priority":    info.get("priority", ""),
                "uniprot":     info.get("uniprot", ""),
                "note":        info.get("note", ""),
                "phase":       3,
            })

    # Remove já baixados
    existing_ids = {p.stem.lower() for p in pdb_dir.glob("*.pdb")}
    to_download  = sorted(all_ids - existing_ids)
    log.info(f"\nTotal: {len(all_ids)} | Already present: {len(existing_ids)} "
             f"| To download: {len(to_download)}")

    ok = fail = 0
    for pid in tqdm(to_download, desc="Download Phase 3"):
        if download_pdb_file(pid, pdb_dir):
            ok += 1
        else:
            fail += 1

    # Salva metadados de alvos
    if priority_meta:
        import pandas as _pd
        df = _pd.DataFrame(priority_meta)
        out_csv = _base / "data" / "phase3_priority_targets.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        log.info(f"Target metadata saved to: {out_csv}")

    lig_counts = _download_ligands_for_ids(sorted(all_ids), sdf_dir, label="(Phase 3)")
    log.info(f"Phase 3 complete: {ok} PDBs, {lig_counts['downloaded']} unique SDFs")

    done_flag.touch()
    return sorted(all_ids)



# ══════════════════════════════════════════════════════════════════════════════
# DATASETS EXTERNOS — FASE 1
# ══════════════════════════════════════════════════════════════════════════════

EXTERNAL_DIR = BASE_DIR / "external"

def _build_rcsb_wide_query(
    max_resolution: float,
    start: int = 0,
    rows: int = 1000,
    min_resolution: float = 0.0,
) -> dict:
    """Monta a query base para busca ampla de complexos proteína-ligante no RCSB."""
    nodes = []

    if min_resolution > 0:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "greater_or_equal",
                "value": round(min_resolution, 3),
            },
        })

    nodes.extend([
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_resolution,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater_or_equal",
                "value": 1,
            },
        },
    ])

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [{
                "sort_by": "rcsb_entry_info.resolution_combined",
                "direction": "asc",
            }],
        },
    }


def _extract_rcsb_ids(payload: dict) -> list[str]:
    """Extrai PDB IDs do payload do RCSB, normalizando para lowercase."""
    ids = []
    for item in payload.get("result_set", []):
        identifier = str(item.get("identifier", "")).strip()
        if identifier:
            ids.append(identifier.lower())
    return ids


def _search_rcsb_wide_paginated(
    max_resolution: float = 2.5,
    page_size: int = 1000,
) -> tuple[list[str], bool]:
    """Busca ampla usando paginação iterativa do RCSB até esgotar os resultados.

    Retorna (ids, used_pagination_ok). Se detectar paginação inconsistente,
    devolve False para acionar o fallback por faixas de resolução.
    """
    seen_ids: set[str] = set()
    start = 0
    page_idx = 0
    last_signature = None

    log.info(
        f"RCSB-wide: busca paginada iterativa "
        f"(resolution <= {max_resolution:.2f} A, rows={page_size})"
    )

    while True:
        query = _build_rcsb_wide_query(
            max_resolution=max_resolution,
            start=start,
            rows=page_size,
        )
        try:
            resp = requests.post(RCSB_SEARCH, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Falha na paginação do RCSB-wide em start={start}: {e}")
            return sorted(seen_ids), False

        page_ids = _extract_rcsb_ids(data)
        total_count = data.get("total_count")

        if not page_ids:
            log.info("RCSB-wide: paginação encerrada sem novos resultados.")
            break

        signature = (page_ids[0], page_ids[-1], len(page_ids))
        if page_idx > 0 and signature == last_signature:
            log.warning(
                "RCSB-wide: página repetida detectada com 'start' crescente. "
                "Assumindo paginação inconsistente e ativando fallback por resolução."
            )
            return sorted(seen_ids), False
        last_signature = signature

        before = len(seen_ids)
        seen_ids.update(page_ids)
        added = len(seen_ids) - before
        page_idx += 1

        log.info(
            f"  Página {page_idx:>3}: {len(page_ids)} hits, "
            f"{added} novos, total único={len(seen_ids)}"
        )

        if total_count is not None and len(seen_ids) >= int(total_count):
            break

        if len(page_ids) < page_size:
            break

        start += page_size
        time.sleep(REQUEST_DELAY)

    return sorted(seen_ids), True


def _search_rcsb_wide_resolution_bands(
    max_resolution: float = 2.5,
    page_size: int = 1000,
) -> list[str]:
    """Fallback sem depender de paginação: divide a busca por faixas de resolução.

    A ideia é manter cada consulta abaixo do limite de rows. Se uma faixa ainda
    vier saturada, ela é subdividida recursivamente.
    """
    pending_bands: list[tuple[float, float]] = [(0.0, max_resolution)]
    seen_ids: set[str] = set()
    min_band_width = 0.01

    log.info(
        f"RCSB-wide: fallback por faixas de resolução "
        f"(resolution <= {max_resolution:.2f} A, rows={page_size})"
    )

    while pending_bands:
        low, high = pending_bands.pop(0)
        query = _build_rcsb_wide_query(
            max_resolution=high,
            min_resolution=low,
            start=0,
            rows=page_size,
        )

        try:
            resp = requests.post(RCSB_SEARCH, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Faixa {low:.2f}-{high:.2f} A falhou: {e}")
            continue

        band_ids = _extract_rcsb_ids(data)
        band_width = round(high - low, 4)

        if len(band_ids) >= page_size and band_width > min_band_width:
            mid = round((low + high) / 2.0, 3)
            log.info(
                f"  Faixa {low:.2f}-{high:.2f} A saturou com {len(band_ids)} hits; "
                f"dividindo em {low:.2f}-{mid:.2f} e {mid:.2f}-{high:.2f} A"
            )
            pending_bands.insert(0, (mid, high))
            pending_bands.insert(0, (low, mid))
            time.sleep(REQUEST_DELAY)
            continue

        if len(band_ids) >= page_size and band_width <= min_band_width:
            log.warning(
                f"  Faixa {low:.2f}-{high:.2f} A continua saturada "
                f"({len(band_ids)} hits) mesmo no limite mínimo. "
                "Mantendo resultado potencialmente truncado dessa faixa."
            )

        before = len(seen_ids)
        seen_ids.update(band_ids)
        added = len(seen_ids) - before

        log.info(
            f"  Faixa {low:.2f}-{high:.2f} A: "
            f"{len(band_ids)} hits, {added} novos, total único={len(seen_ids)}"
        )
        time.sleep(REQUEST_DELAY)

    return sorted(seen_ids)


def collect_rcsb_wide(
    max_entries: int = 50000,
    max_resolution: float = 2.5,
    page_size: int = 1000,
) -> bool:
    """Coleta ampla de complexos proteína-ligante do RCSB com resolução <= 2.5 A.

    Estratégia:
      1. tenta paginação iterativa até esgotar os resultados;
      2. se a paginação parecer inconsistente, cai para subdivisão por
         faixas de resolução até cada faixa caber em uma página.
    """
    wide_dir = EXTERNAL_DIR / "rcsb_wide"
    pdb_dir = RAW_DIR / "pdb" / "rcsb_wide"
    sdf_dir = RAW_DIR / "sdf" / "rcsb_wide"

    wide_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    done_flag = wide_dir / ".download_complete"
    ids_json = wide_dir / "rcsb_wide_ids.json"

    existing_pdb = len(list(pdb_dir.glob("*.pdb")))
    existing_sdf = len(list(sdf_dir.glob("*.sdf")))

    if done_flag.exists() and existing_pdb > 0 and existing_sdf > 0:
        log.info(
            f"RCSB-wide já coletado ({existing_pdb} PDBs, {existing_sdf} SDFs). "
            "Delete .download_complete para forçar novo download."
        )
        return True

    if done_flag.exists() and existing_pdb > 0 and existing_sdf == 0:
        log.warning(
            "RCSB-wide marcado como concluído, mas sem SDFs. "
            "Continuando para preencher os ligantes."
        )

    log.info("═" * 60)
    log.info("RCSB-wide — Busca ampla de complexos proteína-ligante")
    log.info(f"Critérios: resolução <= {max_resolution:.2f} A e >= 1 ligante não-polimérico")
    log.info("═" * 60)

    pdb_ids, pagination_ok = _search_rcsb_wide_paginated(
        max_resolution=max_resolution,
        page_size=page_size,
    )

    if not pagination_ok:
        log.warning(
            "Paginação iterativa indisponível ou inconsistente. "
            "Usando fallback por faixas de resolução."
        )
        pdb_ids = _search_rcsb_wide_resolution_bands(
            max_resolution=max_resolution,
            page_size=page_size,
        )

    if not pdb_ids:
        log.error("RCSB-wide não retornou nenhum PDB elegível.")
        return False

    ids_json.write_text(json.dumps(pdb_ids, indent=2))
    log.info(f"PDB IDs salvos em: {ids_json}")
    log.info(f"Total de PDB IDs elegíveis: {len(pdb_ids)}")

    if max_entries and len(pdb_ids) > max_entries:
        pdb_ids = pdb_ids[:max_entries]
        log.info(f"Limitando RCSB-wide aos primeiros {max_entries} complexos elegíveis")

    existing_ids = {p.stem.lower() for p in pdb_dir.glob("*.pdb")}
    to_download = sorted(set(pdb_ids) - existing_ids)
    log.info(
        f"PDBs já presentes: {len(existing_ids)} | "
        f"Para baixar: {len(to_download)}"
    )

    ok = fail = 0
    for pid in tqdm(to_download, desc="PDB (RCSB-wide)"):
        out = pdb_dir / f"{pid}.pdb"
        if out.exists():
            ok += 1
            continue
        try:
            r = requests.get(
                f"{RCSB_FILES}/{pid.upper()}.pdb",
                timeout=60,
                stream=True,
            )
            if r.status_code == 200:
                with open(out, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        time.sleep(REQUEST_DELAY)

    log.info(f"PDB: {ok} baixados, {fail} falhas")

    lig_counts = _download_ligands_for_ids(pdb_ids, sdf_dir, label="(RCSB-wide)")
    done_flag.touch()

    log.info("RCSB-wide concluído.")
    log.info(f"  PDB: {pdb_dir}")
    log.info(f"  SDF: {sdf_dir}")
    log.info(
        f"  Ligantes: {lig_counts['downloaded']} SDFs baixados, "
        f"{lig_counts['failed']} falhas, "
        f"{lig_counts['no_ligands']} PDBs sem ligantes elegíveis"
    )
    return True

def collect_lp_pdbbind(max_entries: int = 50000) -> bool:
    """Baixa e importa o LP-PDBBind (Leak-Proof PDBBind) via GitHub.

    Dataset de ~17.000 pares proteína-ligante curado sem data leakage,
    com splits anti-vazamento baseados em similaridade de sequência e estrutura.
    Não requer registro — download direto do GitHub.

    Referência: Li et al. (2024) arXiv:2308.09639
    GitHub: https://github.com/THGLab/LP-PDBBind
    """
    lp_dir  = EXTERNAL_DIR / "lp_pdbbind"
    pdb_dir = RAW_DIR / "pdb" / "lp_pdbbind"
    sdf_dir = RAW_DIR / "sdf" / "lp_pdbbind"
    lp_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    done_flag = lp_dir / ".download_complete"

    existing_pdb = len(list(pdb_dir.glob("*.pdb")))
    existing_sdf = len(list(sdf_dir.glob("*.sdf")))

    if done_flag.exists() and existing_pdb > 0 and existing_sdf > 0:
        log.info(
            f"LP-PDBBind já baixado ({existing_pdb} PDBs, {existing_sdf} SDFs). "
            "Delete .download_complete para forçar novo download."
        )
        return True

    if done_flag.exists() and existing_pdb > 0 and existing_sdf == 0:
        log.warning(
            "LP-PDBBind marcado como concluído, mas sem SDFs. "
            "Continuando para preencher os ligantes."
        )

    log.info("═" * 60)
    log.info("LP-PDBBind — Download via GitHub")
    log.info("═" * 60)

    csv_url = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv"
    csv_path = lp_dir / "LP-PDBBind.csv"
    if not csv_path.exists():
        log.info("Baixando LP-PDBBind.csv...")
        try:
            r = requests.get(csv_url, timeout=60)
            r.raise_for_status()
            csv_path.write_bytes(r.content)
            log.info(f"Metadados salvos: {csv_path}")
        except Exception as e:
            log.error(f"Falha ao baixar LP-PDBBind.csv: {e}")
            return False

    df = pd.read_csv(csv_path)
    id_col = next(
        (c for c in df.columns if c.lower() in ("pdbid", "pdb_id", "pdb", "code", "id")),
        df.columns[0],
    )
    log.info(f"Coluna de PDB ID detectada: '{id_col}'")

    if "CL1" in df.columns and "covalent" in df.columns:
        df = df[(df["CL1"] == True) & (df["covalent"] == False)]
        log.info(f"Filtro CL1 não-covalente: {len(df)} entradas")

    pdb_ids = df[id_col].dropna().astype(str).str.lower().unique().tolist()
    log.info(f"PDB IDs no CSV: {len(pdb_ids)}")

    if max_entries and len(pdb_ids) > max_entries:
        pdb_ids = pdb_ids[:max_entries]
        log.info(f"Limitando LP-PDBBind aos primeiros {max_entries} complexos")

    log.info(f"Baixando {len(pdb_ids)} PDBs do RCSB...")
    ok = fail = 0
    for pid in tqdm(pdb_ids, desc="PDB (LP-PDBBind)"):
        out = pdb_dir / f"{pid}.pdb"
        if out.exists():
            ok += 1
            continue
        try:
            r = requests.get(
                f"https://files.rcsb.org/download/{pid.upper()}.pdb",
                timeout=60,
                stream=True,
            )
            if r.status_code == 200:
                with open(out, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        time.sleep(REQUEST_DELAY)

    log.info(f"PDB: {ok} baixados, {fail} falhas")

    lig_counts = _download_ligands_for_ids(pdb_ids, sdf_dir, label="(LP-PDBBind)")
    done_flag.touch()

    log.info("LP-PDBBind concluído.")
    log.info(f"  PDB: {pdb_dir}")
    log.info(f"  SDF: {sdf_dir}")
    log.info(
        f"  Ligantes: {lig_counts['downloaded']} SDFs baixados, "
        f"{lig_counts['failed']} falhas, {lig_counts['no_ligands']} PDBs sem ligantes elegíveis"
    )
    return True

def collect_plinder(
    max_entries: int = 50000,
    split: str = "train",
    min_resolution: float = 2.5,
) -> bool:
    """Baixa e importa sistemas do PLINDER via API Python.

    PLINDER (Protein-Ligand INteraction Dataset and Evaluation Resource)
    é o maior dataset aberto de interações proteína-ligante (~449k sistemas).
    Mantido por Basel/NVIDIA/MIT, com splits anti-leakage e anotações extensas.

    Requer: pip install plinder

    Referência: Durairaj et al. (2024) bioRxiv 2024.07.17.603955
    GitHub: https://github.com/plinder-org/plinder
    """
    pdb_dir = RAW_DIR / "pdb" / "plinder"
    sdf_dir = RAW_DIR / "sdf" / "plinder"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    done_flag = EXTERNAL_DIR / "plinder" / ".download_complete"
    done_flag.parent.mkdir(parents=True, exist_ok=True)

    if done_flag.exists():
        log.info("PLINDER já baixado. Delete .download_complete para novo download.")
        return True

    # Verifica instalação
    try:
        from plinder.core import PlinderSystem
        from plinder.core.split.split import get_split
        import plinder
        log.info(f"PLINDER versão: {plinder.__version__}")
    except ImportError:
        log.error("PLINDER não instalado. Execute: pip install plinder")
        log.error("Documentação: https://plinder-org.github.io/plinder/")
        return False

    log.info("═" * 60)
    log.info(f"PLINDER — Importando até {max_entries} sistemas (split={split})")
    log.info("═" * 60)

    try:
        # Carrega split oficial com splits anti-leakage
        split_df = get_split()
        systems_in_split = split_df[split_df["split"] == split]["system_id"].tolist()
        log.info(f"Sistemas no split '{split}': {len(systems_in_split)}")

        # Filtra por resolução se disponível
        if max_entries < len(systems_in_split):
            systems_in_split = systems_in_split[:max_entries]
            log.info(f"Limitando a {max_entries} sistemas")

        ok = fail = 0
        for sys_id in tqdm(systems_in_split, desc="PLINDER sistemas"):
            try:
                system = PlinderSystem(system_id=sys_id)

                # Salva receptor PDB
                pdb_out = pdb_dir / f"{sys_id}.pdb"
                if not pdb_out.exists() and system.receptor_pdb:
                    import shutil
                    shutil.copy2(system.receptor_pdb, pdb_out)

                # Salva ligantes SDF
                if system.ligand_sdfs:
                    for i, sdf_src in enumerate(system.ligand_sdfs):
                        sdf_out = sdf_dir / f"{sys_id}_lig{i}.sdf"
                        if not sdf_out.exists():
                            shutil.copy2(sdf_src, sdf_out)
                ok += 1
            except Exception as e:
                fail += 1
                log.debug(f"  {sys_id}: {e}")

        log.info(f"PLINDER: {ok} sistemas importados, {fail} falhas")
        done_flag.touch()
        return True

    except Exception as e:
        log.error(f"Erro ao acessar PLINDER: {e}")
        return False


def collect_moad(moad_dir: str = "", max_entries: int = 50000) -> bool:
    """Importa o Binding MOAD de diretório local previamente baixado.

    O Binding MOAD (~41.000 estruturas proteína-ligante) requer registro
    gratuito em bindingmoad.org. Após download manual, aponte o diretório:

    Passos:
      1. Registre-se em: http://www.bindingmoad.org/
      2. Baixe: BindingMOAD_2020.tar.gz ou versão mais recente
      3. Extraia: tar -xzf BindingMOAD_2020.tar.gz
      4. Execute: python pipeline_collect.py --dataset moad --moad-dir ./BindingMOAD_2020/

    Referência: Benson et al. (2008) Nucleic Acids Research 36:D674-D678
    URL: http://www.bindingmoad.org/
    """
    if not moad_dir:
        log.error("Binding MOAD requer download manual e o argumento --moad-dir.")
        log.error("Passos:")
        log.error("  1. Registre-se em: http://www.bindingmoad.org/")
        log.error("  2. Baixe e extraia o arquivo .tar.gz")
        log.error("  3. Execute com: --dataset moad --moad-dir /caminho/para/BindingMOAD/")
        return False

    src_dir = Path(moad_dir)
    if not src_dir.exists():
        log.error(f"Diretório não encontrado: {src_dir}")
        return False

    pdb_dir = RAW_DIR / "pdb" / "moad"
    sdf_dir = RAW_DIR / "sdf" / "moad"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info(f"Binding MOAD — Importando de {src_dir}")
    log.info("═" * 60)

    # Procura PDBs e SDFs no diretório extraído
    pdb_files = sorted(src_dir.rglob("*.pdb"))
    sdf_files = sorted(src_dir.rglob("*.sdf"))

    if max_entries and len(pdb_files) > max_entries:
        pdb_files = pdb_files[:max_entries]
        log.info(f"Limitando MOAD aos primeiros {max_entries} PDBs encontrados")
    if max_entries and len(sdf_files) > max_entries:
        sdf_files = sdf_files[:max_entries]
        log.info(f"Limitando MOAD aos primeiros {max_entries} SDFs encontrados")

    log.info(f"Encontrados: {len(pdb_files)} PDBs, {len(sdf_files)} SDFs")

    ok_pdb = ok_sdf = 0
    for pdb_src in tqdm(pdb_files, desc="PDB (MOAD)"):
        dst = pdb_dir / pdb_src.name
        if not dst.exists():
            shutil.copy2(pdb_src, dst)
        ok_pdb += 1

    for sdf_src in tqdm(sdf_files, desc="SDF (MOAD)"):
        dst = sdf_dir / sdf_src.name
        if not dst.exists():
            shutil.copy2(sdf_src, dst)
        ok_sdf += 1

    log.info(f"MOAD importado: {ok_pdb} PDBs, {ok_sdf} SDFs")
    log.info(f"  PDB: {pdb_dir}")
    log.info(f"  SDF: {sdf_dir}")
    return True


def collect_external_dataset(dataset: str, moad_dir: str = "",
                              max_entries: int = 50000) -> bool:
    """Dispatcher para os datasets externos disponíveis.

    Args:
        dataset:     nome do dataset ('lp-pdbbind','rcsb-wide','plinder','moad')
        moad_dir:    diretório local do MOAD (obrigatório para dataset='moad')
        max_entries: número máximo de complexos/sistemas a importar na Fase 1
    """
    dataset = dataset.lower().strip()

    if dataset == "lp-pdbbind":
        return collect_lp_pdbbind(max_entries=max_entries)

    elif dataset == "rcsb-wide":
        return collect_rcsb_wide(max_entries=max_entries)

    elif dataset == "plinder":
        return collect_plinder(max_entries=max_entries)

    elif dataset == "moad":
        return collect_moad(moad_dir, max_entries=max_entries)

    else:
        log.error(f"Dataset '{dataset}' não reconhecido.")
        log.error("Opções: lp-pdbbind | rcsb-wide | plinder | moad")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# STATUS GERAL
# ══════════════════════════════════════════════════════════════════════════════

def show_status(job_dir: "Path | str | None" = None):
    """Mostra o status da pipeline em três fases.

    Estrutura esperada após os ajustes:
      Phase 1 (global):
        <root>/data/raw/pdb/<dataset>/
        <root>/data/curated/
        <root>/data/tensors/
        <root>/checkpoints/phase1/

      Phase 2 (job):
        <job_dir>/data/raw/pdb/phase2/
        <job_dir>/data/curated/phase2/
        <job_dir>/data/tensors/phase2/
        <job_dir>/checkpoints/phase2/

      Phase 3 (job):
        <job_dir>/data/raw/pdb/phase3/
        <job_dir>/data/curated/phase3/
        <job_dir>/data/tensors/phase3/
        <job_dir>/checkpoints/phase3/
    """
    root_dir = BASE_DIR.parent
    job_path = Path(job_dir) if job_dir else None

    def _count_files(path: Path, pattern: str) -> int:
        if not path.exists():
            return 0
        return len(list(path.glob(pattern)))

    def _count_pair_dirs(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for item in path.iterdir() if item.is_dir())

    def _size_mb(path: Path, pattern: str) -> float:
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.glob(pattern) if f.is_file()) / 1e6

    def _ckpt_status(path: Path) -> str:
        best = path / "best_model.pt"
        last = path / "last_model.pt"
        if best.exists():
            return "best_model.pt"
        if last.exists():
            return "last_model.pt only"
        return "missing"

    log.info("\n" + "═" * 72)
    log.info("CODEEP DATA STATUS")
    log.info("═" * 72)

    # ── Phase 1: global ------------------------------------------------------
    log.info("\n[PHASE 1] General pretraining dataset (global)")

    phase1_sources = {
        "lp_pdbbind": {
            "pdb": RAW_DIR / "pdb" / "lp_pdbbind",
            "sdf": RAW_DIR / "sdf" / "lp_pdbbind",
        },
        "rcsb_wide": {
            "pdb": RAW_DIR / "pdb" / "rcsb_wide",
            "sdf": RAW_DIR / "sdf" / "rcsb_wide",
        },           
        "plinder": {
            "pdb": RAW_DIR / "pdb" / "plinder",
            "sdf": RAW_DIR / "sdf" / "plinder",
        },
        "moad": {
            "pdb": RAW_DIR / "pdb" / "moad",
            "sdf": RAW_DIR / "sdf" / "moad",
        },     
    }

    phase1_total_raw = 0
    phase1_total_sdf = 0
    any_phase1_raw = False

    for name, paths in phase1_sources.items():
        n_pdb = _count_files(paths["pdb"], "*.pdb")
        n_sdf = _count_files(paths["sdf"], "*.sdf")
        if n_pdb > 0 or n_sdf > 0:
            any_phase1_raw = True
            phase1_total_raw += n_pdb
            phase1_total_sdf += n_sdf
            log.info(f"  Raw PDB  [{name:<12}] : {n_pdb:>6} files ({_size_mb(paths['pdb'], '*.pdb'):.0f} MB)")
            log.info(f"  Raw SDF  [{name:<12}] : {n_sdf:>6} files ({_size_mb(paths['sdf'], '*.sdf'):.0f} MB)")

    if not any_phase1_raw:
        log.info("  Raw PDB                : no Phase 1 dataset collected yet")
        log.info("  Raw SDF                : no Phase 1 dataset collected yet")
    phase1_curated_dir = root_dir / "data" / "curated"
    phase1_tensor_dir = root_dir / "data" / "tensors"
    phase1_ckpt_dir = root_dir / "checkpoints" / "phase1"

    phase1_curated = _count_pair_dirs(phase1_curated_dir)
    phase1_tensors = _count_pair_dirs(phase1_tensor_dir)
    phase1_tensor_index = (phase1_tensor_dir / "tensor_index.csv").exists()

    log.info(f"  Raw PDB total          : {phase1_total_raw:>6}")
    log.info(f"  Raw SDF total          : {phase1_total_sdf:>6}")
    log.info(f"  Curated pairs          : {phase1_curated:>6}")
    log.info(f"  Tensor pairs           : {phase1_tensors:>6}")
    log.info(f"  Tensor index           : {'present' if phase1_tensor_index else 'missing'}")
    log.info(f"  Checkpoint             : {_ckpt_status(phase1_ckpt_dir)}")

    # ── Job-local phases -----------------------------------------------------
    if not job_path:
        log.info("\n[PHASE 2 / PHASE 3] Job-specific specialization")
        log.info("  No job_dir provided. Job-local status cannot be evaluated.")
        log.info("  Pass JOBS/<job_id> to show_status(job_dir=...) to inspect phases 2 and 3.")
        log.info("─" * 72)
        return

    log.info(f"\nJob directory: {job_path}")

    phase2_raw_dir = job_path / "data" / "raw" / "pdb" / "phase2"
    phase2_curated_dir = job_path / "data" / "curated" / "phase2"
    phase2_tensor_dir = job_path / "data" / "tensors" / "phase2"
    phase2_ckpt_dir = job_path / "checkpoints" / "phase2"

    phase3_raw_dir = job_path / "data" / "raw" / "pdb" / "phase3"
    phase3_curated_dir = job_path / "data" / "curated" / "phase3"
    phase3_tensor_dir = job_path / "data" / "tensors" / "phase3"
    phase3_ckpt_dir = job_path / "checkpoints" / "phase3"

    phase2_sdf_dir = job_path / "data" / "raw" / "sdf" / "phase2"
    phase3_sdf_dir = job_path / "data" / "raw" / "sdf" / "phase3"

    # Phase 2
    log.info("\n[PHASE 2] Family / organism specialization")
    p2_raw = _count_files(phase2_raw_dir, "*.pdb")
    p2_curated = _count_pair_dirs(phase2_curated_dir)
    p2_tensors = _count_pair_dirs(phase2_tensor_dir)
    p2_tensor_index = (phase2_tensor_dir / "tensor_index.csv").exists()
    p2_sdf = _count_files(phase2_sdf_dir, "*.sdf")
    p2_ready = (phase1_ckpt_dir / "best_model.pt").exists()

    log.info(f"  Raw PDB                : {p2_raw:>6}")
    log.info(f"  Raw SDF (job shared)   : {p2_sdf:>6}")
    log.info(f"  Curated pairs          : {p2_curated:>6}")
    log.info(f"  Tensor pairs           : {p2_tensors:>6}")
    log.info(f"  Tensor index           : {'present' if p2_tensor_index else 'missing'}")
    log.info(f"  Resume from Phase 1    : {'ready' if p2_ready else 'missing phase1/best_model.pt'}")
    log.info(f"  Checkpoint             : {_ckpt_status(phase2_ckpt_dir)}")

    # Phase 3
    log.info("\n[PHASE 3] Target-specific specialization")
    p3_raw = _count_files(phase3_raw_dir, "*.pdb")
    p3_curated = _count_pair_dirs(phase3_curated_dir)
    p3_tensors = _count_pair_dirs(phase3_tensor_dir)
    p3_tensor_index = (phase3_tensor_dir / "tensor_index.csv").exists()
    p3_sdf = _count_files(phase3_sdf_dir, "*.sdf")
    p3_ready = (phase2_ckpt_dir / "best_model.pt").exists()

    log.info(f"  Raw PDB                : {p3_raw:>6}")
    log.info(f"  Raw SDF (job shared)   : {p3_sdf:>6}")
    log.info(f"  Curated pairs          : {p3_curated:>6}")
    log.info(f"  Tensor pairs           : {p3_tensors:>6}")
    log.info(f"  Tensor index           : {'present' if p3_tensor_index else 'missing'}")
    log.info(f"  Resume from Phase 2    : {'ready' if p3_ready else 'missing phase2/best_model.pt'}")
    log.info(f"  Checkpoint             : {_ckpt_status(phase3_ckpt_dir)}")

    log.info("\n[SUMMARY]")
    log.info(f"  Phase 1 ready for generation : {'yes' if (phase1_ckpt_dir / 'best_model.pt').exists() else 'no'}")
    log.info(f"  Phase 2 ready for training   : {'yes' if p2_raw > 0 and p2_curated > 0 and p2_tensors > 0 and p2_ready else 'no'}")
    log.info(f"  Phase 3 ready for training   : {'yes' if p3_raw > 0 and p3_curated > 0 and p3_tensors > 0 and p3_ready else 'no'}")
    log.info(f"  Phase 2 model available      : {'yes' if (phase2_ckpt_dir / 'best_model.pt').exists() else 'no'}")
    log.info(f"  Phase 3 model available      : {'yes' if (phase3_ckpt_dir / 'best_model.pt').exists() else 'no'}")

    log.info("─" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO DE TREINAMENTO EM FASES
# ══════════════════════════════════════════════════════════════════════════════

def print_training_plan():
    """Imprime o plano completo de treinamento em fases."""
    log.info("\n" + "═"*60)
    log.info("PLANO DE TREINAMENTO EM FASES")
    log.info("═"*60)
    log.info("""
  FASE 1 — Pré-treino geral (base de conhecimento)
  ─────────────────────────────────────────────────
  Dados:   LP-PDBBind (dataset externo principal da Fase 1)
  Objetivo: aprender física geral de interação proteína-ligante
  Download + treino:
    python pipeline_collect.py --dataset lp-pdbbind

  FASE 2 — Especialização por família / organismo
  ───────────────────────────────────────────────
  Dados:   Estruturas recuperadas com os termos definidos pelo usuário
           em organismo e/ou descrição proteica
  Partida: checkpoints_phase1/best_model.pt
  Epochs:  30–50
  LR:      1e-5 (10x menor — preserva conhecimento da Fase 1)
  Objetivo: especializar o modelo em uma família ou classe de alvo
  Comando:
    python 05_train.py --epochs 50 --lr 1e-5 \\
      --resume checkpoints_phase1/best_model.pt \\
      --checkpoint-dir checkpoints_phase2_target_family/

  FASE 3 — Especialização em alvos específicos
  ────────────────────────────────────────────
  Dados:   PDB IDs explícitos fornecidos pelo usuário para o alvo final
  Partida: checkpoints_phase2_target_family/best_model.pt
  Epochs:  20–30
  LR:      5e-6
  Objetivo: maximizar a relevância para o alvo específico do projeto
  Comando:
    python 05_train.py --epochs 30 --lr 5e-6 \\
      --resume checkpoints_phase2_target_family/best_model.pt \\
      --checkpoint-dir checkpoints_phase3_target_specific/

  GERAÇÃO FINAL
  ─────────────
    python 06_evaluate.py --ckpt checkpoints_phase3_target_specific/ --generate
""")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coleta expandida de dados proteína-ligante orientada a alvos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Fase 1 com lp-PDBBind
  python pipeline_collect.py --dataset lp-pdbbind
  
  # Fase 1 com busca ampla no RCSB (complexos com resolucao <= 2.5 A)
  python pipeline_collect.py --dataset rcsb-wide
  
  # Fase 1 com PLINDER (maior dataset, requer: pip install plinder)
  python pipeline_collect.py --dataset plinder --plinder-max 50000

  # Fase 1 com Binding MOAD (requer download manual em bindingmoad.org)
  python pipeline_collect.py --dataset moad --moad-dir ./BindingMOAD_2020/

  # Fase 2 por organismo
  python pipeline_collect.py --phase 2 \
      --phase2-organism "Homo sapiens" "Mus musculus"

  # Fase 2 por descrição proteica
  python pipeline_collect.py --phase 2 \
      --phase2-description "protein kinase" "ATP-binding"

  # Fase 3 por PDB IDs explícitos
  python pipeline_collect.py --phase 3 \
      --phase3-pdb-ids 1ATP 2SRC 4HJB

  # Tudo de uma vez
  python pipeline_collect.py --dataset lp-pdbbind --phase all \
      --phase2-organism "Homo sapiens" \
      --phase2-description "protein kinase" \
      --phase3-pdb-ids 1ATP 2SRC 4HJB
        """
    )

    parser.add_argument(
        "--dataset",
        choices=["lp-pdbbind", "rcsb-wide", "plinder", "moad"],
        default=None,
        metavar="DATASET",
        help="lp-pdbbind | rcsb-wide | plinder | moad"
    )
    parser.add_argument(
        "--moad-dir", type=str, default="",
        help="Diretório local do Binding MOAD extraído (obrigatório com --dataset moad)"
    )
    parser.add_argument(
        "--plinder-max", type=int, default=50000,
        help="Número máximo de sistemas PLINDER a importar (default: 50000)"
    )

    parser.add_argument(
        "--phase", choices=["2", "3", "all"],
        help="Fase a executar (2=busca configurável, 3=alvos específicos, all=ambas)"
    )
    parser.add_argument(
        "--phase2-organism", nargs="*", default=None, metavar="TERM",
        help="Termos de organismo para a Fase 2 (ex: 'Homo sapiens')"
    )
    parser.add_argument(
        "--phase2-description", nargs="*", default=None, metavar="TERM",
        help="Termos de descrição proteica para a Fase 2 (ex: 'protein kinase')"
    )
    parser.add_argument(
        "--phase3-pdb-ids", nargs="*", default=None, metavar="PDB",
        help="PDB IDs específicos para a Fase 3"
    )
    parser.add_argument(
        "--phase3-target-json", type=str, default="", metavar="FILE",
        help="JSON opcional com metadados de alvos para a Fase 3"
    )

    parser.add_argument("--status", action="store_true",
                        help="Mostra status dos dados coletados")
    parser.add_argument("--plan", action="store_true",
                        help="Mostra plano completo de treinamento em fases")


    args = parser.parse_args()

    if args.status:
        show_status()

    elif args.plan:
        print_training_plan()

    else:
        dataset = args.dataset
        phase2_organism = _normalize_cli_list(args.phase2_organism)
        phase2_description = _normalize_cli_list(args.phase2_description)
        phase3_pdb_ids = _normalize_pdb_ids(args.phase3_pdb_ids)
        phase3_target_list = _load_target_list_json(args.phase3_target_json)

        if dataset:
            log.info(f"Dataset selecionado: {dataset}")
            success = collect_external_dataset(
                dataset=dataset,
                moad_dir=args.moad_dir,
                plinder_max=args.plinder_max,
            )
            if not success:
                log.error(f"Falha ao coletar dataset '{dataset}'.")
            else:
                show_status()

        if args.phase == "2":
            collect_phase2(
                organism_terms=phase2_organism,
                description_terms=phase2_description,
            )
        elif args.phase == "3":
            collect_phase3(
                target_list=phase3_target_list,
                extra_pdb_ids=phase3_pdb_ids,
            )
        elif args.phase == "all":
            collect_phase2(
                organism_terms=phase2_organism,
                description_terms=phase2_description,
            )
            collect_phase3(
                target_list=phase3_target_list,
                extra_pdb_ids=phase3_pdb_ids,
            )
            show_status()

        if not dataset and not args.phase:
            show_status()
            print_training_plan()
