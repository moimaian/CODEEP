"""
Pipeline - Benchmark multialvo do CODEEP
========================================

Executa um benchmark de geracao sobre um conjunto de alvos estruturais,
reutilizando o pipeline de geracao do CODEEP e exportando resultados em um
formato padrao por candidato, por alvo e agregado.

Uso:
    python MODULES/pipeline_benchmark.py --config TEST/pilot_crossdocked10_config.json
"""

import argparse
import csv
import importlib.util
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))


def _load_local_module(module_name: str):
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_local_module("pipeline_generate")
eval_mod = _load_local_module("pipeline_evaluate")


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if getattr(gen, "HAS_TORCH", False):
        gen.torch.manual_seed(seed)
        if gen.torch.cuda.is_available():
            gen.torch.cuda.manual_seed_all(seed)


def _load_train_smiles(tensor_dir: Path) -> set[str]:
    try:
        mod = gen.importlib.util.spec_from_file_location(
            "pipeline_train", Path(__file__).resolve().parent / "pipeline_train.py"
        )
        train_mod = gen.importlib.util.module_from_spec(mod)
        mod.loader.exec_module(train_mod)
        train_ds = train_mod.SiteLigandDataset(tensor_dir=tensor_dir, split="train")
        smiles = set()
        for smi in train_ds.records["ligand_smiles"].dropna().tolist():
            mol = gen.Chem.MolFromSmiles(str(smi))
            if mol is not None:
                smiles.add(gen.Chem.MolToSmiles(mol))
        return smiles
    except Exception as e:
        log.warning(f"Nao foi possivel carregar SMILES de treino para novelty: {e}")
        return set()


def _target_id_from_protein_path(protein_rel: str) -> str:
    stem = Path(protein_rel).stem
    return stem[:-9] if stem.endswith("_pocket10") else stem


def _load_targets(config: dict) -> list[dict]:
    if config.get("targets"):
        targets = []
        for item in config["targets"]:
            targets.append({
                "target_id": item.get("target_id") or Path(item["pdb"]).stem,
                "pdb": _resolve_path(item["pdb"]),
                "ligand_ref": _resolve_path(item.get("ligand_ref")),
                "site_pdb": _resolve_path(item.get("site_pdb")),
            })
        return targets

    targets_tsv = _resolve_path(config.get("targets_tsv"))
    dataset_root = _resolve_path(config.get("dataset_root"))
    if targets_tsv is None or dataset_root is None:
        raise ValueError("Config deve definir `targets` ou `targets_tsv` + `dataset_root`.")
    if not targets_tsv.exists():
        raise FileNotFoundError(f"Lista de alvos nao encontrada: {targets_tsv}")

    targets = []
    with open(targets_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            protein_rel = row.get("Protein", "").strip()
            ligand_rel = row.get("Ligand", "").strip()
            if not protein_rel:
                continue
            targets.append({
                "target_id": _target_id_from_protein_path(protein_rel),
                "pdb": dataset_root / protein_rel,
                "ligand_ref": dataset_root / ligand_rel if ligand_rel else None,
                "site_pdb": None,
            })
    return targets


def _summarize_run(
    benchmark_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    train_smiles: set[str],
    target_id: str,
    run_seed: int,
    method_name: str,
    n_generated: int,
    generation_time_s: float,
) -> dict:
    valid_df = benchmark_df[benchmark_df["valid"] == True].copy()
    unique_df = benchmark_df[benchmark_df["unique"] == True].copy()

    smiles_for_metrics = [s for s in unique_df.get("smiles", pd.Series(dtype=str)).dropna().tolist() if s]
    chem_metrics = eval_mod.compute_molecule_metrics(smiles_for_metrics, train_smiles=train_smiles)

    no_clash_fraction = None
    if "has_pocket_clash" in unique_df.columns and not unique_df.empty:
        clash_series = unique_df["has_pocket_clash"].dropna()
        if not clash_series.empty:
            no_clash_fraction = float((~clash_series.astype(bool)).mean())

    summary = {
        "target_id": target_id,
        "run_seed": run_seed,
        "method_name": method_name,
        "n_generated": n_generated,
        "n_valid": int(len(valid_df)),
        "n_unique": int(len(unique_df)),
        "validity": round(len(valid_df) / max(n_generated, 1), 4),
        "uniqueness": round(len(unique_df) / max(len(valid_df), 1), 4) if len(valid_df) else 0.0,
        "novelty": round(float(unique_df["novel"].dropna().mean()), 4) if "novel" in unique_df and unique_df["novel"].dropna().size else None,
        "internal_diversity": chem_metrics.get("internal_diversity"),
        "qed_mean": chem_metrics.get("qed_mean"),
        "sa_score_mean": chem_metrics.get("sa_score_mean"),
        "mw_mean": chem_metrics.get("mw_mean"),
        "logp_mean": chem_metrics.get("logp_mean"),
        "lipinski_fraction": round(float(unique_df["lipinski"].fillna(False).astype(bool).mean()), 4) if "lipinski" in unique_df and not unique_df.empty else None,
        "veber_fraction": round(float(unique_df["veber"].fillna(False).astype(bool).mean()), 4) if "veber" in unique_df and not unique_df.empty else None,
        "no_clash_fraction": round(no_clash_fraction, 4) if no_clash_fraction is not None else None,
        "top1_score": round(float(ranked_df["score"].iloc[0]), 4) if not ranked_df.empty else None,
        "top10_score_mean": round(float(ranked_df["score"].head(10).mean()), 4) if not ranked_df.empty else None,
        "generation_time_s": round(generation_time_s, 3),
    }
    return summary


def _run_single_target(
    model,
    mod,
    ckpt_voxel_config: dict,
    target: dict,
    config: dict,
    output_root: Path,
    train_smiles: set[str],
    run_seed: int,
) -> tuple[pd.DataFrame, dict]:
    target_id = target["target_id"]
    pdb_path = Path(target["pdb"])
    ligand_ref = Path(target["ligand_ref"]) if target.get("ligand_ref") else None
    site_pdb = Path(target["site_pdb"]) if target.get("site_pdb") else None

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB do alvo nao encontrado: {pdb_path}")
    if ligand_ref is not None and not ligand_ref.exists():
        raise FileNotFoundError(f"Ligante de referencia nao encontrado: {ligand_ref}")
    if site_pdb is not None and not site_pdb.exists():
        raise FileNotFoundError(f"Site PDB nao encontrado: {site_pdb}")

    run_dir = output_root / target_id / f"seed_{run_seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(run_seed)
    site_overrides = gen.resolve_site_config_overrides(
        ckpt_voxel_config,
        config.get("site_config_overrides") or {},
    )

    t0 = time.perf_counter()
    (site_tensor, origin, voxel_size, site_coords, site_elements,
     site_mol, site_is_backbone) = gen.pdb_to_site_tensor(
        pdb_path,
        config.get("cutoff", 6.0),
        config.get("min_atoms", 5),
        site_pdb=site_pdb,
        config_overrides=site_overrides,
    )
    np.savez_compressed(
        run_dir / "site_tensor.npz",
        tensor=site_tensor,
        origin=origin,
        voxel_size=np.array([voxel_size]),
    )

    ref_mol = None
    if ligand_ref is not None:
        ref_mol, ref_sdf_path = gen.load_reference_ligand(ligand_ref, run_dir)
        if ref_sdf_path is not None:
            ligand_ref = ref_sdf_path

    candidates = gen.generate_candidates(
        model,
        mod,
        site_tensor,
        tensor_dir=_resolve_path(config.get("tensor_dir")) or (PROJECT_ROOT / "data" / "tensors"),
        n_samples=config.get("n_samples", 50),
        temperature=config.get("temperature", 1.0),
        voxel_size=voxel_size,
        site_coords=site_coords,
        site_elements=site_elements,
        site_mol=site_mol,
        site_is_backbone=site_is_backbone,
        final_ff=config.get("final_ff", "mmff_uff"),
        induced_fit=bool(config.get("induced_fit", False)),
        ligand_charge_method=config.get("ligand_charge_method", "gas"),
    )
    generation_time_s = time.perf_counter() - t0

    ranked_df = gen.rank_candidates(candidates, ref_mol=ref_mol)
    ranked_df.drop(columns=["_mol"], errors="ignore").to_csv(run_dir / "candidates.csv")
    benchmark_df = gen.export_benchmark_candidates(
        candidates=candidates,
        ranked_df=ranked_df,
        out_path=run_dir / "benchmark_candidates.csv",
        target_id=target_id,
        run_seed=run_seed,
        method_name=config.get("method_name", "CODEEP"),
        n_generated=config.get("n_samples", 50),
        generation_time_s=generation_time_s,
        novelty_reference_smiles=train_smiles,
    )

    if not ranked_df.empty:
        gen.save_sdf(ranked_df, run_dir / f"top{config.get('top_n', 20)}_candidates.sdf", config.get("top_n", 20))
        gen.save_grid_image(ranked_df, run_dir / f"top{config.get('top_n', 20)}_candidates.png", config.get("top_n", 20))

    summary = _summarize_run(
        benchmark_df=benchmark_df,
        ranked_df=ranked_df,
        train_smiles=train_smiles,
        target_id=target_id,
        run_seed=run_seed,
        method_name=config.get("method_name", "CODEEP"),
        n_generated=config.get("n_samples", 50),
        generation_time_s=generation_time_s,
    )

    meta = {
        "target_id": target_id,
        "run_seed": run_seed,
        "pdb": str(pdb_path),
        "ligand_ref": str(ligand_ref) if ligand_ref else None,
        "site_pdb": str(site_pdb) if site_pdb else None,
        "voxel_size": float(voxel_size),
        "generation_time_s": round(generation_time_s, 3),
        "summary": summary,
    }
    with open(run_dir / "benchmark_run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return benchmark_df, summary


def run_benchmark(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    targets = _load_targets(config)
    output_root = _resolve_path(config.get("output_dir")) or (PROJECT_ROOT / "results" / "benchmarks" / config.get("benchmark_name", "benchmark"))
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = _resolve_path(config.get("dataset_root"))
    if dataset_root is not None and not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root nao encontrado: {dataset_root}. Extraia o arquivo do conjunto piloto antes de rodar o benchmark."
        )

    ckpt_dir = _resolve_path(config.get("ckpt")) or (PROJECT_ROOT / "checkpoints" / "phase1")
    tensor_dir = _resolve_path(config.get("tensor_dir")) or (PROJECT_ROOT / "data" / "tensors")
    train_smiles = _load_train_smiles(tensor_dir)

    model, mod, ckpt_voxel_config = gen.load_model(ckpt_dir, project_dir=PROJECT_ROOT)
    run_seeds = config.get("run_seeds") or [0]

    all_candidate_frames = []
    all_summaries = []

    log.info(f"Benchmark: {config.get('benchmark_name', 'benchmark')}")
    log.info(f"Targets   : {len(targets)}")
    log.info(f"Seeds     : {run_seeds}")
    log.info(f"Output    : {output_root}")

    for target in targets:
        for run_seed in run_seeds:
            log.info(f"\n[Target {target['target_id']}] seed={run_seed}")
            bench_df, summary = _run_single_target(
                model=model,
                mod=mod,
                ckpt_voxel_config=ckpt_voxel_config,
                target=target,
                config=config,
                output_root=output_root,
                train_smiles=train_smiles,
                run_seed=run_seed,
            )
            all_candidate_frames.append(bench_df)
            all_summaries.append(summary)

    if all_candidate_frames:
        pd.concat(all_candidate_frames, ignore_index=True).to_csv(
            output_root / "all_candidates.csv", index=False
        )
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(output_root / "target_runs_summary.csv", index=False)
        aggregate = {
            "benchmark_name": config.get("benchmark_name", "benchmark"),
            "method_name": config.get("method_name", "CODEEP"),
            "n_targets": len(targets),
            "n_runs": len(all_summaries),
            "validity_mean": round(float(summary_df["validity"].mean()), 4),
            "uniqueness_mean": round(float(summary_df["uniqueness"].mean()), 4),
            "novelty_mean": round(float(summary_df["novelty"].dropna().mean()), 4) if summary_df["novelty"].dropna().size else None,
            "internal_diversity_mean": round(float(summary_df["internal_diversity"].dropna().mean()), 4) if summary_df["internal_diversity"].dropna().size else None,
            "qed_mean": round(float(summary_df["qed_mean"].dropna().mean()), 4) if summary_df["qed_mean"].dropna().size else None,
            "sa_score_mean": round(float(summary_df["sa_score_mean"].dropna().mean()), 4) if summary_df["sa_score_mean"].dropna().size else None,
            "no_clash_fraction_mean": round(float(summary_df["no_clash_fraction"].dropna().mean()), 4) if summary_df["no_clash_fraction"].dropna().size else None,
            "top1_score_mean": round(float(summary_df["top1_score"].dropna().mean()), 4) if summary_df["top1_score"].dropna().size else None,
            "top10_score_mean": round(float(summary_df["top10_score_mean"].dropna().mean()), 4) if summary_df["top10_score_mean"].dropna().size else None,
            "generation_time_s_mean": round(float(summary_df["generation_time_s"].mean()), 4),
        }
        with open(output_root / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner de benchmark multialvo do CODEEP")
    parser.add_argument("--config", required=True, help="Arquivo JSON de configuracao do benchmark")
    args = parser.parse_args()
    run_benchmark(Path(args.config))