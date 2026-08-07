# -*- coding: utf-8 -*-
"""
Pipeline - Etapa 6: Diagnóstico e Avaliação do Modelo Treinado
==============================================================
Verifica a saúde do treinamento e avalia o modelo nos dados de teste.

Uso:
    python 06_evaluate.py                        # usa checkpoints/ padrão
    python 06_evaluate.py --ckpt meu_dir/        # diretório personalizado
    python 06_evaluate.py --ckpt checkpoints/ --generate  # gera ligantes novos
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Verificações de dependências ───────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors, QED
    from rdkit.Chem import rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR = PROJECT_ROOT / "data"

def _load_train_module():
    import importlib.util

    train_path = Path(__file__).resolve().parent / "pipeline_train.py"
    spec = importlib.util.spec_from_file_location("pipeline_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível criar spec para {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — ANÁLISE DO HISTÓRICO DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def analyze_training_history(ckpt_dir: Path):
    """Lê training_history.csv e extrai métricas de saúde do treinamento."""
    csv_path = ckpt_dir / "training_history.csv"
    if not csv_path.exists():
        log.error(f"Arquivo não encontrado: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    log.info(f"Histórico carregado: {len(df)} epochs")

    # ── Métricas principais ────────────────────────────────────────────────────
    last = df.iloc[-1]
    best_epoch = df.loc[df["val_total"].idxmin()]

    report = {
        "total_epochs":      len(df),
        "best_epoch":        int(best_epoch["epoch"]),
        "best_val_loss":     round(float(best_epoch["val_total"]), 4),
        "final_train_loss":  round(float(last["train_total"]), 4),
        "final_val_loss":    round(float(last["val_total"]), 4),
        "final_kl":          round(float(last.get("val_kl", 0)), 4),
        "final_recon":       round(float(last.get("val_recon", 0)), 4),
        "overfit_gap":       round(float(last["val_total"]) -
                                   float(last["train_total"]), 4),
    }

    # ── Diagnósticos ──────────────────────────────────────────────────────────
    issues = []

    # 1. KL collapse: se KL ≈ 0, o encoder não aprendeu distribuição útil
    if report["final_kl"] < 0.01:
        issues.append("⚠️  KL ≈ 0 (posterior collapse): o encoder pode estar "
                      "ignorando o espaço latente. Tente reduzir beta ou "
                      "aumentar beta_warmup.")

    # 2. KL muito alto: reconstrução sacrificada
    if report["final_kl"] > 50:
        issues.append("⚠️  KL muito alto: o modelo está priorizando regularização "
                      "sobre reconstrução. Reduza beta_max.")

    # 3. Overfitting
    if report["overfit_gap"] > 0.5:
        issues.append(f"⚠️  Overfitting detectado: gap train/val = "
                      f"{report['overfit_gap']:.3f}. "
                      "Aumente dropout ou reduza o modelo.")

    # 4. Loss não convergiu
    if len(df) >= 10:
        early_val = df["val_total"].iloc[:5].mean()
        late_val  = df["val_total"].iloc[-5:].mean()
        if late_val >= early_val * 0.99:
            issues.append("⚠️  Loss de validação não convergiu. O modelo pode "
                          "precisar de mais epochs, LR maior, ou dados.")

    # 5. Loss NaN
    if df["train_total"].isna().any() or df["val_total"].isna().any():
        issues.append("❌ NaN detectado no histórico! Verifique o LR "
                      "(provavelmente muito alto) e grad_clip.")

    report["issues"] = issues
    return report, df


def plot_training_curves(df: pd.DataFrame, ckpt_dir: Path, fig_dir: Path):
    """Plota curvas de treinamento detalhadas."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Training Diagnostics — CVAE", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    epochs = df["epoch"]

    plots = [
        ("Total Loss",       "train_total",  "val_total",   gs[0, 0]),
        ("Reconstruction",   "train_recon",  "val_recon",   gs[0, 1]),
        ("KL Divergence",    "train_kl",     "val_kl",      gs[0, 2]),
        ("Atom Loss",        "train_atom",   "val_atom",    gs[1, 0]),
        ("Bond Loss",        "train_bond",   "val_bond",    gs[1, 1]),
        ("3D Position Loss", "train_pos",    "val_pos",     gs[1, 2]),
    ]

    for title, train_col, val_col, pos in plots:
        ax = fig.add_subplot(pos)
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], label="Train", color="#4C72B0", linewidth=1.5)
        if val_col in df.columns:
            ax.plot(epochs, df[val_col],   label="Val",   color="#DD8452",
                    linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Marca melhor epoch
        if val_col in df.columns:
            best_i = df[val_col].idxmin()
            ax.axvline(df["epoch"].iloc[best_i], color="green",
                       linestyle=":", alpha=0.6, label=f"Best e{df['epoch'].iloc[best_i]}")

    for ext, kw in [("png", {"dpi": 150}), ("svg", {})]:
        out = fig_dir / f"training_curves.{ext}"
        plt.savefig(out, bbox_inches="tight", **kw)
        log.info(f"Curvas salvas em: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — AVALIAÇÃO NO CONJUNTO DE TESTE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(ckpt_dir: Path, tensor_dir: Path) -> dict:
    """Carrega o melhor modelo e avalia no conjunto de teste."""
    if not HAS_TORCH:
        log.warning("PyTorch não disponível — pulando avaliação no teste.")
        return {}

    best_ckpt = ckpt_dir / "best_model.pt"
    if not best_ckpt.exists():
        log.warning(f"best_model.pt não encontrado em {ckpt_dir}")
        return {}

    from torch.utils.data import DataLoader

    try:
        mod = _load_train_module()
        SiteLigandDataset = mod.SiteLigandDataset
        LigandCVAE = mod.LigandCVAE
        cvae_loss = mod.cvae_loss
    except Exception as e:
        log.error(f"Não foi possível importar pipeline_train.py: {e}")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega checkpoint
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    model = LigandCVAE(
        in_channels  = 5,
        latent_dim   = cfg.get("latent_dim",   256),
        hidden_dim   = cfg.get("hidden_dim",   256),
        n_gnn_layers = cfg.get("n_gnn_layers", 4),
    ).to(device)

    encoder_fc_weight = ckpt["model_state"].get("encoder.fc.0.weight")
    if encoder_fc_weight is not None:
        ckpt_in_features = int(encoder_fc_weight.shape[1])
        current_fc = model.encoder.fc[0]
        if current_fc.in_features != ckpt_in_features:
            model.encoder.fc[0] = torch.nn.Linear(
                ckpt_in_features,
                current_fc.out_features,
                bias=current_fc.bias is not None,
            ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    log.info(f"Modelo carregado do epoch {ckpt.get('epoch', '?')} "
             f"(val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Dataset de teste
    try:
        test_ds = SiteLigandDataset(tensor_dir=tensor_dir, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=4, shuffle=False,
            collate_fn=SiteLigandDataset.collate_fn,
        )
    except Exception as e:
        log.warning(f"Dataset de teste: {e}")
        return {"loaded": True, "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")}

    # Loop de avaliação
    # Inicializa todos os componentes que cvae_loss pode retornar
    totals = {k: 0.0 for k in ["total","recon","stop","atom","bond","exist","pos","charge","kl"]}
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            site        = batch["tensor"].to(device)
            graphs      = [g.to(device) for g in batch["graphs"]]
            voxel_sizes = batch["voxel_sizes"].to(device)
            preds = model(site, graphs, voxel_sizes)
            losses = cvae_loss(preds, beta=1.0)
            # Acumula apenas as chaves presentes — robusto a versões diferentes
            for k in totals:
                if k in losses and torch.isfinite(losses[k]):
                    totals[k] += losses[k].item()
            n += 1

    if n == 0:
        return {"loaded": True, "n_test_batches": 0}

    return {
        "loaded":           True,
        "best_epoch":       ckpt.get("epoch"),
        "val_loss_ckpt":    round(float(ckpt.get("val_loss", 0)), 4),
        "test_total":       round(totals["total"]  / n, 4),
        "test_recon":       round(totals["recon"]  / n, 4),
        "test_stop":        round(totals["stop"]   / n, 6),
        "test_atom":        round(totals["atom"]   / n, 6),  # 6 casas: valores ~1e-5
        "test_bond":        round(totals["bond"]   / n, 6),
        "test_exist":       round(totals["exist"]  / n, 6),
        "test_pos":         round(totals["pos"]    / n, 4),
        "test_charge":      round(totals["charge"] / n, 6),
        "test_kl":          round(totals["kl"]     / n, 4),
        "n_test_batches":   n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — GERAÇÃO E VALIDAÇÃO DE LIGANTES
# ══════════════════════════════════════════════════════════════════════════════

def _load_generate_module():
    import importlib.util
    gen_path = Path(__file__).resolve().parent / "pipeline_generate.py"
    spec = importlib.util.spec_from_file_location("pipeline_generate", gen_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível criar spec para {gen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_predictions_to_mol(candidate: dict):
    """Converte um candidato de LigandCVAE.sample() (decoder autorregressivo)
    em molécula RDKit.

    Fina wrapper sobre pipeline_generate.preds_to_mol — reaproveita a MESMA
    lógica de validação química (valência, distância, reparo de fragmentos
    desconexos) usada na geração real, em vez de manter uma cópia duplicada
    aqui que divergiria com o tempo (era exatamente o que acontecia com a
    versão anterior, escrita para o decoder de topologia fixa por template
    — já incompatível com o formato de saída atual).
    """
    if not HAS_RDKIT:
        return None
    try:
        gen = _load_generate_module()
        return gen.preds_to_mol(candidate)
    except Exception as e:
        log.warning(f"  decode_predictions_to_mol: {e}")
        return None


def compute_molecule_metrics(smiles_list) -> dict:
    """Calcula métricas de qualidade de moléculas geradas.

    Métricas padrão em drug design generativo:
      - Validity    : % de SMILES que são moléculas válidas
      - Uniqueness  : % de moléculas únicas entre as válidas
      - QED         : Drug-likeness (0-1, maior = mais drug-like)
      - MW          : Peso molecular médio
      - LogP        : Coeficiente de partição (lipofilia)
      - SA Score    : Facilidade de síntese (1=fácil, 10=difícil)
      - Lipinski    : % que passa na Regra dos 5 de Lipinski
    """
    if not HAS_RDKIT or not smiles_list:
        return {}

    valid_mols = []
    for smi in smiles_list:
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_mols.append((smi, mol))

    validity = len(valid_mols) / len(smiles_list) if smiles_list else 0
    unique_smiles = set(s for s, _ in valid_mols)
    uniqueness = len(unique_smiles) / len(valid_mols) if valid_mols else 0

    if not valid_mols:
        return {"validity": 0.0, "uniqueness": 0.0, "n_valid": 0}

    qed_scores, mws, logps, lipinski_pass = [], [], [], []

    for smi, mol in valid_mols:
        try:
            qed_scores.append(QED.qed(mol))
            mws.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            mw   = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd  = rdMolDescriptors.CalcNumHBD(mol)
            hba  = rdMolDescriptors.CalcNumHBA(mol)
            lipinski_pass.append(
                mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
            )
        except Exception:
            pass

    return {
        "n_total":     len(smiles_list),
        "n_valid":     len(valid_mols),
        "validity":    round(validity, 3),
        "uniqueness":  round(uniqueness, 3),
        "qed_mean":    round(np.mean(qed_scores), 3)    if qed_scores else None,
        "qed_std":     round(np.std(qed_scores), 3)     if qed_scores else None,
        "mw_mean":     round(np.mean(mws), 1)           if mws else None,
        "logp_mean":   round(np.mean(logps), 2)         if logps else None,
        "lipinski_%":  round(np.mean(lipinski_pass)*100, 1) if lipinski_pass else None,
    }


def save_generated_sdf(generated_records, out_sdf: Path, real_smiles: str):
    """Salva os ligantes gerados com coordenadas 3D em formato SDF."""
    writer = Chem.SDWriter(str(out_sdf))
    for idx, record in enumerate(generated_records, start=1):
        mol = record.get("mol")
        if mol is None:
            continue
        mol.SetProp("_Name", f"generated_{idx}")
        mol.SetProp("SMILES", record.get("smiles", ""))
        if real_smiles:
            mol.SetProp("Reference_SMILES", real_smiles)
        writer.write(mol)
    writer.close()
    log.info(f"\n  3D SDF saved at: {out_sdf}")


def generate_ligands_for_site(
    ckpt_dir: Path,
    tensor_dir: Path,
    n_samples: int = 20,
    temperature: float = 1.0,
):
    """Gera ligantes para os sítios do conjunto de teste e avalia qualidade."""
    if not HAS_TORCH:
        return []

    try:
        mod = _load_train_module()
        SiteLigandDataset = mod.SiteLigandDataset
        LigandCVAE = mod.LigandCVAE
    except Exception as e:
        log.error(f"Erro ao importar pipeline_train.py: {e}")
        return []

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_ckpt = ckpt_dir / "best_model.pt"
    ckpt      = torch.load(best_ckpt, map_location=device, weights_only=False)
    cfg       = ckpt.get("config", {})

    model = LigandCVAE(
        in_channels=5,
        latent_dim=cfg.get("latent_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_gnn_layers=cfg.get("n_gnn_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Pega um par do teste como sítio de referência (só o tensor do sítio —
    # o decoder autorregressivo não usa mais nenhum "template graph")
    test_ds = SiteLigandDataset(tensor_dir=tensor_dir, split="test")
    from torch.utils.data import DataLoader
    loader      = DataLoader(test_ds, batch_size=1, shuffle=True,
                             collate_fn=SiteLigandDataset.collate_fn)
    batch       = next(iter(loader))
    if batch is None:
        return []

    site        = batch["tensor"].to(device)
    real_smiles = batch["smiles"][0]

    log.info(f"Sítio de referência: {batch['pair_ids'][0]}")
    log.info(f"Ligante real (cristal): {real_smiles}")
    log.info(f"Gerando {n_samples} ligantes (temperatura={temperature})...")

    # Amostragem — construção sequencial em lockstep (ver LigandCVAE.sample)
    generated_records = []
    voxel_sizes = batch["voxel_sizes"].to(device)
    with torch.no_grad():
        samples = model.sample(site,
                               n_samples=n_samples,
                               temperature=temperature,
                               voxel_sizes=voxel_sizes)
        for candidate in samples:
            mol = decode_predictions_to_mol(candidate)
            if mol:
                smi = Chem.MolToSmiles(mol)
                generated_records.append({"smiles": smi, "mol": mol})
            else:
                generated_records.append({"smiles": "", "mol": None})

    return real_smiles, generated_records


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    sep = "═" * 60
    log.info(f"\n{sep}")
    log.info(f"  {title}")
    log.info(sep)


def plot_performance_report(
    df,
    test_results,
    gen_metrics,
    real_smiles,
    gen_smiles,
    ckpt_dir,
    fig_dir,
):
    """Painel de desempenho: 3 paineis lado a lado.

    [1] Barras — componentes da loss no conjunto de teste
    [2] Histograma — distribuicao de QED das moleculas geradas
    [3] Tabela — metricas de geracao

    Salva: performance_report.png e performance_report.svg
    Nao repete os graficos de treinamento (ja cobertos por training_curves).
    """
    C_BAR  = "#5B9BD5"
    C_QED  = "#55A868"
    C_MEAN = "#C44E52"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"wspace": 0.38})

    _best_vl  = df["val_total"].min() if "val_total" in df.columns else 0
    _test_tot = test_results.get("test_total", "N/A")
    fig.suptitle(
        "CODEEP - Performance | Val loss: " + str(round(float(_best_vl), 4)) +
        " | Test loss: " + str(_test_tot),
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Painel 1: barras dos componentes de teste
    ax1 = axes[0]
    components = {
        "Total":   test_results.get("test_total",  0) or 0,
        "Reconstruction":   test_results.get("test_recon",  0) or 0,
        "3D Position": test_results.get("test_pos",    0) or 0,
        "KL":      test_results.get("test_kl",     0) or 0,
        "Atom":   test_results.get("test_atom",   0) or 0,
        "Charge":   test_results.get("test_charge", 0) or 0,
    }
    vals  = list(components.values())
    bars  = ax1.bar(list(components.keys()), vals,
                    color=C_BAR, edgecolor="white", width=0.55)
    ymax  = max(vals) * 1.18 if max(vals) > 0 else 1
    ax1.set_ylim(0, ymax)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + ymax * 0.02,
                 str(round(val, 4)), ha="center", va="bottom", fontsize=8)
    ax1.set_title("Test Loss Components", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", labelsize=8, rotation=15)
    ax1.tick_params(axis="y", labelsize=8)

    # Panel 2: QED histogram
    ax2 = axes[1]
    qed_vals = []
    if HAS_RDKIT and gen_smiles:
        for smi in gen_smiles:
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    try:
                        qed_vals.append(QED.qed(mol))
                    except Exception:
                        pass

    if qed_vals:
        ax2.hist(qed_vals, bins=min(12, len(qed_vals)),
                 color=C_QED, edgecolor="white", alpha=0.85)
        mu = float(np.mean(qed_vals))
        ax2.axvline(mu, color=C_MEAN, lw=2, ls="--",
                    label="Mean = " + str(round(mu, 3)))
        ax2.set_xlim(0, 1)
        ax2.legend(fontsize=9)
    else:
        msg = "Run with --generate" if not gen_smiles else "No valid molecules"
        ax2.text(0.5, 0.5, msg, ha="center", va="center",
                 fontsize=10, color="gray", transform=ax2.transAxes)
    ax2.set_title("QED Distribution (Generated)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("QED", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

    # Panel 3: generation metrics table
    ax3 = axes[2]
    ax3.axis("off")
    if gen_metrics and gen_metrics.get("n_valid", 0) > 0:
        n_val = gen_metrics.get("n_valid", 0)
        n_tot = gen_metrics.get("n_total", 0)
        rows = [
            ["Generated",    str(n_tot)],
            ["Valid",        str(n_val) + " (" + str(round(gen_metrics.get("validity",0)*100)) + "%)"],
            ["Unique",       str(round(gen_metrics.get("uniqueness",0)*100)) + "%"],
            ["Mean QED",     str(gen_metrics.get("qed_mean","N/A")) + " +/- " + str(gen_metrics.get("qed_std","N/A"))],
            ["Mean MW",      str(gen_metrics.get("mw_mean","N/A")) + " Da"],
            ["Mean LogP",    str(gen_metrics.get("logp_mean", "N/A"))],
            ["Lipinski",     str(gen_metrics.get("lipinski_%","N/A")) + "%"],
        ]
        tbl = ax3.table(cellText=rows, colLabels=["Metric", "Value"],
                        cellLoc="center", loc="center",
                        bbox=[0.0, 0.05, 1.0, 0.92])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            cell.set_height(0.125)
            if row == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F0F4F8")
    else:
        ax3.text(0.5, 0.5, "Run with --generate to see metrics",
                 ha="center", va="center", fontsize=10, color="gray",
                 transform=ax3.transAxes)
    ax3.set_title("Generation Metrics", fontsize=10, fontweight="bold")

    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("png", {"dpi": 150}), ("svg", {})]:
        out = fig_dir / ("performance_report." + ext)
        plt.savefig(out, bbox_inches="tight", **kw)
        log.info(f"Performance report saved at: {out}")
    plt.close()

def main(ckpt_dir: Path, do_generate: bool, tensor_dir: Path, fig_dir: Path):
    

    print_section("1. TRAINING HISTORY ANALYSIS")

    result = analyze_training_history(ckpt_dir)
    if not result:
        return
    report, df = result

    log.info(f"  Trained epochs      : {report['total_epochs']}")
    log.info(f"  Best epoch          : {report['best_epoch']}")
    log.info(f"  Best val_loss       : {report['best_val_loss']}")
    log.info(f"  Final val loss      : {report['final_val_loss']}")
    log.info(f"  Final train loss    : {report['final_train_loss']}")
    log.info(f"  Overfitting gap     : {report['overfit_gap']:+.4f}")
    log.info(f"  Final KL (val)      : {report['final_kl']}")
    log.info(f"  Final recon (val)   : {report['final_recon']}")

    if report["issues"]:
        log.info("\n  DIAGNOSTICS:")
        for issue in report["issues"]:
            log.info(f"    {issue}")
    else:
        log.info("\n  ✓ No issues detected during training.")

    plot_training_curves(df, ckpt_dir, fig_dir)
    

    print_section("2. TEST SET EVALUATION")

    test_results = evaluate_on_test(ckpt_dir, tensor_dir)
    if test_results:
        for k, v in test_results.items():
            log.info(f"  {k:<25}: {v}")

        # Generalization: compare val_loss from checkpoint with test_loss
        if "val_loss_ckpt" in test_results and "test_total" in test_results:
            gap = test_results["test_total"] - test_results["val_loss_ckpt"]
            if abs(gap) < 0.1:
                log.info("\n  ✓ Good generalization: val ≈ test")
            elif gap > 0.3:
                log.info(f"\n  ⚠️  Test loss ({test_results['test_total']}) "
                         f"> Val loss ({test_results['val_loss_ckpt']}): "
                         "possible data leakage or small dataset.")

    if do_generate:
        print_section("3. GENERATION OF NEW LIGANDS")

        real_smi, generated_records = generate_ligands_for_site(
            ckpt_dir=ckpt_dir,
            tensor_dir=tensor_dir,
            n_samples=20,
            temperature=1.0,
        )

        gen_smiles = [record["smiles"] for record in generated_records]

        metrics = compute_molecule_metrics(gen_smiles)
        log.info(f"  Real crystal ligand: {real_smi}")
        log.info(f"  Generated molecules     : {metrics.get('n_total', 0)}")
        log.info(f"  Valid                   : {metrics.get('n_valid', 0)} "
                 f"({metrics.get('validity', 0)*100:.1f}%)")
        log.info(f"  Unique                  : {metrics.get('uniqueness', 0)*100:.1f}%")
        log.info(f"  Mean QED                : {metrics.get('qed_mean', 'N/A')} "
                 f"(± {metrics.get('qed_std', 'N/A')})")
        log.info(f"  Mean MW                 : {metrics.get('mw_mean', 'N/A')} Da")
        log.info(f"  Mean LogP               : {metrics.get('logp_mean', 'N/A')}")
        log.info(f"  Lipinski Rule           : {metrics.get('lipinski_%', 'N/A')}%")

        out_sdf = ckpt_dir / "generated_ligands_3d.sdf"
        save_generated_sdf(generated_records, out_sdf, real_smi)

        # Save generated SMILES
        out_smi = ckpt_dir / "generated_smiles.smi"
        with open(out_smi, "w") as f:
            f.write(f"# Real crystal ligand: {real_smi}\n")
            for smi in gen_smiles:
                if smi:
                    f.write(smi + "\n")
        log.info(f"\n  SMILES saved at: {out_smi}")

    # Generate full performance report
    _gen_metrics = metrics     if (do_generate and "metrics"    in dir()) else {}
    _gen_smiles  = gen_smiles  if (do_generate and "gen_smiles" in dir()) else []
    _real_smi    = real_smi    if (do_generate and "real_smi"   in dir()) else ""
    plot_performance_report(
        df=df,
        test_results=test_results or {},
        gen_metrics=_gen_metrics,
        real_smiles=_real_smi,
        gen_smiles=_gen_smiles,
        ckpt_dir=ckpt_dir,
        fig_dir=fig_dir,
    )

    print_section("SUMMARY AND NEXT STEPS")

    # Recommendation logic based on diagnostics
    issues = report.get("issues", [])
    kl     = report["final_kl"]
    gap    = report["overfit_gap"]

    if not issues:
        log.info("  ✓ Healthy training!")
        log.info("  Suggested next steps:")
        log.info("    1. Run with --generate to evaluate generated molecules")
        log.info("    2. Increase the dataset (more ABC transporter proteins)")
        log.info("    3. Try higher QM level: 3-21G or DFT/B3LYP")
        log.info("    4. Fine-tuning: train more epochs with lower LR (1e-5)")
    else:
        log.info("  Recommended actions based on diagnostics:")
        if kl < 0.01:
            log.info("    → Posterior collapse: reduce --beta to 0.1 and increase")
            log.info("      --beta-warmup to 40 in the next training")
        if gap > 0.5:
            log.info("    → Overfitting: increase --dropout to 0.3 or")
            log.info("      collect more training data")
        if "NaN" in str(issues):
            log.info("    → NaN: reduce --lr to 1e-5 and add --grad-clip 0.5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "phase1"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--tensor-dir",
        type=str,
        default=str(BASE_DIR / "tensors"),
        help="Tensor directory for the phase to evaluate",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(BASE_DIR / "figures"),
        help="Directory where the figures will be saved",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new ligands and evaluate chemical quality",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt)
    tensor_dir = Path(args.tensor_dir)
    fig_dir = Path(args.fig_dir)

    if not ckpt_dir.exists():
        log.error(f"Directory not found: {ckpt_dir}")
        log.error("Check the path with --ckpt <directory>")
        raise SystemExit(1)

    main(
        ckpt_dir=ckpt_dir,
        do_generate=args.generate,
        tensor_dir=tensor_dir,
        fig_dir=fig_dir,
    )
