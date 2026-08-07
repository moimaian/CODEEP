"""
Pipeline - Etapa 3: Análise Exploratória da Base de Dados
=========================================================
Gera estatísticas, distribuições e um relatório visual
da base de dados curada proteína-ligante do job atual.

Requisitos:
    pip install pandas matplotlib seaborn

Uso:
    python 03_explore_database.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR     = PROJECT_DIR / "data"
CUR_DIR      = BASE_DIR / "curated"
FIG_DIR      = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})


def _save_fig(fig: plt.Figure, stem: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = fig_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Figura salva em: {out}")


def load_curated_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata não encontrada: {metadata_csv}")
    return pd.read_csv(metadata_csv)


def plot_summary(df: pd.DataFrame, fig_dir: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Curated Database\nExploratory Analysis",
                fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Distribuição de pesos moleculares
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["approx_mw"], bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Approx. Molecular Weight (Da)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Ligand Molecular Weight Distribution")
    ax1.axvline(df["approx_mw"].median(), color="#DD8452", linestyle="--",
                label=f"Median: {df['approx_mw'].median():.0f} Da")
    ax1.legend(fontsize=8)

    # 2. Distribuição de átomos pesados
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["n_heavy_atoms"], bins=25, color="#55A868", edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Heavy Atoms Number")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Ligand Size Distribution")
    ax2.axvline(df["n_heavy_atoms"].median(), color="#DD8452", linestyle="--",
                label=f"Median: {df['n_heavy_atoms'].median():.0f}")
    ax2.legend(fontsize=8)

    # 3. Distribuição de resíduos no sítio
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["binding_site_residues"], bins=20, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("Number of Binding Site Residues")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Binding Site Size Distribution")
    ax3.axvline(df["binding_site_residues"].median(), color="#DD8452", linestyle="--",
                label=f"Median: {df['binding_site_residues'].median():.0f}")
    ax3.legend(fontsize=8)

    # 4. Top 15 ligantes mais frequentes
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_ligs = df["ligand_name"].value_counts().head(15)
    colors = sns.color_palette("Blues_d", len(top_ligs))
    bars = ax4.barh(top_ligs.index[::-1], top_ligs.values[::-1], color=colors)
    ax4.set_xlabel("Structures Count")
    ax4.set_title("Top 15 Most Frequent Ligands")
    for bar, val in zip(bars, top_ligs.values[::-1]):
        ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=8)

    # 5. Summary Statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    stats = [
        ("Curated Site-Ligand Pairs", len(df)),
        ("Unique PDBs", df["pdb_id"].nunique()),
        ("Unique Ligands", df["ligand_name"].nunique()),
        ("MMedian MW (Da)", f"{df['approx_mw'].median():.1f}"),
        ("Min / Max MW (Da)", f"{df['approx_mw'].min():.0f} / {df['approx_mw'].max():.0f}"),
        ("Binding Site Residues (med.)", f"{df['binding_site_residues'].median():.0f}"),
        ("Heavy Atoms (med.)", f"{df['n_heavy_atoms'].median():.0f}"),
    ]
    table_data = [[k, str(v)] for k, v in stats]
    table = ax5.table(cellText=table_data, colLabels=["Metric", "Value"],
                      cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2F7")
    ax5.set_title("Summary Statistics", fontweight="bold")

    _save_fig(fig, "summary_dashboard", fig_dir)
    plt.close()


def print_text_report(df: pd.DataFrame):
    sep = "─" * 55
    print(f"\n{sep}")
    print("  REPORT — CURATED PROTEIN-LIGAND DATABASE")
    print(sep)
    print(f"  Curated Site-Ligand Pairs : {len(df)}")
    print(f"  Unique PDBs               : {df['pdb_id'].nunique()}")
    print(f"  Unique Ligands            : {df['ligand_name'].nunique()}")
    print(f"\n  Most Frequent Ligands:")
    for name, count in df["ligand_name"].value_counts().head(10).items():
        print(f"    {name:<8} — {count} structures")
    print(f"\n  Ligand Molecular Weight:")
    print(f"    Minimum  : {df['approx_mw'].min():.1f} Da")
    print(f"    Median   : {df['approx_mw'].median():.1f} Da")
    print(f"    Maximum  : {df['approx_mw'].max():.1f} Da")
    print(f"\n  Binding Sites:")
    print(f"    Residues (med.) : {df['binding_site_residues'].median():.0f}")
    print(f"    Residues (min)  : {df['binding_site_residues'].min()}")
    print(f"    Residues (max)  : {df['binding_site_residues'].max()}")
    print(sep + "\n")


def list_pair_dirs(curated_dir: Path) -> list[Path]:
    """Lists all curated pair directories."""
    return sorted([d for d in curated_dir.iterdir() if d.is_dir()])


def main(
    curated_dir: "Path | str | None" = None,
    metadata_csv: "Path | str | None" = None,
    fig_dir: "Path | str | None" = None,
):
    curated_dir = Path(curated_dir) if curated_dir else BASE_DIR / "curated"
    metadata_csv = Path(metadata_csv) if metadata_csv else BASE_DIR / "curated_metadata.csv"
    fig_dir = Path(fig_dir) if fig_dir else BASE_DIR / "figures"

    df = load_curated_metadata(metadata_csv)
    print_text_report(df)
    plot_summary(df, fig_dir)

    pairs = list_pair_dirs(curated_dir)
    print(f"Diretórios de pares curados: {len(pairs)}")
    print("Exemplos:")
    for p in pairs[:5]:
        files = [f.name for f in p.iterdir()]
        print(f"  {p.name}/  →  {files}")


if __name__ == "__main__":
    main()
