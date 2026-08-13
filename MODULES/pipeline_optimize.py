# -*- coding: utf-8 -*-

# SPDX-License-Identifier: GPL-3.0-or-later
#
# CODEEP – Deep Learning Pipeline for Ligand Generation
# Módulo: Otimização de Hiperparâmetros (Etapa 6)
# Copyright (C) 2024–2026 Moisés Maia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
pipeline_optimize.py
====================
Etapa 6 — Otimização de hiperparâmetros (HPO) do CVAE do CODEEP.

Motivação
---------
Um treino completo de Fase 1 custa ~19 min/epoch × 150 epochs ≈ 30 h. Grid
search ou random search sobre runs completos é inviável (20 configurações =
25 dias). Este módulo torna a busca praticável com três alavancas:

  1. **Subamostragem do dataset** — um subconjunto por CLUSTER PDB (nunca por
     par isolado, para não quebrar a separação treino/val por cluster que o
     SiteLigandDataset faz). 20% do dataset → ~3,8 min/epoch.
  2. **Poucos epochs por trial** — 12–15 epochs bastam para ranquear
     configurações; a ordem relativa entre boas e más se estabelece cedo.
  3. **Poda (ASHA / SuccessiveHalving)** — trials ruins são interrompidos nas
     primeiras epochs. Na prática ~60-70% morrem cedo, o que multiplica por
     ~3 o número de configurações exploradas no mesmo tempo de GPU.

O resultado é um *screen* barato: use-o para reduzir de ~40 configurações
para as 3 melhores, e então rode essas 3 no dataset completo (Etapa 7) para
decidir de fato. O `val_recon` correlaciona positivamente com a qualidade de
geração, mas não é substituto dela.

Objetivo otimizado
------------------
`val_recon` quando `beta_max` ou `beta_warmup` estão no espaço de busca,
`val_total` caso contrário. Motivo: `total = recon + β·KL`; se β varia entre
trials, quem sorteia β menor ganha vantagem no total sem ser melhor modelo.
Ver `resolve_objective_metric()`.

Uso pela linha de comando
-------------------------
    python MODULES/pipeline_optimize.py \\
        --tensor-dir data/tensors --n-trials 40 --trial-epochs 15 \\
        --subset-frac 0.20 --params lr,beta_max,pos_weight,dropout

    # lista os hiperparâmetros disponíveis e seus defaults
    python MODULES/pipeline_optimize.py --list-params

O mesmo código roda pela interface (Etapa 6), que apenas monta o dicionário
`search_space` a partir dos checkboxes e caixas de texto.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import sys
import time as _time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# Import do pacote MODULES funciona tanto via `import MODULES.pipeline_optimize`
# (interface) quanto via `python MODULES/pipeline_optimize.py` (terminal).
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ESPAÇO DE BUSCA
# ══════════════════════════════════════════════════════════════════════════════
#
# `priority` define o estado inicial do checkbox na interface:
#   high/medium → marcado    low → desmarcado
#
# `kind`:
#   float_log   → suggest_float(low, high, log=True)   valor: "low,high"
#   float       → suggest_float(low, high)             valor: "low,high"
#   int         → suggest_int(low, high)               valor: "low,high"
#   categorical → suggest_categorical([...])           valor: "v1,v2,v3"
#
# `train_kw` é o nome do argumento correspondente em pipeline_train.train().

SEARCH_SPACE_SPEC: list[dict] = [
    # ── Alta prioridade ───────────────────────────────────────────────────
    {
        "name": "lr", "train_kw": "lr", "priority": "high",
        "kind": "float_log", "default": (1e-4, 1e-3), "cast": float,
        "help": "Taxa de aprendizado do AdamW. O hiperparâmetro mais "
                "impactante; busque sempre em escala logarítmica.",
    },
    {
        "name": "beta_max", "train_kw": "beta_max", "priority": "high",
        "kind": "float_log", "default": (0.1, 2.0), "cast": float,
        "help": "Peso máximo do termo KL. Controla o trade-off reconstrução "
                "vs. regularização e é o principal culpado por posterior "
                "collapse (KL → 0, decoder ignora z, geração perde "
                "diversidade).",
    },
    {
        "name": "pos_weight", "train_kw": "pos_weight", "priority": "high",
        "kind": "float", "default": (0.5, 4.0), "cast": float,
        "help": "Peso do termo de posição 3D na recon loss. Domina ~78% da "
                "loss nos defaults atuais (pos_weight=2.0); é o gargalo "
                "declarado do modelo.",
    },
    {
        "name": "stop_weight", "train_kw": "stop_weight", "priority": "high",
        "kind": "float", "default": (0.5, 4.0), "cast": float,
        "help": "Peso da cabeça de parada (quando encerrar a construção do "
                "ligante). Controla diretamente a distribuição de tamanhos "
                "das moléculas geradas.",
    },
    # ── Média prioridade ──────────────────────────────────────────────────
    {
        "name": "beta_warmup", "train_kw": "beta_warmup", "priority": "medium",
        "kind": "int", "default": (10, 50), "cast": int,
        "help": "Nº de epochs até beta atingir beta_max (KL annealing). "
                "Warmup curto demais favorece posterior collapse.",
    },
    {
        "name": "dropout", "train_kw": "dropout", "priority": "medium",
        "kind": "float", "default": (0.05, 0.35), "cast": float,
        "help": "Regularização. Com o dataset atual (~26k pares de treino) e "
                "val abaixo de train, há indício de subajuste — valores "
                "baixos tendem a ganhar.",
    },
    {
        "name": "hidden_dim", "train_kw": "hidden_dim", "priority": "medium",
        "kind": "categorical", "default": (128, 256, 384), "cast": int,
        "help": "Capacidade do decoder GNN. Sem overfitting observado, há "
                "folga para subir.",
    },
    {
        "name": "n_gnn_layers", "train_kw": "n_gnn_layers", "priority": "medium",
        "kind": "categorical", "default": (3, 4, 5, 6), "cast": int,
        "help": "Profundidade do decoder GNN. Mais camadas ampliam o campo "
                "receptivo sobre o grafo parcial, ao custo de over-smoothing.",
    },
    # ── Baixa prioridade ──────────────────────────────────────────────────
    {
        "name": "latent_dim", "train_kw": "latent_dim", "priority": "low",
        "kind": "categorical", "default": (128, 256), "cast": int,
        "help": "Dimensão do espaço latente. A KL é média (não soma) por "
                "elemento, então mudar latent_dim NÃO exige reajustar beta.",
    },
    {
        "name": "batch_size", "train_kw": "batch_size", "priority": "low",
        "kind": "categorical", "default": (16, 32, 64), "cast": int,
        "help": "Interage fortemente com lr — tunar isolado engana. Só faz "
                "sentido com lr também marcado.",
    },
    {
        "name": "weight_decay", "train_kw": "weight_decay", "priority": "low",
        "kind": "float_log", "default": (1e-5, 1e-3), "cast": float,
        "help": "Regularização L2 do AdamW.",
    },
    {
        "name": "grad_clip", "train_kw": "grad_clip", "priority": "low",
        "kind": "float", "default": (0.2, 2.0), "cast": float,
        "help": "Norma máxima do gradiente. Em geral não compensa buscar — "
                "mantenha fixo salvo se houver instabilidade (NaN) no treino.",
    },
]

SPEC_BY_NAME: dict[str, dict] = {s["name"]: s for s in SEARCH_SPACE_SPEC}

#: Hiperparâmetros marcados por padrão na interface (alta + média prioridade).
DEFAULT_SELECTED: list[str] = [
    s["name"] for s in SEARCH_SPACE_SPEC if s["priority"] in ("high", "medium")
]

#: Se qualquer um destes estiver no espaço de busca, o objetivo passa a ser
#: `val_recon` — ver resolve_objective_metric().
BETA_PARAMS = frozenset({"beta_max", "beta_warmup"})


# ══════════════════════════════════════════════════════════════════════════════
# FIDELIDADE DA BUSCA
# ══════════════════════════════════════════════════════════════════════════════
#
# A busca é uma TRIAGEM (proxy) ou uma COMPARAÇÃO DIRETA conforme o orçamento
# que o usuário escolher — não por natureza. São três lacunas entre o que a
# busca mede e o que se quer saber, e duas delas fecham com orçamento:
#
#   1. Dados     (subset_frac < 1)  — o ranking obtido num subconjunto pode não
#      transferir para o dataset inteiro. Atinge sobretudo a capacidade
#      (hidden_dim, n_gnn_layers): modelos grandes são penalizados em dataset
#      pequeno e perderiam de configurações que, com todos os dados, venceriam.
#   2. Orçamento (trial_epochs << epochs do treino real) — ranquear cedo
#      favorece quem aprende rápido (lr alto, beta baixo), não quem termina
#      melhor.
#   3. Objetivo  — val_recon/val_total não é qualidade de geração. Esta NÃO
#      fecha com orçamento nenhum: continua sendo uma loss, e a correlação com
#      validade química, unicidade e docking é positiva mas fraca.
#
# Só faz sentido mandar "confirme na Etapa 7 com um treino completo" enquanto
# 1 ou 2 estiverem abertas. Com frac=1.0 e trial_epochs = epochs de referência,
# cada trial JÁ É um treino completo e essa recomendação seria redundante.

#: Fração a partir da qual o subconjunto deixa de ser um proxy relevante.
FULL_DATA_FRAC = 0.90

#: Épocas do treino real (Etapa 7), lidas do próprio default de train() para
#: não virarem um número solto que sai de sincronia quando o default mudar.
def _reference_epochs() -> int:
    try:
        import inspect
        from MODULES.pipeline_train import train
        return int(inspect.signature(train).parameters["epochs"].default)
    except Exception:
        return 150


def fidelity_report(subset_frac: float, trial_epochs: int,
                    reference_epochs: Optional[int] = None) -> dict:
    """Classifica a busca em triagem / quase-completa / completa.

    Returns dict com `level` ("screen" | "near_full" | "full"), `gaps` (lista
    das lacunas abertas) e `advice` (texto para o log).
    """
    ref = reference_epochs or _reference_epochs()
    data_gap   = subset_frac < FULL_DATA_FRAC
    budget_gap = trial_epochs < ref

    gaps = []
    if data_gap:
        gaps.append(
            f"dados ({subset_frac:.0%} do dataset): o ranking pode não "
            f"transferir para 100%, sobretudo para hidden_dim/n_gnn_layers, "
            f"que são penalizados em dataset pequeno"
        )
    if budget_gap:
        gaps.append(
            f"orçamento ({trial_epochs} de {ref} epochs): ranquear cedo "
            f"favorece quem aprende rápido (lr alto, beta baixo), não quem "
            f"termina melhor"
        )

    if not gaps:
        level = "full"
        advice = (
            "Cada trial foi um treino COMPLETO (dataset inteiro, epochs de "
            "referência) — isto não é uma triagem, é uma comparação direta. "
            "Não é preciso repetir o treino na Etapa 7: o melhor trial já é o "
            "modelo treinado.\n"
            "  Resta apenas a lacuna de OBJETIVO, que nenhum orçamento fecha: "
            "val_recon é uma loss, não qualidade de geração. Compare as "
            "melhores configurações pelas métricas das Etapas 8-9 (validade, "
            "unicidade, docking, checks tipo PoseBusters) antes de decidir."
        )
    elif len(gaps) == 1 and not data_gap:
        level = "near_full"
        advice = (
            "Dataset completo, mas orçamento de epochs reduzido — o ranking é "
            "confiável quanto aos dados e aproximado quanto à convergência.\n"
            "  Lacuna aberta: " + gaps[0] + ".\n"
            "  Confirme as 2-3 melhores com um treino de duração completa na "
            "Etapa 7, e depois compare pelas métricas de geração (Etapas 8-9)."
        )
    else:
        level = "screen"
        advice = (
            "Isto foi uma TRIAGEM (proxy) — use-a para selecionar 2-3 "
            "configurações, não para decidir.\n"
            + "".join(f"  Lacuna aberta: {g}.\n" for g in gaps)
            + "  Confirme as finalistas com um treino completo na Etapa 7 e "
              "compare pelas métricas de geração (Etapas 8-9), não pela loss."
        )

    return {"level": level, "gaps": gaps, "advice": advice,
            "reference_epochs": ref}


def format_default(name: str) -> str:
    """Formata o valor default de um hiperparâmetro como string editável.

    É o texto que a interface coloca na caixa ao lado do checkbox e o que
    `parse_range()` sabe reler.
    """
    spec = SPEC_BY_NAME[name]
    vals = spec["default"]
    if spec["kind"] == "float_log":
        return ",".join(f"{v:g}" for v in vals)
    if spec["kind"] == "float":
        return ",".join(f"{v:g}" for v in vals)
    return ",".join(str(v) for v in vals)


def parse_range(name: str, text: str) -> tuple:
    """Converte o texto da interface/CLI de volta para a tupla do espaço.

    Aceita vírgula, ponto-e-vírgula ou espaço como separador. Valida a
    aridade: faixas (float/int/float_log) exigem exatamente 2 valores;
    categóricos exigem >= 2 opções.
    """
    spec = SPEC_BY_NAME[name]
    raw = [t for t in text.replace(";", ",").replace(" ", ",").split(",") if t]
    if not raw:
        raise ValueError(f"'{name}': nenhum valor informado.")
    try:
        vals = tuple(spec["cast"](v) for v in raw)
    except ValueError as e:
        raise ValueError(f"'{name}': valor inválido em '{text}' ({e}).") from e

    if spec["kind"] == "categorical":
        if len(vals) < 2:
            raise ValueError(
                f"'{name}': categórico exige ao menos 2 opções "
                f"(recebido: {text!r}). Se quer o valor fixo, desmarque o "
                f"checkbox e ajuste-o na Etapa 7."
            )
        return vals

    if len(vals) != 2:
        raise ValueError(
            f"'{name}': faixa exige exatamente 2 valores 'mín,máx' "
            f"(recebido: {text!r})."
        )
    if vals[0] >= vals[1]:
        raise ValueError(f"'{name}': mínimo ({vals[0]}) >= máximo ({vals[1]}).")
    return vals


def build_search_space(selected: dict[str, str]) -> dict[str, tuple]:
    """Monta o espaço de busca a partir de {nome: texto_da_caixa}.

    Levanta ValueError com mensagem legível se algo estiver malformado — a
    interface mostra isso direto num QMessageBox.
    """
    space: dict[str, tuple] = {}
    for name, text in selected.items():
        if name not in SPEC_BY_NAME:
            raise ValueError(f"Hiperparâmetro desconhecido: '{name}'.")
        space[name] = parse_range(name, text)
    if not space:
        raise ValueError(
            "Nenhum hiperparâmetro selecionado — marque ao menos um "
            "checkbox para definir o espaço de busca."
        )
    return space


def resolve_objective_metric(search_space: dict) -> str:
    """Decide entre 'recon' e 'total' como métrica-objetivo.

    `total = recon + beta·KL`. Se beta (ou o seu warmup) varia entre trials,
    comparar `total` premia quem sorteou beta menor em vez de quem
    reconstruiu melhor — o ranking mediria o espaço de busca, não o modelo.
    Nesse caso o objetivo passa a ser `recon`, que é invariante a beta.
    """
    return "recon" if (BETA_PARAMS & set(search_space)) else "total"


def suggest_params(trial, search_space: dict[str, tuple]) -> dict[str, Any]:
    """Traduz o espaço de busca em chamadas trial.suggest_* do Optuna."""
    out: dict[str, Any] = {}
    for name, vals in search_space.items():
        spec = SPEC_BY_NAME[name]
        kind = spec["kind"]
        if kind == "float_log":
            out[name] = trial.suggest_float(name, vals[0], vals[1], log=True)
        elif kind == "float":
            out[name] = trial.suggest_float(name, vals[0], vals[1])
        elif kind == "int":
            out[name] = trial.suggest_int(name, int(vals[0]), int(vals[1]))
        else:
            out[name] = trial.suggest_categorical(name, list(vals))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SUBAMOSTRAGEM DO DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_key(df: pd.DataFrame) -> pd.Series:
    """Mesma regra de cluster do SiteLigandDataset (pdb_id, senão pair_id[:4])."""
    if "pdb_id" in df.columns:
        return df["pdb_id"].astype(str).str.upper()
    return df["pair_id"].astype(str).str[:4]


def make_subset(
    tensor_dir: str | Path,
    frac: float,
    out_dir: str | Path,
    seed: int = 42,
) -> Path:
    """Cria um subconjunto do diretório de tensores para trials baratos.

    A amostragem é feita por **cluster PDB**, não por par: o
    SiteLigandDataset divide treino/val/teste por cluster para evitar
    vazamento, e amostrar pares soltos poderia partir um cluster entre os
    splits do subconjunto.

    Não copia dados — cria *symlinks* para os diretórios de cada par. Um
    subconjunto de 20% de 11 GB custa alguns MB de metadados, não 2,2 GB.

    Reutiliza o diretório se já existir um subconjunto com o mesmo
    (frac, seed, origem), registrado em `subset_info.json`.
    """
    tensor_dir = Path(tensor_dir).resolve()
    out_dir    = Path(out_dir)

    if not (0.0 < frac <= 1.0):
        raise ValueError(f"subset_frac deve estar em (0, 1]; recebido: {frac}")

    if frac >= 1.0:
        log.info("subset_frac=1.0 — usando o dataset completo, sem subconjunto.")
        return tensor_dir

    marker = out_dir / "subset_info.json"
    if marker.exists():
        try:
            info = json.loads(marker.read_text())
            if (info.get("frac") == frac and info.get("seed") == seed
                    and info.get("source") == str(tensor_dir)):
                log.info(f"Subconjunto reaproveitado: {out_dir} "
                         f"({info.get('n_pairs')} pares, "
                         f"{info.get('n_clusters')} clusters)")
                return out_dir
            log.info("Subconjunto existente tem outra configuração — refazendo.")
        except Exception:
            log.warning("subset_info.json ilegível — refazendo o subconjunto.")
        shutil.rmtree(out_dir, ignore_errors=True)

    index_path = tensor_dir / "tensor_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"tensor_index.csv não encontrado em {tensor_dir}. "
            f"Rode a Etapa 5 (voxelização) antes da otimização."
        )

    df = pd.read_csv(
        index_path, sep=None, engine="python",
        true_values=["True", "TRUE", "true"],
        false_values=["False", "FALSE", "false"],
    )
    df["_cluster"] = _cluster_key(df)

    clusters = sorted(df["_cluster"].unique())
    rng = random.Random(seed)
    rng.shuffle(clusters)
    n_keep = max(1, int(math.ceil(len(clusters) * frac)))
    keep   = set(clusters[:n_keep])

    sub = df[df["_cluster"].isin(keep)].drop(columns=["_cluster"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "tensor_index.csv", index=False)

    # voxel_config.json é lido por train() para embutir a config no checkpoint
    vc = tensor_dir / "voxel_config.json"
    if vc.exists():
        shutil.copy2(vc, out_dir / "voxel_config.json")

    n_linked = 0
    for pair_id in sub["pair_id"].astype(str):
        src = tensor_dir / pair_id
        dst = out_dir / pair_id
        if dst.exists() or dst.is_symlink():
            continue
        if src.is_dir():
            os.symlink(src, dst, target_is_directory=True)
            n_linked += 1

    marker.write_text(json.dumps({
        "source":     str(tensor_dir),
        "frac":       frac,
        "seed":       seed,
        "n_pairs":    int(len(sub)),
        "n_clusters": len(keep),
        "n_linked":   n_linked,
    }, indent=2))

    log.info(f"Subconjunto criado em {out_dir}: {len(sub)} pares "
             f"({len(keep)}/{len(clusters)} clusters, {n_linked} symlinks)")
    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# ESTIMATIVA DE ESPAÇO EM DISCO
# ══════════════════════════════════════════════════════════════════════════════

#: Margem de segurança exigida além da estimativa (o disco não deve encostar
#: em 100% por causa da HPO).
DISK_HEADROOM_BYTES = 2 * 1024 ** 3


def _count_params(latent_dim: int, hidden_dim: int, n_gnn_layers: int) -> int:
    """Conta parâmetros instanciando o modelo na CPU.

    Estimar por fórmula exigiria replicar a arquitetura aqui e sairia do ar
    a cada mudança em SiteEncoder/GNNDecoder. Instanciar custa ~0,2 s e é
    exato.
    """
    import torch  # import tardio: pesado, e nem todo caminho de código precisa
    from MODULES.pipeline_train import LigandCVAE

    with torch.no_grad():
        m = LigandCVAE(
            in_channels=5, latent_dim=latent_dim, hidden_dim=hidden_dim,
            n_gnn_layers=n_gnn_layers, dropout=0.0,
        )
        n = sum(p.numel() for p in m.parameters())
        del m
    return int(n)


def estimate_disk_usage(
    search_space: dict[str, tuple],
    n_trials: int,
    trial_epochs: int,
    keep_all_checkpoints: bool = False,
    base_latent: int = 256,
    base_hidden: int = 256,
    base_layers: int = 4,
) -> dict:
    """Estima o espaço em disco necessário pela busca.

    Contabilidade (medida nos checkpoints reais do CODEEP):
      * `best_model.pt` = model_state + optim_state (AdamW guarda 2 momentos
        por parâmetro) + histórico ≈ **12 bytes/parâmetro**.
      * checkpoint periódico (a cada 10 epochs) = só model_state
        ≈ **4 bytes/parâmetro**.

    Usa o MAIOR modelo do espaço de busca — subestimar aqui é o erro que
    trava a busca no meio.
    """
    latent = max(search_space.get("latent_dim",   (base_latent,)))
    hidden = max(search_space.get("hidden_dim",   (base_hidden,)))
    layers = max(search_space.get("n_gnn_layers", (base_layers,)))

    try:
        n_params = _count_params(int(latent), int(hidden), int(layers))
    except Exception as e:  # pragma: no cover — só se o torch faltar
        log.warning(f"Não foi possível contar parâmetros ({e}); "
                    f"usando estimativa grosseira de 3M.")
        n_params = 3_000_000

    best_ckpt   = n_params * 12
    periodic    = n_params * 4
    n_periodic  = trial_epochs // 10
    per_trial   = best_ckpt + n_periodic * periodic

    # Sem retenção, o pipeline mantém no disco apenas o trial corrente e o
    # melhor até agora — daí o fator 2.
    n_retained = n_trials if keep_all_checkpoints else 2
    ckpt_bytes = per_trial * n_retained

    db_bytes = 5 * 1024 ** 2          # SQLite do Optuna: alguns MB, generoso
    misc     = 50 * 1024 ** 2         # logs, CSVs, symlinks do subconjunto

    total = ckpt_bytes + db_bytes + misc
    return {
        "n_params":       n_params,
        "per_trial":      per_trial,
        "n_retained":     n_retained,
        "checkpoints":    ckpt_bytes,
        "database":       db_bytes,
        "misc":           misc,
        "total":          total,
        "required":       total + DISK_HEADROOM_BYTES,
    }


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_disk_space(
    out_dir: str | Path,
    search_space: dict[str, tuple],
    n_trials: int,
    trial_epochs: int,
    keep_all_checkpoints: bool = False,
) -> dict:
    """Verifica se há espaço para a busca ANTES de começar.

    Retorna dict com `ok`, `free`, `required` e `message` (texto pronto para
    exibição). Quem chama decide o que fazer: a interface mostra um diálogo
    com opção de cancelar; a CLI exige --yes para prosseguir.
    """
    out_dir = Path(out_dir)
    probe = out_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    est  = estimate_disk_usage(search_space, n_trials, trial_epochs,
                               keep_all_checkpoints)
    free = shutil.disk_usage(probe).free
    ok   = free >= est["required"]

    retention = ("todos os trials" if keep_all_checkpoints
                 else "apenas o melhor trial + o corrente")
    msg = (
        f"Estimativa de espaço em disco para a otimização\n"
        f"───────────────────────────────────────────────\n"
        f"  Parâmetros do maior modelo : {est['n_params']:,}\n"
        f"  Checkpoint por trial       : {_fmt_bytes(est['per_trial'])}\n"
        f"  Retenção                   : {retention} "
        f"({est['n_retained']} trial(s))\n"
        f"  Checkpoints                : {_fmt_bytes(est['checkpoints'])}\n"
        f"  Banco Optuna (SQLite)      : {_fmt_bytes(est['database'])}\n"
        f"  Logs / CSVs / symlinks     : {_fmt_bytes(est['misc'])}\n"
        f"  ─────────────────────────────────────────────\n"
        f"  Total estimado             : {_fmt_bytes(est['total'])}\n"
        f"  Margem de segurança        : {_fmt_bytes(DISK_HEADROOM_BYTES)}\n"
        f"  Necessário                 : {_fmt_bytes(est['required'])}\n"
        f"  Livre em {str(probe):<18}: {_fmt_bytes(free)}\n"
    )
    if ok:
        msg += "\n✅ Espaço suficiente."
    else:
        msg += (
            f"\n⚠️  ESPAÇO INSUFICIENTE — faltam "
            f"{_fmt_bytes(est['required'] - free)}.\n"
            f"Opções: reduzir o nº de trials, reduzir as epochs por trial, "
            f"desmarcar 'manter checkpoints de todos os trials', ou liberar "
            f"espaço no disco."
        )

    return {"ok": ok, "free": free, "required": est["required"],
            "estimate": est, "message": msg}


# ══════════════════════════════════════════════════════════════════════════════
# BUSCA
# ══════════════════════════════════════════════════════════════════════════════

def optimize(
    tensor_dir: str | Path,
    out_dir: str | Path,
    search_space: dict[str, tuple] | None = None,
    *,
    n_trials: int = 40,
    trial_epochs: int = 15,
    subset_frac: float = 0.20,
    seed: int = 42,
    pruner_min_resource: int = 3,
    pruner_reduction_factor: int = 3,
    study_name: str = "codeep_hpo",
    device: str = "auto",
    keep_all_checkpoints: bool = False,
    resume: Optional[str] = None,
    freeze_encoder: bool = False,
    fixed_params: Optional[dict] = None,
    skip_disk_check: bool = False,
) -> dict:
    """Executa a busca de hiperparâmetros e devolve o melhor resultado.

    Args:
        tensor_dir   : diretório de tensores da fase (saída da Etapa 5).
        out_dir      : onde ficam o SQLite, os checkpoints dos trials e os
                       relatórios.
        search_space : {nome: tupla}. Se None, usa alta+média prioridade
                       com os defaults do módulo.
        n_trials     : nº de configurações a testar (podadas incluídas).
        trial_epochs : epochs por trial — orçamento, não busca.
        subset_frac  : fração de CLUSTERS PDB usada nos trials (1.0 = tudo).
        resume/freeze_encoder : repassados a train(), para otimizar
                       hiperparâmetros de fine-tuning (Fases 2/3).
        fixed_params : valores fixos para hiperparâmetros NÃO buscados
                       (sobrescrevem os defaults de train()).

    Returns:
        dict com best_value, best_params, objective_metric e caminhos.
    """
    import optuna
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.samplers import TPESampler

    from MODULES.pipeline_train import train

    # O logger do Optuna duplicaria cada linha no log da interface.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if search_space is None:
        search_space = {n: SPEC_BY_NAME[n]["default"] for n in DEFAULT_SELECTED}
    if not search_space:
        raise ValueError("search_space vazio — selecione ao menos um "
                         "hiperparâmetro.")

    unknown = set(search_space) - set(SPEC_BY_NAME)
    if unknown:
        raise ValueError(f"Hiperparâmetros desconhecidos: {sorted(unknown)}")

    objective_metric = resolve_objective_metric(search_space)
    fixed_params = dict(fixed_params or {})

    # ── Verificação de disco ──────────────────────────────────────────────
    if not skip_disk_check:
        disk = check_disk_space(out_dir, search_space, n_trials, trial_epochs,
                                keep_all_checkpoints)
        for line in disk["message"].splitlines():
            log.info(line)
        if not disk["ok"]:
            raise RuntimeError(
                "Espaço em disco insuficiente para a otimização — veja a "
                "estimativa acima. Execução cancelada."
            )

    # ── Cabeçalho ─────────────────────────────────────────────────────────
    log.info("═" * 72)
    log.info("Etapa 6 — Otimização de hiperparâmetros")
    log.info("═" * 72)
    log.info(f"  Trials                : {n_trials}")
    log.info(f"  Epochs por trial      : {trial_epochs}")
    log.info(f"  Fração do dataset     : {subset_frac:.0%}")
    log.info(f"  Métrica-objetivo      : val_{objective_metric}"
             + ("  (beta está no espaço de busca — val_total premiaria "
                "beta menor, não modelo melhor)" if objective_metric == "recon"
                else ""))
    log.info(f"  Poda                  : ASHA "
             f"(min_resource={pruner_min_resource}, "
             f"reduction_factor={pruner_reduction_factor})")

    fid = fidelity_report(subset_frac, trial_epochs)
    log.info(f"  Fidelidade            : "
             + {"screen":    "TRIAGEM (proxy)",
                "near_full": "quase completa (dados inteiros, epochs reduzidas)",
                "full":      "COMPARAÇÃO DIRETA (cada trial é um treino completo)"
                }[fid["level"]])
    if fid["level"] == "full":
        # Com orçamento total o ASHA vira o principal risco: os rungs ficam
        # nas primeiras epochs de um treino longo, e uma configuração que só
        # decola tarde é morta antes de mostrar serviço.
        rungs, r = [], pruner_min_resource
        while r < trial_epochs:
            rungs.append(r)
            r *= pruner_reduction_factor
        if rungs:
            log.warning(
                f"  ⚠  Orçamento completo com poda ativa: o ASHA avalia nas "
                f"epochs {rungs} de {trial_epochs}. Uma configuração de "
                f"convergência lenta pode ser podada antes de mostrar seu "
                f"valor. Para uma comparação sem esse viés, use "
                f"min_resource >= {trial_epochs} (desliga a poda na prática)."
            )

    log.info(f"  Espaço de busca ({len(search_space)}):")
    for name, vals in search_space.items():
        kind = SPEC_BY_NAME[name]["kind"]
        rng_txt = (f"{vals}" if kind == "categorical"
                   else f"[{vals[0]:g}, {vals[1]:g}]"
                        + ("  (log)" if kind == "float_log" else ""))
        log.info(f"      {name:<16} {kind:<12} {rng_txt}")
    if fixed_params:
        log.info(f"  Fixos                 : {fixed_params}")

    # ── Subconjunto ───────────────────────────────────────────────────────
    work_tensor_dir = make_subset(
        tensor_dir, subset_frac, out_dir / "tensors_subset", seed=seed
    )

    # ── Study ─────────────────────────────────────────────────────────────
    db_path = out_dir / "optuna_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=SuccessiveHalvingPruner(
            min_resource=pruner_min_resource,
            reduction_factor=pruner_reduction_factor,
        ),
        storage=storage,
        load_if_exists=True,
    )
    log.info(f"  Storage               : {db_path}")
    n_done = len([t for t in study.trials
                  if t.state.name in ("COMPLETE", "PRUNED")])
    if n_done:
        log.info(f"  Study existente: {n_done} trial(s) já registrados — "
                 f"a busca continua de onde parou.")
    log.info("═" * 72)

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    # Tempo é a restrição que de fato decide se uma configuração de busca é
    # viável — muito antes do disco. Não dá para estimar a priori (depende de
    # GPU, hidden_dim e tamanho do subconjunto), então medimos a primeira
    # epoch real e projetamos a partir dela.
    state = {"best_value": float("inf"), "best_trial": None,
             "epoch_times": [], "t_start": None}

    def _fmt_dur(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        d, rem = divmod(seconds, 86400)
        h, rem = divmod(rem, 3600)
        m, _ = divmod(rem, 60)
        if d:
            return f"{d}d {h}h {m}min"
        if h:
            return f"{h}h {m}min"
        return f"{m}min"

    def _log_time_projection():
        """Projeta a duração total da busca a partir das epochs já medidas."""
        times = state["epoch_times"]
        if not times:
            return
        per_epoch = sum(times) / len(times)
        # Limite superior: nenhum trial podado. Limite inferior otimista: o
        # ASHA costuma cortar ~2/3 dos trials nos primeiros rungs.
        worst = n_trials * trial_epochs * per_epoch
        typical = worst * 0.45 if pruner_min_resource < trial_epochs else worst
        log.info(
            f"  ⏱  {per_epoch/60:.1f} min/epoch medido → busca completa entre "
            f"~{_fmt_dur(typical)} (com poda típica) e ~{_fmt_dur(worst)} "
            f"(sem poda alguma), para {n_trials} trials × {trial_epochs} epochs."
        )

    def _cleanup(trial_number: int):
        """Apaga os checkpoints de um trial que não é o melhor."""
        if keep_all_checkpoints:
            return
        d = trials_dir / f"trial_{trial_number:04d}"
        shutil.rmtree(d, ignore_errors=True)

    def objective(trial) -> float:
        import torch

        params = suggest_params(trial, search_space)
        kwargs = dict(fixed_params)
        for name, value in params.items():
            kwargs[SPEC_BY_NAME[name]["train_kw"]] = value

        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        log.info("")
        log.info("─" * 72)
        log.info(f"Trial {trial.number + 1}/{n_trials}  →  "
                 + "  ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in params.items()))
        log.info("─" * 72)

        # Melhor valor da métrica-objetivo DENTRO deste trial. Rastreado
        # aqui (e não lido de best_model.pt) porque train() seleciona o
        # melhor checkpoint por val_total, enquanto o objetivo pode ser
        # val_recon — os dois divergem quando beta varia.
        best_in_trial = {"value": float("inf")}
        state["t_start"] = _time.monotonic()

        def epoch_callback(epoch, train_losses, val_losses):
            now = _time.monotonic()
            if state["t_start"] is not None:
                first = not state["epoch_times"]
                state["epoch_times"].append(now - state["t_start"])
                state["t_start"] = now
                if first:
                    # Projeção logo após a primeira epoch medida: se o
                    # orçamento escolhido levar semanas, o usuário descobre
                    # em minutos e não depois de meio dia de GPU.
                    _log_time_projection()

            value = float(val_losses[objective_metric])
            best_in_trial["value"] = min(best_in_trial["value"], value)
            trial.report(value, epoch)
            if trial.should_prune():
                log.info(f"  ✂  Trial {trial.number + 1} podado na epoch "
                         f"{epoch} (val_{objective_metric}={value:.4f}).")
                raise optuna.TrialPruned()

        try:
            train(
                epochs=trial_epochs,
                checkpoint_dir=str(trial_dir),
                tensor_dir=str(work_tensor_dir),
                device=device,
                resume=resume,
                freeze_encoder=freeze_encoder,
                # Sem early stopping interno: quem decide interromper é o
                # ASHA, com informação de todos os trials. patience baixo
                # aqui mataria trials bons antes do pruner opinar.
                patience=10 ** 6,
                epoch_callback=epoch_callback,
                **kwargs,
            )
        finally:
            # Libera a VRAM entre trials — sem isso, o modelo do trial
            # anterior ainda ocupa memória quando o próximo instancia o
            # dele, e configurações grandes estouram a GPU no meio da busca.
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        value = best_in_trial["value"]
        if not math.isfinite(value):
            raise RuntimeError(
                f"Trial {trial.number} não produziu métrica finita — "
                f"provável divergência (NaN) no treino."
            )

        if value < state["best_value"]:
            previous = state["best_trial"]
            state["best_value"] = value
            state["best_trial"] = trial.number
            log.info(f"  ★ Melhor trial até agora: #{trial.number + 1} "
                     f"(val_{objective_metric}={value:.4f})")
            if previous is not None:
                _cleanup(previous)
        else:
            _cleanup(trial.number)

        return value

    def _on_trial_end(study_, trial_):
        """Poda também limpa checkpoints — o finally do objective não roda
        para trials podados antes de qualquer escrita, mas roda depois."""
        if trial_.state.name == "PRUNED" and trial_.number != state["best_trial"]:
            _cleanup(trial_.number)

    study.optimize(objective, n_trials=n_trials, callbacks=[_on_trial_end],
                   gc_after_trial=True)

    # ── Relatório ─────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    failed    = [t for t in study.trials if t.state.name == "FAIL"]

    log.info("")
    log.info("═" * 72)
    log.info("Otimização concluída")
    log.info("═" * 72)
    log.info(f"  Trials completos : {len(completed)}")
    log.info(f"  Trials podados   : {len(pruned)}")
    log.info(f"  Trials falhados  : {len(failed)}")

    if not completed:
        log.warning("Nenhum trial completou — nada a reportar. Verifique os "
                    "erros acima (VRAM, dataset ou espaço de busca inválido).")
        return {"best_value": None, "best_params": {},
                "objective_metric": objective_metric,
                "study_db": str(db_path), "out_dir": str(out_dir)}

    best = study.best_trial
    log.info(f"  Melhor trial     : #{best.number + 1}")
    log.info(f"  val_{objective_metric:<12}: {best.value:.4f}")
    log.info("  Hiperparâmetros:")
    for k, v in best.params.items():
        log.info(f"      {k:<16} = {v}")

    # Tabela completa dos trials — permite inspecionar o ranking depois sem
    # reabrir o SQLite.
    rows = []
    for t in study.trials:
        row = {"trial": t.number + 1, "state": t.state.name,
               f"val_{objective_metric}": t.value}
        row.update(t.params)
        rows.append(row)
    csv_path = out_dir / "hpo_trials.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = {
        "best_value":       float(best.value),
        "best_params":      dict(best.params),
        "best_trial":       best.number + 1,
        "objective_metric": objective_metric,
        "n_completed":      len(completed),
        "n_pruned":         len(pruned),
        "n_failed":         len(failed),
        "search_space":     {k: list(v) for k, v in search_space.items()},
        "trial_epochs":     trial_epochs,
        "subset_frac":      subset_frac,
        "study_db":         str(db_path),
        "trials_csv":       str(csv_path),
        "out_dir":          str(out_dir),
    }
    best_json = out_dir / "best_params.json"
    best_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    log.info("")
    log.info(f"  Tabela de trials : {csv_path}")
    log.info(f"  Melhores params  : {best_json}")
    log.info("")
    # O texto abaixo depende do orçamento que o usuário escolheu: com
    # frac=1.0 e epochs de referência isto deixa de ser triagem e mandar
    # "confirme com um treino completo" seria mandar repetir o que já foi
    # feito. Ver fidelity_report().
    log.info(f"  PRÓXIMO PASSO  ({subset_frac:.0%} do dataset, "
             f"{trial_epochs} de {fid['reference_epochs']} epochs)")
    for line in fid["advice"].splitlines():
        log.info(f"  {line}" if not line.startswith("  ") else line)
    if state["epoch_times"]:
        total = sum(state["epoch_times"])
        log.info(f"  Tempo de GPU consumido: {_fmt_dur(total)} "
                 f"({len(state['epoch_times'])} epochs efetivas)")
    log.info("═" * 72)
    result["fidelity"] = fid["level"]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_params_table():
    print("\nHiperparâmetros disponíveis para busca\n" + "─" * 72)
    for prio in ("high", "medium", "low"):
        rotulo = {"high": "ALTA", "medium": "MÉDIA", "low": "BAIXA"}[prio]
        marca  = "marcado por padrão" if prio != "low" else "desmarcado por padrão"
        print(f"\n  Prioridade {rotulo}  ({marca})")
        for spec in SEARCH_SPACE_SPEC:
            if spec["priority"] != prio:
                continue
            print(f"    {spec['name']:<16} {spec['kind']:<12} "
                  f"default: {format_default(spec['name'])}")
            print(f"    {'':<16} {spec['help']}")
    print("\n" + "─" * 72)
    print("Use --params nome1,nome2  e  --range nome=v1,v2  para customizar.\n")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="CODEEP — Etapa 6: otimização de hiperparâmetros (Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Busca padrão (alta+média prioridade), 40 trials de 15 epochs em 20% dos dados
  python MODULES/pipeline_optimize.py --tensor-dir data/tensors

  # Só os de alta prioridade, com faixa de lr customizada
  python MODULES/pipeline_optimize.py --tensor-dir data/tensors \\
      --params lr,beta_max,pos_weight,stop_weight --range lr=2e-4,8e-4

  # Busca rápida de sanidade (5 trials curtos em 5% dos dados)
  python MODULES/pipeline_optimize.py --tensor-dir data/tensors \\
      --n-trials 5 --trial-epochs 3 --subset-frac 0.05

  # Só a estimativa de disco, sem rodar nada
  python MODULES/pipeline_optimize.py --tensor-dir data/tensors --check-disk
        """,
    )
    parser.add_argument("--tensor-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "tensors"),
                        help="Diretório de tensores da Etapa 5")
    parser.add_argument("--out-dir", type=str,
                        default=str(PROJECT_ROOT / "hpo"),
                        help="Saída: SQLite, checkpoints dos trials, relatórios")
    parser.add_argument("--params", type=str, default=None,
                        help="Lista separada por vírgula dos hiperparâmetros a "
                             "buscar (default: alta+média prioridade)")
    parser.add_argument("--range", action="append", default=[], metavar="NOME=v1,v2",
                        help="Sobrescreve a faixa/opções de um hiperparâmetro "
                             "(pode repetir)")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--trial-epochs", type=int, default=15)
    parser.add_argument("--subset-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-resource", type=int, default=3,
                        help="ASHA: epochs mínimos antes de podar (default: 3)")
    parser.add_argument("--reduction-factor", type=int, default=3,
                        help="ASHA: fator de redução (default: 3)")
    parser.add_argument("--study-name", type=str, default="codeep_hpo")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="Mantém os checkpoints de TODOS os trials "
                             "(consome muito disco)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint base para otimizar fine-tuning "
                             "(Fases 2/3)")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--list-params", action="store_true",
                        help="Lista os hiperparâmetros disponíveis e sai")
    parser.add_argument("--check-disk", action="store_true",
                        help="Só mostra a estimativa de espaço e sai")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Prossegue mesmo sem espaço em disco suficiente")

    args = parser.parse_args(argv)

    if args.list_params:
        _print_params_table()
        return 0

    names = ([n.strip() for n in args.params.split(",") if n.strip()]
             if args.params else list(DEFAULT_SELECTED))
    selected = {n: format_default(n) for n in names if n in SPEC_BY_NAME}
    invalid  = [n for n in names if n not in SPEC_BY_NAME]
    if invalid:
        print(f"Hiperparâmetros desconhecidos: {invalid}\n"
              f"Use --list-params para ver os disponíveis.", file=sys.stderr)
        return 2

    for override in args.range:
        if "=" not in override:
            print(f"--range malformado: {override!r} (esperado NOME=v1,v2)",
                  file=sys.stderr)
            return 2
        name, text = override.split("=", 1)
        name = name.strip()
        if name not in SPEC_BY_NAME:
            print(f"--range para hiperparâmetro desconhecido: {name!r}",
                  file=sys.stderr)
            return 2
        selected[name] = text

    try:
        space = build_search_space(selected)
    except ValueError as e:
        print(f"Erro no espaço de busca: {e}", file=sys.stderr)
        return 2

    # Verificação de disco: na CLI o usuário confirma com --yes; sem isso a
    # execução é cancelada, que é o comportamento pedido (avisar e permitir
    # cancelar antes de gastar horas de GPU).
    disk = check_disk_space(args.out_dir, space, args.n_trials,
                            args.trial_epochs, args.keep_all_checkpoints)
    print(disk["message"])
    if args.check_disk:
        return 0 if disk["ok"] else 1
    if not disk["ok"] and not args.yes:
        print("\nExecução cancelada. Use --yes para prosseguir mesmo assim.",
              file=sys.stderr)
        return 1

    optimize(
        tensor_dir=args.tensor_dir,
        out_dir=args.out_dir,
        search_space=space,
        n_trials=args.n_trials,
        trial_epochs=args.trial_epochs,
        subset_frac=args.subset_frac,
        seed=args.seed,
        pruner_min_resource=args.min_resource,
        pruner_reduction_factor=args.reduction_factor,
        study_name=args.study_name,
        device=args.device,
        keep_all_checkpoints=args.keep_all_checkpoints,
        resume=args.resume,
        freeze_encoder=args.freeze_encoder,
        skip_disk_check=True,   # já verificado acima
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
