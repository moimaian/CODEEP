"""
Pipeline - Etapa 5: Dataset PyTorch + Arquitetura GNN Encoder-Decoder
=====================================================================
Define:
  1. SiteLigandDataset  — Dataset PyTorch que carrega pares (tensor, grafo)
  2. SiteEncoder        — CNN 3D que comprime o tensor do sítio em vetor latente
  3. LigandDecoder      — GNN auto-regressivo que gera grafos moleculares
  4. CVAE               — Encoder-Decoder condicional completo
  5. Funções de treino/validação

Requisitos:
    pip install torch torch-geometric numpy pandas tqdm

Uso:
    python 05_train.py --epochs 50 --batch 16 --latent 256

Arquitetura:

    [Tensor do sítio (32,32,32,5)]
              ↓
    SiteEncoder (3D-CNN + MLP)
              ↓
    z_mean, z_logvar  (μ, σ do VAE — dimensão latente = 256)
              ↓  reparametrização
    z  (vetor latente amostrado)
              ↓
    LigandDecoder (GNN auto-regressivo)
              ↓
    Grafo molecular predito
    (átomos, ligações, coordenadas 3D)
"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Verificação de PyG ─────────────────────────────────────────────────────────
try:
    from torch_geometric.data import Data as PyGData, Batch
    from torch_geometric.nn import (
        GCNConv, GATConv, global_mean_pool, global_max_pool
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    raise ImportError(
        "PyTorch Geometric necessário:\n"
        "  pip install torch-geometric\n"
        "  pip install torch-scatter torch-sparse -f "
        "https://data.pyg.org/whl/torch-{versão}+cpu.html"
    )

# ── Caminhos ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(globals().get("_CODEEP_ROOT", Path(__file__).resolve().parent.parent))
BASE_DIR = PROJECT_ROOT / "data"
TENSOR_DIR = BASE_DIR / "tensors"

# Referencial das posições do ligante usado no treino. Gravado no checkpoint
# e lido pela geração, que precisa desfazer a transformação. Ver a nota em
# SiteLigandDataset.__getitem__ sobre por que coordenadas absolutas não são
# aprendíveis a partir de uma grade local ao bolso.
#   "grid_origin_angstrom" → Å relativos à origem da grade do sítio
#   ausente/None           → legado: coordenadas cristalográficas absolutas
POS_FRAME = "grid_origin_angstrom"


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SiteLigandDataset(Dataset):
    """Dataset que carrega pares (tensor do sítio 4D, grafo do ligante).

    Cada item retornado é um dicionário:
      {
        "tensor": Tensor (C, X, Y, Z) float32,  # C=5 canais
        "graph":  PyGData,                       # grafo do ligante
        "pair_id": str,
        "smiles":  str,
      }

    A reorganização para (C, X, Y, Z) segue a convenção PyTorch (channels first).
    """

    @staticmethod
    def _read_index(index_path: Path) -> pd.DataFrame:
        """Lê tensor_index.csv em formato atual (CSV) ou legado (semicolon-separated)."""
        df = pd.read_csv(
            index_path,
            sep=None,
            engine="python",
            true_values=["True", "TRUE", "true"],
            false_values=["False", "FALSE", "false"],
        )

        required_columns = {"pair_id", "graph_saved", "n_ligand_atoms", "status"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Arquivo tensor_index.csv com colunas ausentes: {missing_columns}. "
                f"Arquivo: {index_path}"
            )
        return df

    def __init__(
        self,
        tensor_dir: Path = TENSOR_DIR,
        split: str = "train",
        split_ratio: tuple = (0.80, 0.10, 0.10),
        seed: int = 42,
        max_atoms: int = 60,
    ):
        """
        Args:
            tensor_dir  : diretório com as subpastas {pair_id}/
            split       : "train", "val" ou "test"
            split_ratio : proporções (treino, val, teste) — devem somar 1
            seed        : semente para reprodutibilidade da divisão
            max_atoms   : descarta ligantes com mais de N átomos pesados
        """
        self.tensor_dir = Path(tensor_dir)
        self.max_atoms  = max_atoms
        self.split      = split

        index_path = self.tensor_dir / "tensor_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Execute 04_voxelize.py antes de usar o dataset.\n"
                f"Arquivo esperado: {index_path}"
            )

        df = self._read_index(index_path)
        # Filtra apenas pares processados com sucesso e com grafo
        df = df[
            (df["status"].isin(["ok", "skipped_exists"])) &
            (df["graph_saved"] == True)
        ].copy()        
        df = df[df["n_ligand_atoms"] <= max_atoms].copy()
        df = df.reset_index(drop=True)

        if len(df) == 0:
            df_raw = self._read_index(index_path)
            n_total   = len(df_raw)
            n_ok      = len(df_raw[df_raw["status"] == "ok"])
            n_graph   = len(df_raw[(df_raw["status"] == "ok") & (df_raw["graph_saved"] == True)])
            n_atoms   = len(df_raw[(df_raw["status"] == "ok") & (df_raw["graph_saved"] == True)
                                & (df_raw["n_ligand_atoms"] <= max_atoms)])
            raise ValueError(
                f"Nenhum par válido encontrado em: {index_path}\n"
                f"  Total no índice        : {n_total}\n"
                f"  Com status='ok'        : {n_ok}\n"
                f"  Com graph_saved=True   : {n_graph}\n"
                f"  Com n_ligand_atoms<={max_atoms:3d}: {n_atoms}\n"
                f"\nSe n_ligand_atoms<={max_atoms} = 0, aumente max_qm_atoms na etapa 5 "
                f"ou reprocesse a voxelização após corrigir a curação dos pares."
            )

        # Divisão train/val/test por cluster (evita data leakage)
        # Usa o PDB ID (4 primeiros chars do pair_id) como chave de cluster
        if "pdb_id" in df.columns:
            df["pdb_cluster"] = df["pdb_id"].astype(str).str.upper()
        else:
            df["pdb_cluster"] = df["pair_id"].str[:4]
        clusters = sorted(df["pdb_cluster"].unique())
        rng      = random.Random(seed)
        rng.shuffle(clusters)

        n_total = len(clusters)
        n_train = int(n_total * split_ratio[0])
        n_val   = int(n_total * split_ratio[1])

        if split == "train":
            selected = set(clusters[:n_train])
        elif split == "val":
            selected = set(clusters[n_train:n_train + n_val])
        else:
            selected = set(clusters[n_train + n_val:])

        self.records = df[df["pdb_cluster"].isin(selected)].reset_index(drop=True)
        log.info(f"Dataset '{split}': {len(self.records)} pares "
                 f"({len(selected)} clusters PDB únicos)")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _random_rotate90(arr: np.ndarray) -> tuple:
        """Rotação aleatória de 90° em torno de um dos 3 eixos espaciais.

        Devolve (tensor_rotacionado, eixos, k) — os dois últimos para que a
        MESMA rotação seja aplicada às posições do ligante em
        _rotate_positions. Girar só o condicionamento destruiria a
        correspondência espacial que o modelo precisa aprender.

        O tensor tem shape (C, X, Y, Z). Rotacionamos apenas os eixos
        espaciais (1,2,3), preservando a ordem dos canais.
        Cada chamada aplica 0-3 rotações de 90° em um eixo aleatório,
        gerando até 24 orientações distintas por sítio.
        """
        axis_pairs = [(1,2), (1,3), (2,3)]  # pares de eixos espaciais
        ax = axis_pairs[np.random.randint(3)]
        k  = np.random.randint(4)           # 0, 90, 180 ou 270 graus
        return np.rot90(arr, k=k, axes=ax).copy(), ax, k

    @staticmethod
    def _rotate_positions(pos: "torch.Tensor", ax: tuple, k: int,
                          box_A: float) -> "torch.Tensor":
        """Aplica ao ligante a MESMA rotação sofrida pela grade do sítio.

        Sem isto a augmentation ensina geometria errada: o bolso gira, o
        ligante fica onde estava, e a correspondência espacial entre
        condicionamento e alvo — que é exatamente o que o modelo precisa
        aprender — é destruída em 3 de cada 4 amostras.

        A transformação vem da convenção do np.rot90 e foi conferida contra
        ela nos 9 casos (3 planos × 3 ângulos): para cada quarto de volta no
        plano (a, b), (u_a, u_b) → (L − u_b, u_a), com L o lado da caixa.
        É a versão contínua de (i, j) → (N−1−j, i).

        pos entra e sai em Å relativos à origem da grade.
        """
        if k == 0:
            return pos
        a, b = ax[0] - 1, ax[1] - 1     # eixo 0 do tensor é o canal
        out = pos.clone()
        for _ in range(k):
            ua = out[:, a].clone()
            ub = out[:, b].clone()
            out[:, a] = box_A - ub
            out[:, b] = ua
        return out

    def __getitem__(self, idx: int) -> Optional[dict]:
        row     = self.records.iloc[idx]
        pair_id = row["pair_id"]
        pair_dir = self.tensor_dir / pair_id

        # ── Tensor do sítio ──────────────────────────────────────────────────
        tensor_path = pair_dir / "site_tensor.npz"
        npz  = np.load(tensor_path, allow_pickle=True)
        arr  = npz["tensor"]                         # (X, Y, Z, C)
        arr  = np.transpose(arr, (3, 0, 1, 2))       # → (C, X, Y, Z)

        # Origem e passo da grade — necessários para trazer o ligante ao
        # referencial LOCAL do sítio (ver bloco de normalização abaixo).
        origin = np.asarray(npz["origin"], dtype=np.float32)
        voxel_size = float(np.asarray(npz["voxel_size"]).reshape(-1)[0]) \
            if "voxel_size" in npz else float(row.get("voxel_size_A", 1.0))
        box_A = arr.shape[1] * voxel_size          # lado da caixa em Å

        # Data augmentation: rotações aleatórias de 90° nos 3 eixos (apenas treino)
        # Multiplica o dataset efetivo por até 24x sem custo de armazenamento.
        # As rotações preservam a geometria vdW e o MEP do sítio.
        rot_ax, rot_k = None, 0
        if self.split == "train":
            arr, rot_ax, rot_k = self._random_rotate90(arr)

        site = torch.tensor(arr, dtype=torch.float32)

        # ── Grafo do ligante ──────────────────────────────────────────────────
        graph_pt  = pair_dir / "ligand_graph.pt"
        graph_npz = pair_dir / "ligand_graph.npz"

        if graph_pt.exists():
            graph = torch.load(graph_pt, weights_only=False)
        elif graph_npz.exists():
            graph = self._npz_to_pyg(np.load(graph_npz, allow_pickle=True), pair_id)
        else:
            return None

        # Sanitiza graph.x — cargas Gasteiger de metais podem ser inf/NaN
        # e propagam NaN por todo o GNN causando colapso do treino
        if graph is not None and graph.x is not None:
            if not torch.isfinite(graph.x).all():
                graph.x = torch.nan_to_num(graph.x, nan=0.0,
                                           posinf=0.0, neginf=0.0)

        # ── Referencial das posições do ligante ──────────────────────────────
        # As coordenadas gravadas no grafo são CRISTALOGRÁFICAS ABSOLUTAS:
        # cada ligante fica onde o depositante do PDB o colocou, com desvio
        # de ~50 Å por eixo entre pares e valores de −72 a +346 Å. O tensor
        # do sítio, por construção, é uma grade local ao bolso. Regredir a
        # coordenada absoluta a partir dela é pedir informação que não está
        # na entrada, e o modelo responde da única forma possível: prevendo
        # a média global. Foi o que aconteceu no treino anterior — erro
        # mediano de 35 Å, MSE normalizada ≈ 1.0 (variância explicada zero)
        # e, em consequência, 0% de moléculas válidas na geração.
        #
        # Aqui as posições passam para Å RELATIVOS À ORIGEM DA GRADE, que é
        # o mesmo referencial do tensor de entrada. A geração desfaz a
        # transformação somando a origem de volta (ver pipeline_generate.
        # generate_candidates) — a chave "pos_frame" gravada no checkpoint
        # é o que mantém os dois lados sincronizados.
        if graph is not None and getattr(graph, "pos", None) is not None:
            pos_rel = graph.pos - torch.tensor(origin, dtype=graph.pos.dtype)

            # Par cujo ligante não cabe na própria grade é dado corrompido
            # (na amostragem de 400 pares foi 1, com 172 Å de excesso).
            # Deixá-lo passar contamina a escala do termo de posição.
            folga = 2.0
            if bool(((pos_rel < -folga) | (pos_rel > box_A + folga)).any()):
                return None

            if rot_k:
                pos_rel = self._rotate_positions(pos_rel, rot_ax, rot_k, box_A)
            graph.pos = pos_rel

        return {
            "tensor":     site,
            "graph":      graph,
            "pair_id":    pair_id,
            "smiles":     row.get("ligand_smiles", ""),
            "voxel_size": float(row.get("voxel_size_A", 1.0)),
            "origin":     torch.tensor(origin, dtype=torch.float32),
            "box_A":      box_A,
        }

    @staticmethod
    def _npz_to_pyg(npz, pair_id: str) -> "PyGData":
        """Converte grafo salvo em .npz para PyGData."""
        x     = torch.tensor(npz["node_features"], dtype=torch.float32)
        pos   = torch.tensor(npz["positions"],     dtype=torch.float32)
        src   = torch.tensor(npz["edge_src"],      dtype=torch.long)
        dst   = torch.tensor(npz["edge_dst"],      dtype=torch.long)
        ea    = torch.tensor(npz["edge_features"], dtype=torch.float32)
        edge_index = torch.stack([src, dst], dim=0)
        smiles = str(npz["smiles"][0]) if len(npz["smiles"]) > 0 else ""
        return PyGData(x=x, edge_index=edge_index, edge_attr=ea,
                       pos=pos, smiles=smiles, pair_id=pair_id,
                       num_nodes=x.shape[0])

    @staticmethod
    def collate_fn(batch):
        """Agrupa itens do Dataset em mini-batches, filtrando e sanitizando NaN/Inf.

        `graphs` agora é uma LISTA de PyGData (não mais um PyG Batch
        mesclado) — o decoder autorregressivo precisa de acesso a cada
        molécula individualmente (via .atom_order) para construir os
        grafos parciais passo a passo, ver
        GNNDecoder.forward_teacher_forcing.
        """
        batch = [b for b in batch if b is not None]
        # Filtra tensores de sítio corrompidos
        batch = [b for b in batch if torch.isfinite(b["tensor"]).all()]
        # Requer atom_order — grafos gerados ANTES do decoder autorregressivo
        # (ou vindos do fallback .npz sem PyTorch em pipeline_voxelize) não
        # têm; reprocessar a voxelização resolve (ver process_pair).
        batch = [b for b in batch
                 if getattr(b["graph"], "atom_order", None) is not None]
        if not batch:
            return None

        tensors = torch.stack([b["tensor"] for b in batch])   # (B,C,X,Y,Z)
        graphs = []
        for b in batch:
            g = b["graph"]
            if not torch.isfinite(g.x).all():
                g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(g.pos).all():
                g.pos = torch.nan_to_num(g.pos, nan=0.0, posinf=0.0, neginf=0.0)
            graphs.append(g)

        # Descarta o batch inteiro se ainda houver NaN em algum grafo
        if not all(torch.isfinite(g.x).all() and torch.isfinite(g.pos).all() for g in graphs):
            return None

        return {
            "tensor":      tensors,
            "graphs":      graphs,
            "pair_ids":    [b["pair_id"]    for b in batch],
            "smiles":      [b["smiles"]     for b in batch],
            "voxel_sizes": torch.tensor([b["voxel_size"] for b in batch],
                                        dtype=torch.float32),  # (B,)
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER DO SÍTIO (3D-CNN)
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    """Bloco residual 3D com normalização em lote."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SiteEncoder(nn.Module):
    """Encoder CNN 3D que mapeia tensor (C,X,Y,Z) → (z_mean, z_logvar).

    Arquitetura:
        32³×5  →  [Conv+Pool] →  16³×32
                →  [Conv+Pool] →   8³×64
                →  [Conv+Pool] →   4³×128
                →  [Flatten]   →  8192
                →  [MLP]       →  z_mean (D), z_logvar (D)
    """

    def __init__(self, in_channels: int = 5, latent_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim

        # Arquitetura reduzida para datasets pequenos (< 1000 pares).
        # Removido bloco 4 e reduzidos canais: menos parâmetros = menos overfitting.
        # Capacidade original: ~4.2M params → reduzida: ~900k params
        self.encoder = nn.Sequential(
            # Bloco 1: 32 → 16
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.GELU(),
            ResBlock3D(16, dropout),
            nn.MaxPool3d(2),

            # Bloco 2: 16 → 8
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.GELU(),
            ResBlock3D(32, dropout),
            nn.MaxPool3d(2),

            # Bloco 3: 8 → 4
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.GELU(),
            ResBlock3D(64, dropout),
            nn.MaxPool3d(2),
        )

        self.flatten_dim = 64 * 4 * 4 * 4  # = 4096

        # +1 porque voxel_size é concatenado ao vetor achatado antes do MLP
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim + 1, 256),  # 4096 + 1 = 4097
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        self.fc_mean   = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor,
                voxel_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x          : (B, C, X, Y, Z) — tensor do sítio
            voxel_sizes : (B,) — tamanho do voxel em Å para cada sítio do batch
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)                        # (B, 4096)

        # Compatibilidade retroativa: checkpoints antigos foram treinados
        # sem concatenar voxel_size ao vetor achatado do sítio.
        fc_in_features = self.fc[0].in_features
        if fc_in_features == self.flatten_dim + 1:
            vs = voxel_sizes.unsqueeze(1)                # (B, 1)
            h = torch.cat([h, vs], dim=1)                # (B, 4097)
        elif fc_in_features != self.flatten_dim:
            raise RuntimeError(
                f"Dimensão inesperada no encoder.fc: {fc_in_features} "
                f"(esperado {self.flatten_dim} ou {self.flatten_dim + 1})."
            )

        h = self.fc(h)
        return self.fc_mean(h), self.fc_logvar(h)


# ══════════════════════════════════════════════════════════════════════════════
# DECODER DO LIGANTE (GNN)
# ══════════════════════════════════════════════════════════════════════════════

ATOM_TYPE_DIM = 13
BOND_TYPE_DIM = 4

# Número atômico por índice de ATOM_TYPES — mesma ordem de
# pipeline_voxelize.ATOM_TYPES = ["C","N","O","S","P","F","CL","BR","I",
# "FE","ZN","MG","OTHER"]. Usado só para popular o campo [17] (número
# atômico normalizado) ao reconstruir features de átomos já colocados no
# grafo parcial — ver _make_partial_features.
_ATOM_TYPE_Z = torch.tensor(
    [6, 7, 8, 16, 15, 9, 17, 35, 53, 26, 30, 12, 6], dtype=torch.float32
)

# Valência máxima por índice de ATOM_TYPES — mesma tabela de
# pipeline_generate.MAX_VALENCE, aqui indexada por TIPO (não por símbolo)
# para virar feature do nó e máscara de amostragem. "OTHER" recebe 4, o
# mesmo default usado do lado da geração.
_ATOM_TYPE_MAXVAL = torch.tensor(
    [4, 3, 2, 6, 5, 1, 1, 1, 1, 6, 4, 2, 4], dtype=torch.float32
)

# Ordem de ligação consumida por tipo (SINGLE, DOUBLE, TRIPLE, AROMATIC) —
# mesma convenção de pipeline_generate.BOND_ORDER_BUDGET.
_BOND_ORDER = torch.tensor([1.0, 2.0, 3.0, 1.5], dtype=torch.float32)

#: Versão da arquitetura do decoder, gravada no checkpoint.
#:   1 — cabeça de posição determinística (Huber sobre coordenada absoluta
#:       no referencial da grade), decoder sem geometria nem tipo de
#:       ligação nas entradas.
#:   2 — posição ancorada em átomo focal e modelada por mixture density
#:       network; geometria, valência restante e contagem de ligações nas
#:       features dos nós; arestas com tipo + distância; cabeças de foco e
#:       de carga formal.
#: Checkpoints da versão 1 são estruturalmente incompatíveis: os mesmos
#: campos do vetor de features têm outro significado, então carregá-los
#: com strict=False produziria um modelo silenciosamente errado. Ver
#: pipeline_generate.load_model.
ARCH_VERSION = 2

#: Nº de componentes gaussianas da MDN de posição (Pocket2Mol usa a mesma
#: ideia com n_component configurável em models/position.py).
N_POS_COMPONENTS = 6


def _make_partial_features(
    atom_type_idx: torch.Tensor,   # (n,) long
    charge:        torch.Tensor,   # (n,) float
    degree:        torch.Tensor,   # (n,) float/long — grau NO GRAFO PARCIAL
    node_feat_dim: int = 37,
    positions:     "torch.Tensor | None" = None,   # (n,3) — grafo PARCIAL
    valence_used:  "torch.Tensor | None" = None,   # (n,)  — ordem consumida
    bond_counts:   "torch.Tensor | None" = None,   # (n,4) — por tipo
) -> torch.Tensor:
    """Features (37-dim, mesmo layout de pipeline_voxelize.atom_features)
    de átomos JÁ COLOCADOS no grafo parcial — usadas como entrada do
    node_embed, tanto em treino (forward_teacher_forcing) quanto em
    geração (LigandCVAE.sample).

    Popula só os campos igualmente conhecíveis nos dois lados: tipo do
    átomo [0:13], número atômico [17] (derivado do tipo), carga [19], grau
    [21] (recomputado do grafo PARCIAL a cada passo — não é o grau da
    molécula final). Os demais campos do vetor de 37 (hibridização
    [13:17], carga formal [18], H implícitos [20], aromaticidade [22],
    anel [23], quiralidade [24:28]) vêm do RDKit da molécula real completa
    em treino, mas são desconhecidos em geração (dependem da molécula
    final, ainda não decidida) — por isso ficam ZERADOS NOS DOIS LADOS.
    Sem essa máscara, o treino ensinaria o decoder a depender de sinais
    que não existem durante a geração — o mesmo tipo de vazamento
    treino↔inferência corrigido nos canais 3-4 do tensor do sítio.

    Nuance aceita (exposure bias): o campo de carga [19] em treino usa o
    valor REAL (teacher forcing); em geração usa a carga que o PRÓPRIO
    modelo previu para esse átomo no passo em que foi colocado. Essa
    diferença (contexto real vs. autogerado) é inerente a qualquer modelo
    autorregressivo com teacher forcing — não é um vazamento binário como
    os outros dois casos, é o trade-off padrão da literatura (exposure
    bias), com mitigações conhecidas (scheduled sampling) possíveis como
    refinamento futuro.
    """
    n = atom_type_idx.shape[0]
    device = atom_type_idx.device
    clipped = atom_type_idx.clamp(max=ATOM_TYPE_DIM - 1)
    x = torch.zeros((n, node_feat_dim), dtype=torch.float32, device=device)
    x[torch.arange(n, device=device), clipped] = 1.0
    x[:, 17] = _ATOM_TYPE_Z.to(device)[clipped] / 100.0
    x[:, 19] = charge.float()
    x[:, 21] = (degree.float() / 6.0).clamp(0.0, 1.0)

    # ── Campos [28:37]: geometria e saturação do grafo parcial ────────────
    # Eram os slots "reservados para expansão futura". Sem eles o decoder
    # era cego para a própria geometria que estava construindo: head_pos
    # recebia só [média do grafo ; z], não tinha como saber onde havia
    # colocado o átomo anterior e, sob Huber, colapsava para a média
    # condicional (raio de giro gerado ~0,3 Å contra ~3,3 Å real).
    #
    # Todos os campos aqui são igualmente conhecíveis em treino e geração —
    # derivam só do que JÁ foi colocado, nunca da molécula final —, então a
    # paridade treino/inferência das outras features é preservada.
    max_val = _ATOM_TYPE_MAXVAL.to(device)[clipped]

    if positions is not None and n > 0:
        pos = positions.float().to(device)
        # Relativo ao centroide dos átomos já colocados: invariante a
        # translação, que é o que o tensor do sítio já fixa. A escala /10
        # mantém os valores na mesma ordem de grandeza das demais features.
        centroid = pos.mean(dim=0, keepdim=True)
        rel = pos - centroid
        x[:, 28:31] = (rel / 10.0).clamp(-1.0, 1.0)
        x[:, 31] = (rel.norm(dim=-1) / 10.0).clamp(0.0, 1.0)

    if valence_used is not None and n > 0:
        remaining = (max_val - valence_used.float().to(device)).clamp(min=0.0)
        x[:, 32] = (remaining / 6.0).clamp(0.0, 1.0)
    else:
        x[:, 32] = (max_val / 6.0).clamp(0.0, 1.0)

    if bond_counts is not None and n > 0:
        x[:, 33:37] = (bond_counts.float().to(device) / 4.0).clamp(0.0, 1.0)

    return x


def _build_adjacency(graph) -> dict:
    """dict: índice_original_do_átomo -> {vizinho_original: tipo_de_ligação}.

    A partir de graph.edge_index/edge_attr (bidirecional — cada ligação
    aparece como (i,j) e (j,i); o tipo de ligação é o argmax dos primeiros
    BOND_TYPE_DIM canais de edge_attr, mesma convenção de
    pipeline_voxelize.bond_features/mol_to_graph).
    """
    adj: dict = {}
    ei, ea = graph.edge_index, graph.edge_attr
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        bt = int(torch.argmax(ea[k, :BOND_TYPE_DIM]).item())
        adj.setdefault(i, {})[j] = bt
    return adj


class GNNDecoder(nn.Module):
    """Decoder GNN AUTORREGRESSIVO: constrói o grafo do ligante um átomo por
    vez, condicionado no grafo PARCIAL já construído (mensagens GAT rodam
    sobre um grafo que cresce a cada passo, não sobre uma topologia fixa)
    + no vetor latente z do sítio.

    Substitui a versão anterior — que rodava uma única passada sobre a
    topologia de um "template graph" externo (o próprio ligante de
    referência ou um grafo aleatório do treino) e só reclassificava
    tipos/posições sobre essa topologia fixa, sem nunca criar ou remover
    átomos, e com bond_exist treinado de forma trivial (só via arestas do
    template, sempre positivas — nenhum par negativo real). Esta versão diz
    respeito às duas limitações:
      - número de átomos: decisão do modelo a cada passo (cabeça
        parar/continuar), não herdado de um template.
      - topologia: a cada passo, o novo átomo é testado contra TODOS os
        átomos já existentes no grafo parcial — amostragem negativa real
        (a maioria dos pares não é ligação; só uma minoria é).

    Ordem de construção para teacher forcing: ver
    pipeline_voxelize.compute_site_anchored_atom_order (ancorada no sítio,
    estilo Pocket2Mol — Peng et al., 2022). Cada grafo de treino já vem com
    `.atom_order` pré-computado (ver pipeline_voxelize.process_pair).

    Paralelismo: dentro de uma molécula o passo t depende do grafo parcial
    do passo t-1 (sequencial); entre moléculas do mesmo batch, cada passo é
    computado para todas simultaneamente (uma passada em lote por passo,
    não uma por molécula) — ver forward_teacher_forcing (treino) e
    pipeline_generate para a geração em lockstep.
    """

    def __init__(
        self,
        latent_dim:   int = 256,
        hidden_dim:   int = 256,
        n_atom_types: int = ATOM_TYPE_DIM,
        n_bond_types: int = BOND_TYPE_DIM,
        n_gnn_layers: int = 4,
        node_feat_dim: int = 37,
        dropout:      float = 0.2,
        use_ecfp4:    bool = False,
        ecfp4_bits:   int  = 512,
        n_pos_components: int = N_POS_COMPONENTS,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.hidden_dim     = hidden_dim
        self.n_atom_types   = n_atom_types
        self.n_bond_types   = n_bond_types
        self.node_feat_dim  = node_feat_dim
        self.use_ecfp4      = use_ecfp4
        self.ecfp4_bits     = ecfp4_bits

        # ECFP4 DESATIVADO por padrão — ver nota original: ancora o decoder
        # em fingerprints já vistos no treino, reduzindo a diversidade dos
        # ligantes gerados. Quando ativo, é concatenado a z UMA VEZ (não
        # depende do grafo parcial, então fica fora do loop de passos).
        z_input_dim = latent_dim + ecfp4_bits if use_ecfp4 else latent_dim

        self.z_proj = nn.Sequential(
            nn.Linear(z_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # edge_dim: o passo parcial agora carrega features de aresta
        # [one-hot do tipo de ligação (4) ; distância/2 Å (1)]. Antes o
        # GATConv era chamado sem edge_attr — o decoder não via sequer o
        # tipo das ligações que ele próprio já havia decidido, quanto mais
        # a geometria.
        self.edge_feat_dim = n_bond_types + 1
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4,
                    dropout=dropout, concat=True,
                    edge_dim=self.edge_feat_dim)
            for _ in range(n_gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_gnn_layers)
        ])

        # Contexto de cada passo = [pool(grafo parcial) ; z_proj(z)]
        ctx_dim = hidden_dim * 2

        self.head_stop = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head_atom_type = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, n_atom_types)
        )
        self.head_charge = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        # ── Carga FORMAL (−1/0/+1) ────────────────────────────────────────
        # head_charge acima prevê carga de Gasteiger (parcial, contínua),
        # que não serve para Chem.Atom.SetFormalCharge. Esta cabeça prevê a
        # carga formal de verdade, que a montagem da molécula precisa para
        # que grupos legitimamente carregados sobrevivam à sanitização.
        self.head_fcharge = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.GELU(),
            nn.Linear(64, 3)
        )

        # ── Átomo FOCAL ───────────────────────────────────────────────────
        # Escolhe, entre os átomos já colocados, aquele a partir do qual o
        # próximo cresce — o mesmo papel do focal_net do PocketFlow e da
        # seleção de foco do Pocket2Mol. É o que dá uma âncora concreta à
        # posição e o ponto onde a máscara de valência age.
        self.head_focal = nn.Sequential(
            nn.Linear(hidden_dim + ctx_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ── Posição: mixture density network RELATIVA ao focal ────────────
        # Substitui a regressão determinística da coordenada absoluta. Sob
        # incerteza multimodal (o próximo átomo pode ir para várias direções
        # igualmente válidas), a regressão com Huber devolve a média — que
        # é justamente o ponto onde NÃO há átomo nenhum. A MDN representa
        # os modos e é amostrada, não promediada. Mesma formulação de
        # Pocket2Mol (models/position.py): mu relativo ao focal, sigma por
        # componente e por eixo, pesos pi.
        self.n_pos_components = n_pos_components
        self.head_pos_mdn = nn.Sequential(
            nn.Linear(hidden_dim + ctx_dim, 128), nn.GELU(),
            nn.Linear(128, n_pos_components * 7)   # 3 mu + 3 logsigma + 1 pi
        )
        # Embedding usado no lugar de h_focal quando não existe focal
        # (primeiro átomo da molécula, ou reinício de fragmento): nesses
        # casos a MDN prevê a posição ABSOLUTA no referencial da grade,
        # com pos_focal = 0. Treino e geração usam a mesma convenção.
        self.no_focal_embed = nn.Parameter(torch.zeros(hidden_dim))

        # Cabeça de ligação: [embedding do átomo JÁ EXISTENTE ; contexto do
        # NOVO átomo] — uma predição por par (novo, existente_i).
        pair_dim = hidden_dim + ctx_dim
        self.head_bond_type = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bond_types)
        )
        self.head_bond_exist = nn.Sequential(
            nn.Linear(pair_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    # ── Núcleo: um passo, para um lote de grafos parciais já concatenados ──
    def _encode_partial(self, z_input, partial_x, partial_edge_index,
                         partial_batch_vec, n_active, partial_edge_attr=None):
        """z_input: (n_active, z_input_dim). partial_x/edge_index/batch_vec:
        grafo parcial "solto" (não é PyG Batch — tensores concatenados +
        vetor de pertencimento manual, ver forward_teacher_forcing), pode
        ser vazio (t=0, nenhum átomo colocado ainda em nenhuma molécula
        ativa). Retorna h (nós, pode ser vazio) e ctx (n_active, 2*hidden).
        """
        z_proj = self.z_proj(z_input)
        if partial_x.shape[0] == 0:
            pooled = torch.zeros_like(z_proj)
            h = z_proj.new_zeros((0, self.hidden_dim))
        else:
            z_node = z_proj[partial_batch_vec]
            h = self.node_embed(partial_x) + z_node
            if partial_edge_attr is None and partial_edge_index.shape[1] > 0:
                partial_edge_attr = partial_edge_index.new_zeros(
                    (partial_edge_index.shape[1], self.edge_feat_dim),
                    dtype=h.dtype,
                )
            for gnn, ln in zip(self.gnn_layers, self.layer_norms):
                h = ln(h + gnn(h, partial_edge_index, edge_attr=partial_edge_attr))
            pooled = global_mean_pool(h, partial_batch_vec, size=n_active)
        ctx = torch.cat([pooled, z_proj], dim=-1)
        return h, ctx

    def forward_step(self, z_input, partial_x, partial_edge_index,
                      partial_batch_vec, n_active, partial_edge_attr=None):
        """Prediz parar/tipo/carga/carga formal do PRÓXIMO átomo e pontua
        cada átomo já colocado como candidato a FOCAL, para as n_active
        moléculas ativas neste passo. Retorna também (h, ctx) — necessários
        para predict_bonds e para a MDN de posição.

        A posição NÃO sai daqui: ela depende de qual átomo focal foi
        escolhido, e essa escolha difere entre treino (focal real, vindo da
        ordem de construção) e geração (focal amostrado sob máscara de
        valência). Ver position_mdn().
        """
        h, ctx = self._encode_partial(
            z_input, partial_x, partial_edge_index, partial_batch_vec,
            n_active, partial_edge_attr
        )
        if h.shape[0] > 0:
            focal_logits = self.head_focal(
                torch.cat([h, ctx[partial_batch_vec]], dim=-1)
            ).squeeze(-1)
        else:
            focal_logits = h.new_zeros((0,))
        return {
            "stop_logit":       self.head_stop(ctx).squeeze(-1),
            "atom_type_logits": self.head_atom_type(ctx),
            "charge":           self.head_charge(ctx).squeeze(-1),
            "fcharge_logits":   self.head_fcharge(ctx),
            "focal_logits":     focal_logits,
            "_h": h, "_ctx": ctx,
        }

    def _focal_feature(self, h, focal_node_idx, n_active):
        """Embedding do átomo focal de cada molécula ativa.

        focal_node_idx: (n_active,) índices em h, ou −1 quando a molécula
        não tem focal neste passo (primeiro átomo, ou reinício de
        fragmento). Nesses casos entra `no_focal_embed` e a MDN passa a
        prever a posição ABSOLUTA no referencial da grade.
        """
        feat = self.no_focal_embed.unsqueeze(0).expand(n_active, -1).contiguous()
        if h.shape[0] > 0:
            valid = focal_node_idx >= 0
            if bool(valid.any()):
                feat = feat.clone()
                feat[valid] = h[focal_node_idx[valid]]
        return feat

    def position_mdn(self, h, ctx, focal_node_idx, focal_pos):
        """Parâmetros da mixture density network da posição do próximo átomo.

        Args:
            focal_node_idx: (n_active,) índice do focal em h, ou −1.
            focal_pos:      (n_active,3) posição do focal (zeros se −1).

        Returns:
            mu_abs (n_active, K, 3) — médias JÁ em coordenadas absolutas
                (mu relativo + posição do focal, como em Pocket2Mol);
            sigma  (n_active, K, 3);
            pi_logits (n_active, K) — pesos SEM softmax (a loss usa
                log_softmax, a amostragem usa softmax).
        """
        n_active = ctx.shape[0]
        h_focal = self._focal_feature(h, focal_node_idx, n_active)
        raw = self.head_pos_mdn(torch.cat([h_focal, ctx], dim=-1))
        K = self.n_pos_components
        mu_rel   = raw[:, :K * 3].view(n_active, K, 3)
        logsigma = raw[:, K * 3:K * 6].view(n_active, K, 3).clamp(-5.0, 3.0)
        pi_logits = raw[:, K * 6:]
        mu_abs = mu_rel + focal_pos.unsqueeze(1)
        return mu_abs, logsigma.exp(), pi_logits

    def predict_bonds(self, h, ctx, batch_vec_h, active_next_local):
        """h/batch_vec_h: nós/pertencimento do grafo parcial DE TODAS as
        n_active moléculas do passo (saída de forward_step). ctx: contexto
        de todas as n_active moléculas. active_next_local: LongTensor
        (índices dentro de 0..n_active-1, em ordem crescente) das
        moléculas que têm um próximo átomo real neste passo — só elas
        entram na predição de ligação.

        Retorna bond_exist_logits, bond_type_logits — achatados por TODOS
        os pares (molécula ativa-com-próximo, átomo já existente dela),
        na ordem: para cada posição de active_next_local (nessa ordem),
        os átomos existentes dela na ordem em que aparecem em
        partial_batch_vec (ou seja, na ordem de atom_order[:t]). Essa
        ordem é a mesma que forward_teacher_forcing usa para montar os
        targets — ver teste de regressão no changelog desta função.
        """
        empty_logit = h.new_zeros((0,))
        empty_type  = h.new_zeros((0, self.n_bond_types))
        if active_next_local.numel() == 0 or h.shape[0] == 0:
            return empty_logit, empty_type

        n_active = ctx.shape[0]
        lut = torch.full((n_active,), -1, dtype=torch.long, device=h.device)
        lut[active_next_local] = torch.arange(active_next_local.numel(), device=h.device)
        mapped = lut[batch_vec_h]
        keep = mapped >= 0
        if keep.sum() == 0:
            return empty_logit, empty_type

        h_sel = h[keep]
        pair_mol_local = mapped[keep]                       # 0..len(active_next_local)-1
        ctx_sel = ctx[active_next_local][pair_mol_local]      # contexto do novo átomo, repetido por par

        pair_h = torch.cat([h_sel, ctx_sel], dim=-1)
        bond_type_logits  = self.head_bond_type(pair_h)
        bond_exist_logits = self.head_bond_exist(pair_h).squeeze(-1)
        return bond_exist_logits, bond_type_logits

    # ── Treino: teacher forcing completo, todos os passos, todo o batch ────
    def forward_teacher_forcing(self, z: torch.Tensor, graphs: list,
                                 ecfp4_batch: "torch.Tensor | None" = None) -> dict:
        """z: (B, latent_dim), um por molécula, na MESMA ordem de `graphs`.
        graphs: lista de PyGData (um por molécula), cada um com .x
        (n_i,37), .edge_index (2,E_i), .edge_attr (E_i,8), .pos (n_i,3),
        .atom_order (n_i,) LongTensor (ver pipeline_voxelize).

        Itera t = 0..max(n_átomos)-1, construindo o grafo parcial de cada
        molécula ainda ativa (n_átomos >= t) e comparando a predição do
        passo com o estado real do treino (teacher forcing). Retorna um
        dict com tudo ACHATADO por todos os passos e moléculas — pronto
        para cvae_loss (sem necessidade de padding/máscara adicional, já
        que só os itens realmente ativos em cada passo entram nos
        tensores retornados).
        """
        device = z.device
        B = len(graphs)
        n_atoms = [int(g.atom_order.shape[0]) for g in graphs]
        max_t = max(n_atoms) if n_atoms else 0
        adjacencies = [_build_adjacency(g) for g in graphs]

        if self.use_ecfp4 and ecfp4_batch is not None:
            z_input_full = torch.cat([z, ecfp4_batch], dim=-1)
        else:
            z_input_full = z

        acc = {k: [] for k in (
            "stop_logits", "stop_targets", "atom_logits", "atom_targets",
            "charges", "charge_targets", "pos_targets",
            "pos_mu", "pos_sigma", "pos_pi", "focal_nll",
            "fcharge_logits", "fcharge_targets",
            "bond_exist", "bond_exist_targets", "bond_logits", "bond_targets",
        )}

        for t in range(0, max_t + 1):
            active_stop = [i for i in range(B) if n_atoms[i] >= t]
            if not active_stop:
                break
            n_active = len(active_stop)

            xs, eis, eas, batch_vec_parts = [], [], [], []
            pos_maps, offsets = {}, {}
            offset = 0
            for local_i, orig_i in enumerate(active_stop):
                g = graphs[orig_i]
                if t == 0:
                    continue
                dev_g = g.x.device
                order_t = g.atom_order[:t]
                pos_map = {int(idx): k for k, idx in enumerate(order_t.tolist())}
                pos_maps[local_i] = pos_map
                offsets[local_i] = offset
                ei, ea = g.edge_index, g.edge_attr
                keep_cols = [k for k in range(ei.shape[1])
                             if int(ei[0, k]) in pos_map and int(ei[1, k]) in pos_map]

                # Posições dos átomos JÁ colocados: entram como feature do nó
                # (relativas ao centroide) e como distância nas arestas. Sem
                # isso o decoder não via a própria geometria — causa raiz do
                # colapso da cabeça de posição na arquitetura 1.
                pos_t = g.pos[order_t].float()

                # Grau NO GRAFO PARCIAL (não o grau da molécula final) — ver
                # _make_partial_features. Calculado nos índices LOCAIS
                # (0..t-1) antes de aplicar o offset do batch.
                degree_t = torch.zeros(t, dtype=torch.float32, device=dev_g)
                valence_t = torch.zeros(t, dtype=torch.float32, device=dev_g)
                bcounts_t = torch.zeros((t, BOND_TYPE_DIM), dtype=torch.float32,
                                        device=dev_g)
                if keep_cols:
                    idx_t = torch.tensor(keep_cols, dtype=torch.long, device=dev_g)
                    sub_ei_local = ei[:, idx_t].clone()
                    for col in range(sub_ei_local.shape[1]):
                        sub_ei_local[0, col] = pos_map[int(sub_ei_local[0, col])]
                        sub_ei_local[1, col] = pos_map[int(sub_ei_local[1, col])]
                    degree_t = torch.bincount(sub_ei_local[0], minlength=t).float()

                    # Tipo de cada aresta mantida (mesma convenção de
                    # _build_adjacency: argmax dos BOND_TYPE_DIM primeiros
                    # canais de edge_attr).
                    bt_col = ea[idx_t][:, :BOND_TYPE_DIM].argmax(dim=-1)
                    bt_onehot = torch.zeros((bt_col.shape[0], BOND_TYPE_DIM),
                                            dtype=torch.float32, device=dev_g)
                    bt_onehot[torch.arange(bt_col.shape[0], device=dev_g), bt_col] = 1.0
                    order_val = _BOND_ORDER.to(dev_g)[bt_col]
                    # edge_index é bidirecional: somar em [0] dá, para cada
                    # átomo, a ordem total consumida e a contagem por tipo.
                    valence_t = valence_t.index_add(0, sub_ei_local[0], order_val)
                    bcounts_t = bcounts_t.index_add(0, sub_ei_local[0], bt_onehot)

                    d_edge = (pos_t[sub_ei_local[0]] - pos_t[sub_ei_local[1]]
                              ).norm(dim=-1, keepdim=True) / 2.0
                    eis.append(sub_ei_local + offset)
                    eas.append(torch.cat([bt_onehot, d_edge.clamp(0.0, 2.0)], dim=-1))

                atom_type_t = g.x[order_t, :ATOM_TYPE_DIM].argmax(dim=-1)
                charge_t    = g.x[order_t, 19]
                xs.append(_make_partial_features(
                    atom_type_t, charge_t, degree_t,
                    positions=pos_t, valence_used=valence_t, bond_counts=bcounts_t))
                batch_vec_parts.append(
                    torch.full((t,), local_i, dtype=torch.long, device=dev_g))
                offset += t

            if xs:
                partial_x = torch.cat(xs, dim=0).to(device)
                partial_batch_vec = torch.cat(batch_vec_parts, dim=0).to(device)
                partial_edge_index = (torch.cat(eis, dim=1).to(device)
                                      if eis else torch.zeros((2, 0), dtype=torch.long, device=device))
                partial_edge_attr = (torch.cat(eas, dim=0).to(device)
                                     if eas else z.new_zeros((0, self.edge_feat_dim)))
            else:
                partial_x = z.new_zeros((0, self.node_feat_dim))
                partial_batch_vec = torch.zeros(0, dtype=torch.long, device=device)
                partial_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                partial_edge_attr = z.new_zeros((0, self.edge_feat_dim))

            active_stop_t = torch.tensor(active_stop, dtype=torch.long, device=device)
            z_input_active = z_input_full[active_stop_t]
            step_out = self.forward_step(
                z_input_active, partial_x, partial_edge_index,
                partial_batch_vec, n_active, partial_edge_attr
            )

            stop_target = torch.tensor(
                [1.0 if n_atoms[i] == t else 0.0 for i in active_stop], device=device
            )
            acc["stop_logits"].append(step_out["stop_logit"])
            acc["stop_targets"].append(stop_target)

            active_next_local = torch.tensor(
                [jl for jl, i in enumerate(active_stop) if n_atoms[i] > t],
                dtype=torch.long, device=device,
            )
            if active_next_local.numel() == 0:
                continue

            anl_list = active_next_local.tolist()
            next_orig = [int(graphs[active_stop[jl]].atom_order[t]) for jl in anl_list]
            next_feats = torch.stack([
                graphs[active_stop[jl]].x[no] for jl, no in zip(anl_list, next_orig)
            ]).to(device)
            next_pos_t = torch.stack([
                graphs[active_stop[jl]].pos[no] for jl, no in zip(anl_list, next_orig)
            ]).to(device)

            acc["atom_logits"].append(step_out["atom_type_logits"][active_next_local])
            acc["atom_targets"].append(next_feats[:, :ATOM_TYPE_DIM].argmax(dim=-1))
            acc["charges"].append(step_out["charge"][active_next_local])
            acc["charge_targets"].append(next_feats[:, 19])

            # Carga FORMAL (slot 18 do vetor de features) discretizada em
            # {−1, 0, +1} → classes {0, 1, 2}. Cargas fora dessa faixa são
            # raras em ligantes depositados e são saturadas no extremo.
            acc["fcharge_logits"].append(step_out["fcharge_logits"][active_next_local])
            acc["fcharge_targets"].append(
                (next_feats[:, 18].round().clamp(-1.0, 1.0) + 1.0).long())

            # ── Átomo focal real e MDN de posição ─────────────────────────
            # O focal do próximo átomo é o vizinho REAL dele que foi
            # colocado PRIMEIRO na ordem de construção. É a âncora que torna
            # a posição uma quantidade local (~1,5 Å) em vez de uma
            # coordenada absoluta no referencial da grade.
            focal_idx_list, focal_pos_list = [], []
            for jl, no in zip(anl_list, next_orig):
                orig_i = active_stop[jl]
                g = graphs[orig_i]
                adj = adjacencies[orig_i]
                pmap = pos_maps.get(jl, {})
                cand_local = [pmap[nb] for nb in adj.get(no, {}) if nb in pmap]
                if cand_local:
                    f_local = min(cand_local)          # o colocado mais cedo
                    f_orig  = int(g.atom_order[f_local])
                    focal_idx_list.append(offsets[jl] + f_local)
                    focal_pos_list.append(g.pos[f_orig].float())
                else:
                    # Primeiro átomo, ou reinício de fragmento: sem âncora.
                    # A MDN prevê a posição absoluta (focal_pos = 0).
                    focal_idx_list.append(-1)
                    focal_pos_list.append(
                        torch.zeros(3, dtype=torch.float32, device=device))

            focal_idx_t = torch.tensor(focal_idx_list, dtype=torch.long, device=device)
            focal_pos_t = torch.stack(focal_pos_list).to(device)

            mu, sigma, pi_logits = self.position_mdn(
                step_out["_h"], step_out["_ctx"][active_next_local],
                focal_idx_t, focal_pos_t,
            )
            acc["pos_mu"].append(mu)
            acc["pos_sigma"].append(sigma)
            acc["pos_pi"].append(pi_logits)
            acc["pos_targets"].append(next_pos_t)

            # Perda do foco: entropia cruzada DENTRO de cada molécula (os
            # candidatos a focal são os átomos já colocados daquela
            # molécula, não do batch inteiro) — por isso é acumulada aqui
            # como NLL por amostra, já segmentada.
            focal_logits_all = step_out["focal_logits"]
            for k, jl in enumerate(anl_list):
                f_global = focal_idx_list[k]
                if f_global < 0 or t == 0:
                    continue
                seg = focal_logits_all[offsets[jl]: offsets[jl] + t]
                if seg.numel() == 0:
                    continue
                acc["focal_nll"].append(
                    -F.log_softmax(seg, dim=0)[f_global - offsets[jl]].view(1))

            bond_exist_logits, bond_type_logits = self.predict_bonds(
                step_out["_h"], step_out["_ctx"], partial_batch_vec, active_next_local
            )
            if bond_exist_logits.numel() > 0:
                bond_exist_t, bond_type_t = [], []
                for jl, no in zip(anl_list, next_orig):
                    orig_i = active_stop[jl]
                    adj = adjacencies[orig_i]
                    order_t = graphs[orig_i].atom_order[:t].tolist()
                    for exist_o in order_t:
                        if exist_o in adj.get(no, {}):
                            bond_exist_t.append(1.0)
                            bond_type_t.append(adj[no][exist_o])
                        else:
                            bond_exist_t.append(0.0)
                            bond_type_t.append(0)   # dummy — mascarado via bond_exist na loss
                acc["bond_exist"].append(bond_exist_logits)
                acc["bond_exist_targets"].append(torch.tensor(bond_exist_t, device=device))
                acc["bond_logits"].append(bond_type_logits)
                acc["bond_targets"].append(torch.tensor(bond_type_t, dtype=torch.long, device=device))

        def _cat(key, tail_shape=()):
            lst = acc[key]
            if lst:
                return torch.cat(lst, dim=0)
            return z.new_zeros((0,) + tail_shape)

        return {
            "stop_logits":        _cat("stop_logits"),
            "stop_targets":       _cat("stop_targets"),
            "atom_logits":        _cat("atom_logits", (self.n_atom_types,)),
            "atom_targets":       _cat("atom_targets").long(),
            "charges":            _cat("charges"),
            "charge_targets":     _cat("charge_targets"),
            "pos_targets":        _cat("pos_targets", (3,)),
            "pos_mu":             _cat("pos_mu", (self.n_pos_components, 3)),
            "pos_sigma":          _cat("pos_sigma", (self.n_pos_components, 3)),
            "pos_pi":             _cat("pos_pi", (self.n_pos_components,)),
            "focal_nll":          _cat("focal_nll"),
            "fcharge_logits":     _cat("fcharge_logits", (3,)),
            "fcharge_targets":    _cat("fcharge_targets").long(),
            "bond_exist":         _cat("bond_exist"),
            "bond_exist_targets": _cat("bond_exist_targets"),
            "bond_logits":        _cat("bond_logits", (self.n_bond_types,)),
            "bond_targets":       _cat("bond_targets").long(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CVAE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def reparametrize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Truque de reparametrização do VAE: z = μ + ε·σ, ε ~ N(0,I)."""
    if not torch.is_grad_enabled():
        return mean
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


class LigandCVAE(nn.Module):
    """Variational Auto-Encoder Condicional para geração de ligantes.

    Encoder:    SiteEncoder  — tensor do sítio → (z_mean, z_logvar)
    Decoder:    GNNDecoder   — z + grafo real → predições (teacher forcing)
    Gerador:    sample()     — z ~ N(0,I) + sítio → novo grafo (inferência)
    """

    def __init__(
        self,
        in_channels:  int   = 5,
        latent_dim:   int   = 256,
        hidden_dim:   int   = 256,
        n_atom_types: int   = 13,
        n_bond_types: int   = 4,
        n_gnn_layers: int   = 4,
        dropout:      float = 0.2,
        use_ecfp4:    bool  = False,
        ecfp4_bits:   int   = 512,
        n_pos_components: int = N_POS_COMPONENTS,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = SiteEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_atom_types=n_atom_types,
            n_bond_types=n_bond_types,
            n_gnn_layers=n_gnn_layers,
            dropout=dropout,
            use_ecfp4=use_ecfp4,
            ecfp4_bits=ecfp4_bits,
            n_pos_components=n_pos_components,
        )

    def _init_weights(self):
        """Inicialização de pesos para prevenir NaN no início do treino.
        Usa Xavier uniform para camadas lineares e He para convoluções.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, site_tensor: torch.Tensor, graphs: list,
            voxel_sizes: torch.Tensor) -> dict:
        """Passagem completa de treino — teacher forcing autorregressivo.

        graphs: lista de PyGData (um por molécula do batch, MESMA ordem das
            linhas de site_tensor/voxel_sizes), cada um já com .atom_order
            pré-computado (ver
            pipeline_voxelize.compute_site_anchored_atom_order).

        Returns dict com predições+targets de TODOS os passos (achatados,
        ver GNNDecoder.forward_teacher_forcing) + z_mean + z_logvar.
        """
        z_mean, z_logvar = self.encoder(site_tensor, voxel_sizes)
        # Clipa z_logvar para prevenir exp(logvar) → inf
        z_logvar = torch.clamp(z_logvar, -10, 10)
        z = reparametrize(z_mean, z_logvar)

        ecfp4_batch = None
        if self.decoder.use_ecfp4 and all(
            hasattr(g, "ecfp4") and g.ecfp4 is not None for g in graphs
        ):
            ecfp4_batch = torch.stack([g.ecfp4 for g in graphs]).to(z.device)

        preds = self.decoder.forward_teacher_forcing(z, graphs, ecfp4_batch)
        preds["z_mean"]   = z_mean
        preds["z_logvar"] = z_logvar
        return preds

    @torch.no_grad()
    def sample(
        self,
        site_tensor:     torch.Tensor,
        n_samples:       int   = 1,
        temperature:     float = 1.0,
        voxel_sizes:     "torch.Tensor | None" = None,
        max_atoms:       int   = 60,
        stop_threshold:  float = 0.5,
        bond_threshold:  float = 0.5,
        ecfp4:           "torch.Tensor | None" = None,
    ) -> list:
        """Gera novos ligantes condicionados no sítio — construção
        SEQUENCIAL autorregressiva (um átomo por vez, condicionado no
        grafo parcial já decidido), com todos os n_samples candidatos
        avançando EM LOCKSTEP: a cada passo, uma única passada do decoder
        cobre todos os candidatos ainda ativos simultaneamente (candidatos
        que já pararam saem da máscara ativa e não custam mais nada). É a
        mesma estratégia de geração em lote usada por LLMs — o custo real
        é ~max_atoms passadas pequenas, não n_samples passadas
        sequenciais (ver docstring de GNNDecoder sobre paralelismo).

        Args:
            site_tensor : (1, C, X, Y, Z) — tensor do sítio alvo
            n_samples   : quantos candidatos gerar (em paralelo/lockstep)
            temperature : controla diversidade (> 1 = mais diverso) —
                escala tanto o ruído somado a z_mean quanto a softmax do
                tipo de átomo
            max_atoms   : teto de átomos por candidato (mesmo valor do
                filtro do dataset de treino, default 60)
            stop_threshold : probabilidade mínima da cabeça "parar" para
                encerrar um candidato
            bond_threshold : probabilidade mínima da cabeça bond_exist
                para aceitar uma ligação

        Returns:
            Lista de n_samples dicts:
              {"atom_types": LongTensor(n_i,), "charges": FloatTensor(n_i,),
               "positions": FloatTensor(n_i,3),
               "bonds": list[(i, j, bond_type)]}  — índices LOCAIS 0..n_i-1
            prontos para pipeline_generate.preds_to_mol montar a molécula.
        """
        self.eval()
        device = site_tensor.device
        if voxel_sizes is None:
            voxel_sizes = torch.ones(site_tensor.shape[0], device=device)
        z_mean, _ = self.encoder(site_tensor, voxel_sizes)   # (1, latent_dim)
        z_base = z_mean.repeat(n_samples, 1)
        z = z_base + temperature * torch.randn_like(z_base)

        z_input_full = z
        if self.decoder.use_ecfp4 and ecfp4 is not None:
            z_input_full = torch.cat([z, ecfp4.repeat(n_samples, 1)], dim=-1)

        atom_types: list = [[] for _ in range(n_samples)]
        charges:    list = [[] for _ in range(n_samples)]
        fcharges:   list = [[] for _ in range(n_samples)]   # carga FORMAL (−1/0/+1)
        positions:  list = [[] for _ in range(n_samples)]
        bonds:      list = [[] for _ in range(n_samples)]   # (i, j, bond_type) locais
        # Ordem de ligação já consumida e contagem por tipo, por átomo —
        # alimentam as features do nó e a máscara de valência da escolha do
        # focal. Mantidos incrementalmente para não recomputar a cada passo.
        valence_used: list = [[] for _ in range(n_samples)]
        bond_counts:  list = [[] for _ in range(n_samples)]
        finished = [False] * n_samples

        for t in range(max_atoms):
            active = [c for c in range(n_samples) if not finished[c]]
            if not active:
                break

            # ── monta o grafo parcial dos candidatos ativos ────────────────
            xs, eis, eas, batch_vec_parts = [], [], [], []
            offsets = {}
            offset = 0
            for local_i, cand in enumerate(active):
                n_i = len(atom_types[cand])
                if n_i == 0:
                    continue
                offsets[local_i] = offset
                type_t   = torch.tensor(atom_types[cand], dtype=torch.long, device=device)
                charge_t = torch.tensor(charges[cand], dtype=torch.float32, device=device)
                pos_t    = torch.tensor(positions[cand], dtype=torch.float32, device=device)
                val_t    = torch.tensor(valence_used[cand], dtype=torch.float32, device=device)
                bc_t     = torch.tensor(bond_counts[cand], dtype=torch.float32, device=device)
                if bonds[cand]:
                    pairs = [[a, b] for (a, b, _) in bonds[cand]] + \
                            [[b, a] for (a, b, _) in bonds[cand]]
                    ei_local = torch.tensor(pairs, dtype=torch.long, device=device).t()
                    degree_t = torch.bincount(ei_local[0], minlength=n_i).float()
                    # Features de aresta idênticas às do treino:
                    # [one-hot do tipo (4) ; distância/2 Å (1)].
                    bt_list = [bt for (_, _, bt) in bonds[cand]] * 2
                    bt_col = torch.tensor(bt_list, dtype=torch.long, device=device)
                    bt_onehot = torch.zeros((bt_col.shape[0], BOND_TYPE_DIM),
                                            dtype=torch.float32, device=device)
                    bt_onehot[torch.arange(bt_col.shape[0], device=device), bt_col] = 1.0
                    d_edge = (pos_t[ei_local[0]] - pos_t[ei_local[1]]
                              ).norm(dim=-1, keepdim=True) / 2.0
                    eis.append(ei_local + offset)
                    eas.append(torch.cat([bt_onehot, d_edge.clamp(0.0, 2.0)], dim=-1))
                else:
                    degree_t = torch.zeros(n_i, dtype=torch.float32, device=device)
                xs.append(_make_partial_features(
                    type_t, charge_t, degree_t,
                    positions=pos_t, valence_used=val_t, bond_counts=bc_t))
                batch_vec_parts.append(torch.full((n_i,), local_i, dtype=torch.long, device=device))
                offset += n_i

            if xs:
                partial_x = torch.cat(xs, dim=0)
                partial_batch_vec = torch.cat(batch_vec_parts, dim=0)
                partial_edge_index = (torch.cat(eis, dim=1) if eis
                                      else torch.zeros((2, 0), dtype=torch.long, device=device))
                partial_edge_attr = (torch.cat(eas, dim=0) if eas
                                     else z.new_zeros((0, self.decoder.edge_feat_dim)))
            else:
                partial_x = z.new_zeros((0, self.decoder.node_feat_dim))
                partial_batch_vec = torch.zeros(0, dtype=torch.long, device=device)
                partial_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                partial_edge_attr = z.new_zeros((0, self.decoder.edge_feat_dim))

            active_t = torch.tensor(active, dtype=torch.long, device=device)
            z_input_active = z_input_full[active_t]
            step_out = self.decoder.forward_step(
                z_input_active, partial_x, partial_edge_index,
                partial_batch_vec, len(active), partial_edge_attr
            )

            stop_prob = torch.sigmoid(step_out["stop_logit"])
            will_stop = stop_prob > stop_threshold
            if t == max_atoms - 1:
                will_stop = torch.ones_like(will_stop)   # força parar no teto

            type_probs = (step_out["atom_type_logits"] / max(temperature, 1e-3)).softmax(dim=-1)
            next_type    = torch.multinomial(type_probs, 1).squeeze(-1)
            next_charge  = step_out["charge"]
            next_fcharge = step_out["fcharge_logits"].argmax(dim=-1) - 1   # {0,1,2} → {−1,0,+1}

            maxval = _ATOM_TYPE_MAXVAL.to(device)
            border = _BOND_ORDER.to(device)

            # ── Átomo FOCAL, sob máscara de valência ──────────────────────
            # Só átomos com valência sobrando podem receber o próximo. Um
            # candidato cujos átomos estão todos saturados é encerrado em
            # vez de forçar uma ligação impossível — mesma decisão do
            # choose_focal do PocketFlow.
            focal_local, focal_idx_list, focal_pos_list = {}, [], []
            for li in range(len(active)):
                if bool(will_stop[li]):
                    continue
                cand = active[li]
                n_i = len(atom_types[cand])
                if n_i == 0:
                    focal_local[li] = -1      # primeiro átomo: sem âncora
                    continue
                base = offsets[li]
                rem = torch.tensor(
                    [float(maxval[atom_types[cand][k]]) - valence_used[cand][k]
                     for k in range(n_i)], dtype=torch.float32, device=device)
                logits_i = step_out["focal_logits"][base:base + n_i].clone()
                logits_i[rem <= 0.05] = float("-inf")
                if not bool(torch.isfinite(logits_i).any()):
                    will_stop[li] = True
                    continue
                probs_f = (logits_i / max(temperature, 1e-3)).softmax(dim=-1)
                focal_local[li] = int(torch.multinomial(probs_f, 1))

            continue_local = [li for li in range(len(active)) if not bool(will_stop[li])]
            for li in continue_local:
                cand = active[li]
                f = focal_local.get(li, -1)
                if f < 0:
                    focal_idx_list.append(-1)
                    focal_pos_list.append(torch.zeros(3, device=device))
                else:
                    focal_idx_list.append(offsets[li] + f)
                    focal_pos_list.append(torch.tensor(
                        positions[cand][f], dtype=torch.float32, device=device))

            next_pos_map = {}
            bond_exist_probs = bond_type_pred = None
            if continue_local:
                cl_t = torch.tensor(continue_local, dtype=torch.long, device=device)

                # ── Posição AMOSTRADA da mistura, não promediada ──────────
                # É aqui que a arquitetura 2 difere da 1: em vez de um
                # ponto determinístico (que sob incerteza multimodal vira a
                # média — o lugar onde não há átomo nenhum), sorteia-se uma
                # componente da mistura e depois um ponto dentro dela.
                mu, sigma, pi_logits = self.decoder.position_mdn(
                    step_out["_h"], step_out["_ctx"][cl_t],
                    torch.tensor(focal_idx_list, dtype=torch.long, device=device),
                    torch.stack(focal_pos_list),
                )
                pi = (pi_logits / max(temperature, 1e-3)).softmax(dim=-1)
                comp = torch.multinomial(pi, 1).squeeze(-1)
                ar = torch.arange(mu.shape[0], device=device)
                sampled = torch.normal(mu[ar, comp],
                                       sigma[ar, comp] * max(temperature, 1e-3))
                for k, li in enumerate(continue_local):
                    next_pos_map[li] = sampled[k]

                bond_exist_logits, bond_type_logits = self.decoder.predict_bonds(
                    step_out["_h"], step_out["_ctx"], partial_batch_vec, cl_t
                )
                if bond_exist_logits.numel() > 0:
                    bond_exist_probs = torch.sigmoid(bond_exist_logits)
                    bond_type_pred   = bond_type_logits.argmax(dim=-1)

            for li in range(len(active)):
                if bool(will_stop[li]):
                    finished[active[li]] = True

            # distribui as decisões de volta para cada candidato — a ordem
            # dos pares em bond_exist_probs segue exatamente a ordem de
            # continue_local (li crescente) × átomos já existentes (0..n_i-1
            # na ordem de colocação), ver GNNDecoder.predict_bonds.
            cursor = 0
            for li in continue_local:
                cand = active[li]
                n_i = len(atom_types[cand])
                new_type = int(next_type[li])
                new_max  = float(maxval[new_type])
                f_local  = focal_local.get(li, -1)

                probs_i = types_i = None
                if bond_exist_probs is not None and n_i > 0:
                    probs_i = bond_exist_probs[cursor:cursor + n_i]
                    types_i = bond_type_pred[cursor:cursor + n_i]
                    cursor += n_i

                new_used   = [0.0]
                new_counts = [0.0] * BOND_TYPE_DIM

                def _try_bond(k: int, bt: int) -> bool:
                    """Aceita a ligação só se couber na valência dos DOIS
                    lados. Rejeitar aqui, ANTES de a ligação existir, evita
                    a fragmentação que a filtragem pós-hoc causava."""
                    o = float(border[bt])
                    rem_k = float(maxval[atom_types[cand][k]]) - valence_used[cand][k]
                    if o > rem_k + 1e-6 or new_used[0] + o > new_max + 1e-6:
                        return False
                    bonds[cand].append((k, n_i, int(bt)))
                    valence_used[cand][k] += o
                    bond_counts[cand][k][bt] += 1.0
                    new_used[0] += o
                    new_counts[bt] += 1.0
                    return True

                # A ligação com o focal é a âncora da posição: tentada
                # primeiro e, se o tipo previsto não couber na valência
                # restante, rebaixada para simples antes de desistir.
                if f_local >= 0:
                    bt_focal = int(types_i[f_local]) if types_i is not None else 0
                    if not _try_bond(f_local, bt_focal):
                        _try_bond(f_local, 0)

                # Demais ligações, por probabilidade decrescente e só
                # enquanto a valência permitir — o rollback do PocketFlow,
                # aqui de forma determinística (sem reamostragem).
                if probs_i is not None:
                    for k in torch.argsort(probs_i, descending=True).tolist():
                        if k == f_local:
                            continue
                        if float(probs_i[k]) <= bond_threshold:
                            break
                        _try_bond(k, int(types_i[k]))

                atom_types[cand].append(new_type)
                charges[cand].append(float(next_charge[li]))
                fcharges[cand].append(int(next_fcharge[li]))
                positions[cand].append(next_pos_map[li].tolist())
                valence_used[cand].append(new_used[0])
                bond_counts[cand].append(new_counts)
                if len(atom_types[cand]) >= max_atoms:
                    finished[cand] = True

        results = []
        for cand in range(n_samples):
            results.append({
                "atom_types": torch.tensor(atom_types[cand], dtype=torch.long),
                "charges":    torch.tensor(charges[cand], dtype=torch.float32),
                "formal_charges": torch.tensor(fcharges[cand], dtype=torch.long),
                "positions":  (torch.tensor(positions[cand], dtype=torch.float32)
                              if positions[cand] else torch.zeros((0, 3))),
                "bonds":      bonds[cand],
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE LOSS
# ══════════════════════════════════════════════════════════════════════════════

def cvae_loss(
    preds: dict,
    beta: float = 1.0,
    pos_weight: float = 2.0,
    stop_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> dict:
    """Loss total do CVAE (decoder autorregressivo) = reconstruction loss +
    β·KL divergence.

    `preds` já vem com todos os targets embutidos e ACHATADOS por todos os
    passos de construção (ver GNNDecoder.forward_teacher_forcing) — não
    recebe mais o grafo original aqui, ao contrário da versão anterior
    (decoder de topologia fixa).

    Componentes da reconstruction loss:
      L_stop  : BCE na decisão parar/continuar a cada passo (nova)
      L_atom  : cross-entropy no tipo do PRÓXIMO átomo
      L_charge: MSE na carga parcial do próximo átomo
      L_pos   : MSE na posição 3D do próximo átomo (normalizada pelo desvio
                padrão do batch)
      L_exist : BCE na existência de cada ligação candidata (novo átomo ×
                CADA átomo já existente) — com amostragem negativa REAL
                (a maioria dos pares não é ligação). Antes só existiam
                positivos (arestas de um template fixo), o que tornava
                essa cabeça trivial — corrigido aqui por construção.
      L_bond  : cross-entropy no tipo de ligação, SÓ nos pares onde a
                ligação realmente existe (bond_type não tem sentido para
                um par não-ligado).

    KL divergence:
      L_KL = -0.5 · média(1 + logvar - mean² - exp(logvar))
      Média sobre todos os elementos (batch × latent_dim), não soma por
      amostra — por construção, a magnitude de L_KL não escala com
      latent_dim (mudar latent_dim não exige reajustar beta_max).

    O hiperparâmetro beta controla o trade-off reconstrução vs. regularização.
    beta=1 → VAE padrão; beta<1 → mais ênfase em reconstrução.
    """
    device = preds["z_mean"].device
    zero = torch.zeros((), device=device)

    def _pos_weight_for(targets: torch.Tensor) -> "torch.Tensor | None":
        """pos_weight do BCE = nº de negativos / nº de positivos do batch.

        Sem isto, as duas cabeças binárias ficam sistematicamente
        subestimadas: parar acontece 1 vez a cada ~21 passos, e só 8,6% dos
        pares (novo átomo × átomo já colocado) são ligação de verdade. O
        ótimo de um BCE não ponderado sob esse desbalanceamento é prever
        "não" quase sempre — foi o que produziu moléculas grandes demais
        (43-54 átomos contra mediana real de 20) e fragmentadas.
        """
        n_pos = float(targets.sum().item())
        if n_pos <= 0.0:
            return None
        n_neg = float(targets.numel()) - n_pos
        return torch.tensor(max(n_neg / n_pos, 1.0), device=targets.device)

    L_stop = (F.binary_cross_entropy_with_logits(
                  preds["stop_logits"], preds["stop_targets"],
                  pos_weight=_pos_weight_for(preds["stop_targets"]))
              if preds["stop_logits"].shape[0] > 0 else zero)
    L_atom = (F.cross_entropy(preds["atom_logits"], preds["atom_targets"])
              if preds["atom_logits"].shape[0] > 0 else zero)
    L_charge = (F.mse_loss(preds["charges"], preds["charge_targets"])
                if preds["charges"].shape[0] > 0 else zero)

    if preds["pos_targets"].shape[0] > 0:
        # Huber (smooth L1) em Å, com beta = 1 Å: quadrático dentro de 1 Å,
        # linear além disso.
        #
        # A versão anterior normalizava pelo desvio-padrão do BATCH, o que
        # tinha dois defeitos. Primeiro, a escala mudava a cada batch, então
        # o mesmo erro em Å produzia perdas diferentes conforme os alvos
        # sorteados. Segundo, e mais grave, com alvos absolutos esse desvio
        # era de ~28 Å, de modo que errar 1 Å custava (1/28)² ≈ 0,001 —
        # ruído perto do termo de tipo atômico (~0,5). O otimizador não
        # tinha incentivo algum para acertar geometria.
        #
        # Com alvos no referencial local (poucas dezenas de Å) e escala
        # fixa, a perda passa a ser legível em Å: 1 Å de erro custa ~0,5 e
        # 3 Å custam ~2,5, comparáveis ao termo de tipo atômico. O Huber
        # evita que os erros grandes das primeiras épocas dominem o
        # gradiente, como aconteceria com MSE puro.
        # SUBSTITUÍDO na arquitetura 2 por −log-verossimilhança sob a MDN.
        # O Huber acima media a incerteza: quando o próximo átomo pode ir
        # para várias direções igualmente válidas, o mínimo do erro
        # quadrático/Huber é a MÉDIA dessas direções — um ponto onde não há
        # átomo nenhum. Medido: o modelo treinado ficava PIOR que um
        # preditor que sempre chuta o centroide (1,599 contra 1,126), e na
        # geração livre todos os átomos saíam empilhados (raio de giro
        # 0,3 Å contra 3,3 Å real). A mistura representa os modos em vez de
        # promediá-los, e é AMOSTRADA na geração.
        #
        # Calculada em espaço log (logsumexp) e não como produto de
        # densidades: com 3 dimensões e sigma pequeno o produto satura em 0
        # e o log explode. Pocket2Mol multiplica as densidades e loga
        # depois (models/position.py); aqui, a forma estável.
        mu    = preds["pos_mu"]                      # (N, K, 3)
        sigma = preds["pos_sigma"].clamp(min=1e-4)   # (N, K, 3)
        tgt   = preds["pos_targets"].unsqueeze(1)    # (N, 1, 3)
        log_norm = (-0.5 * ((tgt - mu) / sigma) ** 2
                    - sigma.log() - 0.9189385332046727)   # 0.5·log(2π)
        log_comp = log_norm.sum(dim=-1) + F.log_softmax(preds["pos_pi"], dim=-1)
        L_pos = -torch.logsumexp(log_comp, dim=-1).mean()

        # Diagnóstico legível em Å (não entra no gradiente): distância
        # entre o alvo e a média da componente de maior peso. O log da
        # arquitetura 1 imprimia o valor do Huber rotulado como "Å", o que
        # era enganoso.
        with torch.no_grad():
            best = preds["pos_pi"].argmax(dim=-1)
            ar = torch.arange(mu.shape[0], device=mu.device)
            pos_mae = (mu[ar, best] - preds["pos_targets"]).norm(dim=-1).mean()
    else:
        L_pos = zero
        pos_mae = zero

    L_exist = (F.binary_cross_entropy_with_logits(
                   preds["bond_exist"], preds["bond_exist_targets"],
                   pos_weight=_pos_weight_for(preds["bond_exist_targets"]))
               if preds["bond_exist"].shape[0] > 0 else zero)

    # ── Foco e carga formal ───────────────────────────────────────────────
    L_focal = (preds["focal_nll"].mean()
               if preds["focal_nll"].shape[0] > 0 else zero)
    # Carga formal: 99,2% dos átomos do conjunto de treino têm carga 0
    # (0,56% em −1 e 0,22% em +1). Sem ponderação, a entropia cruzada
    # colapsa para "prever 0 sempre" e a cabeça não entrega nada.
    #
    # A ponderação é pela RAIZ da frequência inversa, não pela frequência
    # inversa cheia. Balanceamento total daria ~124× de peso às classes
    # raras, e aqui os dois erros NÃO custam o mesmo: deixar de prever uma
    # carga é inócuo (é o comportamento anterior, em que nenhuma carga era
    # atribuída), enquanto inventar uma carga é nocivo — altera a valência
    # permitida e pode quebrar a molécula na sanitização. A raiz dá ~11× de
    # razão: o bastante para sair do colapso sem virar gatilho fácil.
    #
    # Os pesos são renormalizados para média 1 de modo que a escala de
    # L_fcharge continue comparável à das outras cabeças — o coeficiente
    # 0,5 aplicado adiante segue significando a mesma coisa.
    if preds["fcharge_logits"].shape[0] > 0:
        tgt_fc = preds["fcharge_targets"]
        n_cls  = preds["fcharge_logits"].shape[1]
        counts = torch.bincount(tgt_fc, minlength=n_cls).float().clamp(min=1.0)
        w_fc   = (tgt_fc.numel() / (n_cls * counts)).sqrt().clamp(0.2, 10.0)
        w_fc   = w_fc / w_fc.mean()
        L_fcharge = F.cross_entropy(preds["fcharge_logits"], tgt_fc, weight=w_fc)
    else:
        L_fcharge = zero

    # Tipo de ligação: só nos pares REALMENTE ligados (bond_exist_target==1)
    # — classificar o "tipo" de uma ligação que não existe não tem sentido.
    if preds["bond_logits"].shape[0] > 0:
        pos_mask = preds["bond_exist_targets"] > 0.5
        L_bond = (F.cross_entropy(preds["bond_logits"][pos_mask], preds["bond_targets"][pos_mask])
                  if pos_mask.any() else zero)
    else:
        L_bond = zero

    # KL divergence
    z_mean   = preds["z_mean"]
    z_logvar = preds["z_logvar"]
    L_KL = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    # Total
    L_recon = (stop_weight * L_stop + L_atom + L_bond + L_exist +
               pos_weight * L_pos + 0.5 * L_charge +
               focal_weight * L_focal + 0.5 * L_fcharge)
    L_total = L_recon + beta * L_KL

    return {
        "total":   L_total,
        "recon":   L_recon,
        "stop":    L_stop,
        "atom":    L_atom,
        "bond":    L_bond,
        "exist":   L_exist,
        "pos":     L_pos,
        "pos_mae": pos_mae,
        "focal":   L_focal,
        "fcharge": L_fcharge,
        "charge":  L_charge,
        "kl":      L_KL,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOOP DE TREINO
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     "LigandCVAE",
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    beta:      float,
    grad_clip: float = 0.5,   # reduzido de 1.0 para prevenir explosão
    pos_weight:  float = 2.0,
    stop_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> dict:
    model.train()
    totals = {k: 0.0 for k in ["total","recon","stop","atom","bond","exist","pos",
                               "pos_mae","focal","fcharge","charge","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if batch is None:
            continue
        site   = batch["tensor"].to(device)
        graphs = [g.to(device) for g in batch["graphs"]]
        voxel_sizes = batch["voxel_sizes"].to(device)

        optimizer.zero_grad()
        preds  = model(site, graphs, voxel_sizes)
        losses = cvae_loss(preds, beta=beta,
                           pos_weight=pos_weight, stop_weight=stop_weight,
                           focal_weight=focal_weight)

        # Pula batch se a loss for NaN/Inf (tensor corrompido escapou do filtro)
        if not torch.isfinite(losses["total"]):
            log.warning("  Batch ignorado: loss NaN/Inf detectada")
            optimizer.zero_grad()
            continue

        losses["total"].backward()

        # Detecta explosão de gradiente — NaN nos gradientes corrompe os pesos
        grad_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        if not grad_ok:
            log.warning("  Gradiente NaN detectado — step ignorado, "
                        "zerando gradientes.")
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model:  "LigandCVAE",
    loader: DataLoader,
    device: torch.device,
    beta:   float,
    pos_weight:  float = 2.0,
    stop_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["total","recon","stop","atom","bond","exist","pos",
                               "pos_mae","focal","fcharge","charge","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        if batch is None:
            continue
        site        = batch["tensor"].to(device)
        graphs      = [g.to(device) for g in batch["graphs"]]
        voxel_sizes = batch["voxel_sizes"].to(device)
        preds       = model(site, graphs, voxel_sizes)
        losses = cvae_loss(preds, beta=beta,
                           pos_weight=pos_weight, stop_weight=stop_weight,
                           focal_weight=focal_weight)
        # Pula batch com loss NaN (modelo corrompido ou dado degenerado)
        if not torch.isfinite(losses["total"]):
            continue
        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train(
    # Defaults revisados: eram calibrados para <1000 pares (ver comentário
    # em SiteEncoder — "Arquitetura reduzida para datasets pequenos").
    # Dataset atual tem >33.000 pares e a GPU tem folga de sobra para o
    # modelo (~1,8-2,4M parâmetros) — latent/hidden/gnn_layers/dropout
    # voltam aos defaults de fábrica das classes.
    latent_dim:    int   = 256,
    hidden_dim:    int   = 256,
    n_gnn_layers:  int   = 4,
    batch_size:    int   = 32,
    epochs:        int   = 150,   # máximo — early stopping interrompe antes
    lr:            float = 4e-4,  # ajustado para batch_size=32 (era 3e-4/batch=8)
    beta_max:      float = 0.5,
    beta_warmup:   int   = 30,
    dropout:       float = 0.20,
    checkpoint_dir: str = "checkpoints",
    tensor_dir: str | None = None,
    resume: "str | None" = None,
    freeze_encoder: bool = False,
    patience:      int   = 15,    # epochs sem melhora antes de parar
    min_delta:     float = 1e-4,  # melhora mínima considerada significativa
    device:     str   = "auto",    # "auto" | "cuda" | "cpu" | "cuda:0" | "cuda:1"
    # Pesos relativos das cabeças da loss (ver cvae_loss). Expostos aqui para
    # que a Etapa 6 (otimização de hiperparâmetros) possa buscá-los — antes
    # ficavam presos nos defaults de cvae_loss, inalcançáveis pela interface.
    pos_weight:    float = 2.0,
    stop_weight:   float = 1.0,
    focal_weight:  float = 1.0,
    # Regularização/estabilidade: também estavam presos no corpo de train()
    # (AdamW) e no default de train_epoch. Expostos pelo mesmo motivo.
    weight_decay:  float = 1e-4,
    grad_clip:     float = 0.5,
    # Callback opcional chamado ao fim de cada epoch com
    # (epoch, train_losses, val_losses). Usado pelo pipeline_optimize para
    # reportar métricas intermediárias ao Optuna e podar trials ruins — uma
    # exceção levantada aqui dentro (ex.: optuna.TrialPruned) propaga e
    # interrompe o treino, que é exatamente o comportamento desejado.
    epoch_callback=None,
):
    """Função principal de treino com suporte a fine-tuning em fases.

    Treinamento do zero:
        python 05_train.py --epochs 100 --checkpoint-dir checkpoints_phase1/

    Fine-tuning a partir de um checkpoint:
        python 05_train.py --epochs 50 --lr 1e-5
            --resume checkpoints_phase1/best_model.pt
            --checkpoint-dir checkpoints_phase2_target_family/

    Fine-tuning com encoder congelado:
        python 05_train.py --epochs 30 --lr 5e-6
            --resume checkpoints_phase2_target_family/best_model.pt
            --freeze-encoder
            --checkpoint-dir checkpoints_phase3_target_specific/

    Comportamento do --resume:
      - Carrega os pesos do modelo (model_state_dict)
      - NÃO carrega o estado do otimizador (recomeça com o lr fornecido)
      - NÃO carrega o histórico de epochs (começa do epoch 1)
      - Lê os hiperparâmetros de arquitetura do checkpoint para garantir
        compatibilidade, sobrescrevendo latent_dim/hidden_dim/n_gnn_layers
        se necessário
      - O lr deve ser reduzido manualmente nas fases de especialização

    Comportamento do --freeze-encoder:
      - Congela todos os parâmetros do SiteEncoder
      - Apenas o GNNDecoder é treinado
      - Útil na especialização final para refinar o decoder sem perder o encoder geral
    """
    log_path = Path(checkpoint_dir) / "training.log"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(stream=None),
        ]
    )
    if device == "auto":
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _dev = torch.device(device)
    device = _dev
    log.info(f"Dispositivo: {device}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ── Datasets e loaders ──────────────────────────────────────────────────
    dataset_dir = Path(tensor_dir) if tensor_dir else TENSOR_DIR
    train_ds = SiteLigandDataset(tensor_dir=dataset_dir, split="train")
    val_ds   = SiteLigandDataset(tensor_dir=dataset_dir, split="val")

    # Retrato da config de voxelização que gerou ESSES tensores (ver
    # pipeline_voxelize.save_voxel_config_snapshot) — embutido no checkpoint
    # salvo abaixo (best_model.pt["config"]["voxel_config"]) para que a
    # geração (Etapa 9) possa reconstruir o tensor de um sítio novo com a
    # MESMA config usada aqui, sem depender do usuário reproduzi-la
    # manualmente na interface (risco de mismatch treino↔inferência nos
    # canais 3-4, ver pipeline_generate.load_model). Ausente (dict vazio)
    # para tensores gerados antes desta funcionalidade existir — nesse caso
    # a geração cai para o comportamento legado (config manual da UI).
    voxel_config_snapshot = {}
    _vc_path = dataset_dir / "voxel_config.json"
    if _vc_path.exists():
        try:
            voxel_config_snapshot = json.loads(_vc_path.read_text())
            log.info(f"  Voxel config herdada de {_vc_path}: {voxel_config_snapshot}")
        except Exception as e:
            log.warning(f"  Não foi possível ler {_vc_path}: {e}")
    else:
        log.warning(
            f"  {_vc_path} não encontrado — o checkpoint salvo NÃO terá a config de "
            "voxelização embutida (tensores gerados antes desta funcionalidade existir, "
            "ou tensor_dir não é o mesmo usado na Etapa 5). A geração cairá para a "
            "config manual da interface (Etapa 9)."
        )

    # pipeline_train.py, linhas 840–845
    # num_workers=0: carregamento de dados no mesmo processo, sem fork.
    # O gargalo do treinamento é o compute GPU (forward/backward), não o I/O
    # de disco — num_workers=0 tem impacto negligível no throughput real.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=SiteLigandDataset.collate_fn,
        num_workers=0,                            # evita fork aninhado pós-CUDA
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=SiteLigandDataset.collate_fn,
        num_workers=0,                            # idem
    )
    # Alternativa se o I/O for identificado como gargalo em benchmarks futuros: 
    # usar multiprocessing_context="spawn" no DataLoader em vez de num_workers=0. 
    # Isso mantém o paralelismo de I/O sem fork, mas adiciona overhead de inicialização por epoch. 
    # Para o tamanho típico desse dataset (tensores .npz em SSD), num_workers=0 é suficiente.
    # ── Modelo ────────────────────────────────────────────────────────────────
    if resume:
        # Carrega hiperparâmetros de arquitetura do checkpoint para garantir
        # que o modelo instanciado tem a mesma forma dos pesos salvos
        ckpt_resume = torch.load(resume, map_location=device, weights_only=False)
        ckpt_cfg    = ckpt_resume.get("config", {})

        # Mesma guarda de pipeline_generate.load_model, que faltava aqui.
        # O load_state_dict abaixo usa strict=False: pesos de uma
        # arquitetura anterior entrariam PARCIALMENTE (encoder e camadas
        # comuns do decoder), deixando as cabeças novas — foco, MDN de
        # posição e carga formal — com inicialização aleatória, e o treino
        # seguiria como se nada houvesse, com um mero aviso de "camadas
        # ausentes". O campo "Retomar de" da interface é preenchido a
        # partir do estado salvo do job, então esse cenário é fácil de
        # cair sem perceber.
        _ck_arch = int(ckpt_cfg.get("arch_version", 1))
        if _ck_arch != ARCH_VERSION:
            raise RuntimeError(
                f"Checkpoint de retomada incompatível: {resume}\n"
                f"  arquitetura do checkpoint : versão {_ck_arch}\n"
                f"  arquitetura do código     : versão {ARCH_VERSION}\n\n"
                "Fine-tuning a partir de outra arquitetura carregaria só "
                "parte dos pesos e deixaria as cabeças de foco, posição "
                "(MDN) e carga formal aleatórias.\n"
                "Deixe o campo 'Retomar de' VAZIO para treinar do zero, ou "
                "aponte para um checkpoint da versão "
                f"{ARCH_VERSION}."
            )

        if ckpt_cfg:
            latent_dim   = ckpt_cfg.get("latent_dim",   latent_dim)
            hidden_dim   = ckpt_cfg.get("hidden_dim",   hidden_dim)
            n_gnn_layers = ckpt_cfg.get("n_gnn_layers", n_gnn_layers)
            log.info(f"Arquitetura do checkpoint: latent={latent_dim} "
                     f"hidden={hidden_dim} gnn_layers={n_gnn_layers}")

    model = LigandCVAE(
        in_channels=5, latent_dim=latent_dim, hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers, dropout=dropout,
    ).to(device)

    # ── Carrega pesos para fine-tuning (--resume) ─────────────────────────────
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {resume_path}")

        missing, unexpected = model.load_state_dict(
            ckpt_resume["model_state"], strict=False
        )
        src_epoch    = ckpt_resume.get("epoch", "?")
        src_val_loss = ckpt_resume.get("val_loss", "?")
        log.info(f"Pesos carregados de: {resume_path}")
        log.info(f"  Origem: epoch {src_epoch}, val_loss={src_val_loss}")
        if missing:
            log.warning(f"  Camadas ausentes no checkpoint: {missing}")
        if unexpected:
            log.warning(f"  Camadas inesperadas no checkpoint: {unexpected}")
        mode = "FINE-TUNING"
    else:
        mode = "TREINAMENTO DO ZERO"

    # ── Congela encoder (--freeze-encoder) ───────────────────────────────────
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        log.info("Encoder CONGELADO — apenas o decoder será treinado.")

    # Inicializa pesos (apenas no treino do zero — não no fine-tuning)
    if not resume:
        model._init_weights()
        log.info("Pesos inicializados com Xavier/He uniform.")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Modo: {mode}")
    log.info(f"Parâmetros: {n_trainable:,} treináveis / {n_total:,} total")

    # ── Otimizador e scheduler ────────────────────────────────────────────────
    # Otimiza apenas parâmetros não congelados
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    best_val       = float("inf")
    history        = []
    patience_count = 0

    # ── Loop principal ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # KL annealing
        beta = min(beta_max, beta_max * epoch / beta_warmup)

        log.info(f"\nEpoch {epoch}/{epochs}  β={beta:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_losses = train_epoch(model, train_loader, optimizer, device, beta,
                                   grad_clip=grad_clip,
                                   pos_weight=pos_weight, stop_weight=stop_weight,
                                   focal_weight=focal_weight)
        val_losses   = eval_epoch (model, val_loader,              device, beta,
                                   pos_weight=pos_weight, stop_weight=stop_weight,
                                   focal_weight=focal_weight)
        scheduler.step()

        # Termos separados no log. Só `recon` e `kl` agregados escondiam o
        # que estava acontecendo no treino anterior: o termo de posição
        # ficou travado em ~1,0 (variância explicada zero) por 111 épocas
        # sem que isso aparecesse em lugar nenhum. `pos` está em Å.
        def _linha(rot, L):
            # `pos` agora é −log-verossimilhança da MDN (adimensional, pode
            # ser negativa quando o ajuste é bom); `pos_mae` é o erro
            # geométrico de verdade, em Å. Na arquitetura 1 o valor do
            # Huber era impresso com sufixo "Å", o que induzia a erro.
            return (f"  {rot} → total={L['total']:.4f}  recon={L['recon']:.4f}  "
                    f"kl={L['kl']:.4f}  |  atom={L['atom']:.4f}  "
                    f"posNLL={L['pos']:.4f}  posMAE={L['pos_mae']:.2f}Å  "
                    f"focal={L['focal']:.4f}  exist={L['exist']:.4f}  "
                    f"bond={L['bond']:.4f}  stop={L['stop']:.4f}  "
                    f"charge={L['charge']:.4f}  fcharge={L['fcharge']:.4f}")
        log.info(_linha("Train", train_losses))
        log.info(_linha("Val  ", val_losses))

        row = {"epoch": epoch, "beta": beta}
        row.update({f"train_{k}": v for k, v in train_losses.items()})
        row.update({f"val_{k}":   v for k, v in val_losses.items()})
        history.append(row)

        # Callback de epoch — chamado ANTES de gravar checkpoints: se ele
        # abortar o treino (poda do Optuna), não faz sentido ter escrito
        # ~30 MB de checkpoint para um trial que será descartado.
        if epoch_callback is not None:
            epoch_callback(epoch, train_losses, val_losses)

        # Salva melhor modelo
        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "optim_state":   optimizer.state_dict(),
                "val_loss":      best_val,
                "resumed_from":  str(resume) if resume else None,
                "freeze_encoder": freeze_encoder,
                # Histórico completo junto do checkpoint: sem ele, diagnosticar
                # um treino ruim depois exige remedir tudo do zero.
                "history":       history,
                "config": {
                    "arch_version":     ARCH_VERSION,
                    "n_pos_components": N_POS_COMPONENTS,
                    "latent_dim":   latent_dim,
                    "hidden_dim":   hidden_dim,
                    "n_gnn_layers": n_gnn_layers,
                    "voxel_config": voxel_config_snapshot,
                    # Referencial em que a cabeça de posições foi treinada.
                    # A geração consulta esta chave para saber se deve somar
                    # a origem da grade de volta. Checkpoints anteriores a
                    # esta correção não a têm, e são tratados como legado
                    # (coordenadas absolutas) com aviso — ver
                    # pipeline_generate.generate_candidates.
                    "pos_frame":    POS_FRAME,
                },
            }, ckpt_dir / "best_model.pt")
            log.info(f"  ★ Melhor modelo salvo (val_loss={best_val:.4f})")
            patience_count = 0
        elif epoch > beta_warmup:
            # Só conta paciência DEPOIS que o beta terminou de subir (ver
            # "KL annealing" acima). Durante o warmup, beta·L_KL cresce a
            # cada epoch mesmo com o modelo melhorando de verdade —
            # contar isso como "sem melhora" faria o early stopping
            # disparar por causa do annealing em si, não por estagnação
            # real (ex.: patience=15 < beta_warmup=30 pararia o treino no
            # meio do warmup, antes de beta sequer atingir beta_max).
            patience_count += 1
            log.info(f"  Sem melhora: {patience_count}/{patience} epochs")
        else:
            log.info(f"  Sem melhora, mas dentro do warmup do beta "
                      f"(epoch {epoch}/{beta_warmup}) — não conta para paciência.")

        # Checkpoint periódico
        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pt")

        # Early stopping
        if patience_count >= patience:
            log.info(f"Early stopping: {patience} epochs sem melhora.")
            log.info(f"Melhor val_loss: {best_val:.4f}")
            break

    # Salva histórico
    pd.DataFrame(history).to_csv(ckpt_dir / "training_history.csv", index=False)
    log.info(f"\nTreino concluído. Melhor val_loss: {best_val:.4f}")
    # Retorna também o melhor val_loss: sem isso, quem chama train() por
    # dentro (pipeline_optimize) teria de reabrir best_model.pt só para ler
    # um número. Chamadas antigas que usam só o modelo continuam válidas via
    # desempacotamento (`model, _ = train(...)`); a interface descarta o
    # retorno inteiro, então nada quebra ali.
    return model, best_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Treina o CVAE de geração de ligantes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Fase 1 — treinamento do zero com dataset completo
  python 05_train.py --epochs 100 --checkpoint-dir checkpoints_phase1/

  # Fase 2 — fine-tuning na família/classe de alvo do job
  python 05_train.py --epochs 50 --lr 1e-5 \
      --resume checkpoints_phase1/best_model.pt \
      --checkpoint-dir checkpoints_phase2_target_family/

  # Fase 3 — fine-tuning final em alvos específicos do job
  python 05_train.py --epochs 30 --lr 5e-6 \
      --resume checkpoints_phase2_target_family/best_model.pt \
      --freeze-encoder \
      --checkpoint-dir checkpoints_phase3_target_specific/
        """
    )
    # Defaults revisados: calibrados originalmente para <1000 pares (ver
    # comentário em SiteEncoder). Dataset atual tem >33.000 pares e a GPU
    # tem folga de sobra para o modelo — latent/hidden/gnn_layers/dropout
    # voltam aos defaults de fábrica das classes (SiteEncoder/GNNDecoder/
    # LigandCVAE); epochs unificado com o default de train() (era 40 aqui
    # vs. 200 em train() vs. 100 na interface — três valores diferentes).
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--batch",          type=int,   default=32)
    parser.add_argument("--latent",         type=int,   default=256)
    parser.add_argument("--hidden",         type=int,   default=256)
    parser.add_argument("--lr",             type=float, default=4e-4)
    parser.add_argument("--gnn-layers",     type=int,   default=4)
    parser.add_argument("--dropout",        type=float, default=0.20)
    parser.add_argument("--beta",           type=float, default=0.5)
    parser.add_argument("--beta-warmup",    type=int,   default=30)
    parser.add_argument("--pos-weight",     type=float, default=2.0,
                        help="Peso do termo de posição na recon loss (default: 2.0)")
    parser.add_argument("--stop-weight",    type=float, default=1.0,
                        help="Peso do termo de parada na recon loss (default: 1.0)")
    parser.add_argument("--focal-weight",   type=float, default=1.0,
                        help="Peso do termo de átomo focal na recon loss "
                             "(default: 1.0)")
    parser.add_argument("--weight-decay",   type=float, default=1e-4,
                        help="Weight decay do AdamW (default: 1e-4)")
    parser.add_argument("--grad-clip",      type=float, default=0.5,
                        help="Norma máxima do gradiente (default: 0.5)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "phase1"),
    )
    parser.add_argument("--resume",         type=str,   default=None,
                        metavar="CKPT",
                        help="Caminho para best_model.pt de fase anterior "
                             "(ativa fine-tuning)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Congela o encoder durante o fine-tuning "
                             "(útil na Fase 3)")
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Epochs sem melhora antes de parar (default: 15)")
    parser.add_argument("--min-delta",  type=float, default=1e-4,
                        help="Melhora mínima considerada significativa (default: 1e-4)")
    parser.add_argument(
        "--tensor-dir",
        type=str,
        default=str(TENSOR_DIR),
        help="Diretório com tensor_index.csv e os subdiretórios de tensores",
    )    
    args = parser.parse_args()

    train(
        latent_dim=args.latent,
        hidden_dim=args.hidden,
        n_gnn_layers=args.gnn_layers,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        beta_max=args.beta,
        beta_warmup=args.beta_warmup,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
        stop_weight=args.stop_weight,
        focal_weight=args.focal_weight,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        tensor_dir=args.tensor_dir,
        resume=args.resume,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
        min_delta=args.min_delta,
    )
