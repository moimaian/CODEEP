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


def _make_partial_features(
    atom_type_idx: torch.Tensor,   # (n,) long
    charge:        torch.Tensor,   # (n,) float
    degree:        torch.Tensor,   # (n,) float/long — grau NO GRAFO PARCIAL
    node_feat_dim: int = 37,
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
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4,
                    dropout=dropout, concat=True)
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
        self.head_pos = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.GELU(),
            nn.Linear(128, 3)
        )

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
                         partial_batch_vec, n_active):
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
            for gnn, ln in zip(self.gnn_layers, self.layer_norms):
                h = ln(h + gnn(h, partial_edge_index))
            pooled = global_mean_pool(h, partial_batch_vec, size=n_active)
        ctx = torch.cat([pooled, z_proj], dim=-1)
        return h, ctx

    def forward_step(self, z_input, partial_x, partial_edge_index,
                      partial_batch_vec, n_active):
        """Prediz parar/tipo/carga/posição do PRÓXIMO átomo para as
        n_active moléculas ativas neste passo. Retorna também (h, ctx) —
        necessários para predict_bonds."""
        h, ctx = self._encode_partial(
            z_input, partial_x, partial_edge_index, partial_batch_vec, n_active
        )
        return {
            "stop_logit":       self.head_stop(ctx).squeeze(-1),
            "atom_type_logits": self.head_atom_type(ctx),
            "charge":           self.head_charge(ctx).squeeze(-1),
            "pos":              self.head_pos(ctx),
            "_h": h, "_ctx": ctx,
        }

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
            "charges", "charge_targets", "positions", "pos_targets",
            "bond_exist", "bond_exist_targets", "bond_logits", "bond_targets",
        )}

        for t in range(0, max_t + 1):
            active_stop = [i for i in range(B) if n_atoms[i] >= t]
            if not active_stop:
                break
            n_active = len(active_stop)

            xs, eis, eas, batch_vec_parts = [], [], [], []
            offset = 0
            for local_i, orig_i in enumerate(active_stop):
                g = graphs[orig_i]
                if t == 0:
                    continue
                order_t = g.atom_order[:t]
                pos_map = {int(idx): k for k, idx in enumerate(order_t.tolist())}
                ei, ea = g.edge_index, g.edge_attr
                keep_cols = [k for k in range(ei.shape[1])
                             if int(ei[0, k]) in pos_map and int(ei[1, k]) in pos_map]

                # Grau NO GRAFO PARCIAL (não o grau da molécula final) — ver
                # _make_partial_features. Calculado nos índices LOCAIS
                # (0..t-1) antes de aplicar o offset do batch.
                degree_t = torch.zeros(t, dtype=torch.float32)
                if keep_cols:
                    idx_t = torch.tensor(keep_cols, dtype=torch.long)
                    sub_ei_local = ei[:, idx_t].clone()
                    for col in range(sub_ei_local.shape[1]):
                        sub_ei_local[0, col] = pos_map[int(sub_ei_local[0, col])]
                        sub_ei_local[1, col] = pos_map[int(sub_ei_local[1, col])]
                    degree_t = torch.bincount(sub_ei_local[0], minlength=t).float()
                    sub_ei = sub_ei_local + offset
                    eis.append(sub_ei)
                    eas.append(ea[idx_t])

                atom_type_t = g.x[order_t, :ATOM_TYPE_DIM].argmax(dim=-1)
                charge_t    = g.x[order_t, 19]
                xs.append(_make_partial_features(atom_type_t, charge_t, degree_t))
                batch_vec_parts.append(torch.full((t,), local_i, dtype=torch.long))
                offset += t

            if xs:
                partial_x = torch.cat(xs, dim=0).to(device)
                partial_batch_vec = torch.cat(batch_vec_parts, dim=0).to(device)
                partial_edge_index = (torch.cat(eis, dim=1).to(device)
                                      if eis else torch.zeros((2, 0), dtype=torch.long, device=device))
            else:
                partial_x = z.new_zeros((0, self.node_feat_dim))
                partial_batch_vec = torch.zeros(0, dtype=torch.long, device=device)
                partial_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

            active_stop_t = torch.tensor(active_stop, dtype=torch.long, device=device)
            z_input_active = z_input_full[active_stop_t]
            step_out = self.forward_step(
                z_input_active, partial_x, partial_edge_index, partial_batch_vec, n_active
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
            acc["positions"].append(step_out["pos"][active_next_local])
            acc["pos_targets"].append(next_pos_t)

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
            "positions":          _cat("positions", (3,)),
            "pos_targets":        _cat("pos_targets", (3,)),
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
        positions:  list = [[] for _ in range(n_samples)]
        bonds:      list = [[] for _ in range(n_samples)]   # (i, j, bond_type) locais
        finished = [False] * n_samples

        for t in range(max_atoms):
            active = [c for c in range(n_samples) if not finished[c]]
            if not active:
                break

            # ── monta o grafo parcial dos candidatos ativos ────────────────
            xs, eis, eas, batch_vec_parts = [], [], [], []
            offset = 0
            for local_i, cand in enumerate(active):
                n_i = len(atom_types[cand])
                if n_i == 0:
                    continue
                type_t   = torch.tensor(atom_types[cand], dtype=torch.long, device=device)
                charge_t = torch.tensor(charges[cand], dtype=torch.float32, device=device)
                if bonds[cand]:
                    pairs = [[a, b] for (a, b, _) in bonds[cand]] + \
                            [[b, a] for (a, b, _) in bonds[cand]]
                    ei_local = torch.tensor(pairs, dtype=torch.long, device=device).t()
                    degree_t = torch.bincount(ei_local[0], minlength=n_i).float()
                    ea_local = []
                    for (_, _, bt) in bonds[cand]:
                        v = [0.0] * 8; v[bt] = 1.0
                        ea_local += [v, v]
                    eis.append(ei_local + offset)
                    eas.append(torch.tensor(ea_local, dtype=torch.float32, device=device))
                else:
                    degree_t = torch.zeros(n_i, dtype=torch.float32, device=device)
                xs.append(_make_partial_features(type_t, charge_t, degree_t))
                batch_vec_parts.append(torch.full((n_i,), local_i, dtype=torch.long, device=device))
                offset += n_i

            if xs:
                partial_x = torch.cat(xs, dim=0)
                partial_batch_vec = torch.cat(batch_vec_parts, dim=0)
                partial_edge_index = (torch.cat(eis, dim=1) if eis
                                      else torch.zeros((2, 0), dtype=torch.long, device=device))
            else:
                partial_x = z.new_zeros((0, self.decoder.node_feat_dim))
                partial_batch_vec = torch.zeros(0, dtype=torch.long, device=device)
                partial_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

            active_t = torch.tensor(active, dtype=torch.long, device=device)
            z_input_active = z_input_full[active_t]
            step_out = self.decoder.forward_step(
                z_input_active, partial_x, partial_edge_index, partial_batch_vec, len(active)
            )

            stop_prob = torch.sigmoid(step_out["stop_logit"])
            will_stop = stop_prob > stop_threshold
            if t == max_atoms - 1:
                will_stop = torch.ones_like(will_stop)   # força parar no teto

            type_probs = (step_out["atom_type_logits"] / max(temperature, 1e-3)).softmax(dim=-1)
            next_type  = torch.multinomial(type_probs, 1).squeeze(-1)
            next_pos   = step_out["pos"]
            next_charge = step_out["charge"]

            continue_local = [li for li in range(len(active)) if not bool(will_stop[li])]
            bond_exist_probs = bond_type_pred = None
            if continue_local:
                cl_t = torch.tensor(continue_local, dtype=torch.long, device=device)
                bond_exist_logits, bond_type_logits = self.decoder.predict_bonds(
                    step_out["_h"], step_out["_ctx"], partial_batch_vec, cl_t
                )
                if bond_exist_logits.numel() > 0:
                    bond_exist_probs = torch.sigmoid(bond_exist_logits)
                    bond_type_pred   = bond_type_logits.argmax(dim=-1)

            # distribui as decisões de volta para cada candidato — a ordem
            # dos pares em bond_exist_probs segue exatamente a ordem de
            # continue_local (li crescente) × átomos já existentes (0..n_i-1
            # na ordem de colocação), ver GNNDecoder.predict_bonds.
            cursor = 0
            for li in range(len(active)):
                cand = active[li]
                if bool(will_stop[li]):
                    finished[cand] = True
                    continue
                n_i = len(atom_types[cand])
                if bond_exist_probs is not None and n_i > 0:
                    probs_i = bond_exist_probs[cursor:cursor + n_i]
                    types_i = bond_type_pred[cursor:cursor + n_i]
                    for k in range(n_i):
                        if float(probs_i[k]) > bond_threshold:
                            bonds[cand].append((k, n_i, int(types_i[k])))
                    cursor += n_i
                atom_types[cand].append(int(next_type[li]))
                charges[cand].append(float(next_charge[li]))
                positions[cand].append(next_pos[li].tolist())
                if len(atom_types[cand]) >= max_atoms:
                    finished[cand] = True

        results = []
        for cand in range(n_samples):
            results.append({
                "atom_types": torch.tensor(atom_types[cand], dtype=torch.long),
                "charges":    torch.tensor(charges[cand], dtype=torch.float32),
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

    L_stop = (F.binary_cross_entropy_with_logits(preds["stop_logits"], preds["stop_targets"])
              if preds["stop_logits"].shape[0] > 0 else zero)
    L_atom = (F.cross_entropy(preds["atom_logits"], preds["atom_targets"])
              if preds["atom_logits"].shape[0] > 0 else zero)
    L_charge = (F.mse_loss(preds["charges"], preds["charge_targets"])
                if preds["charges"].shape[0] > 0 else zero)

    if preds["positions"].shape[0] > 0:
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
        L_pos = F.smooth_l1_loss(preds["positions"], preds["pos_targets"], beta=1.0)
    else:
        L_pos = zero

    L_exist = (F.binary_cross_entropy_with_logits(preds["bond_exist"], preds["bond_exist_targets"])
               if preds["bond_exist"].shape[0] > 0 else zero)

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
               pos_weight * L_pos + 0.5 * L_charge)
    L_total = L_recon + beta * L_KL

    return {
        "total":  L_total,
        "recon":  L_recon,
        "stop":   L_stop,
        "atom":   L_atom,
        "bond":   L_bond,
        "exist":  L_exist,
        "pos":    L_pos,
        "charge": L_charge,
        "kl":     L_KL,
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
) -> dict:
    model.train()
    totals = {k: 0.0 for k in ["total","recon","stop","atom","bond","exist","pos","charge","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if batch is None:
            continue
        site   = batch["tensor"].to(device)
        graphs = [g.to(device) for g in batch["graphs"]]
        voxel_sizes = batch["voxel_sizes"].to(device)

        optimizer.zero_grad()
        preds  = model(site, graphs, voxel_sizes)
        losses = cvae_loss(preds, beta=beta)

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
) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["total","recon","stop","atom","bond","exist","pos","charge","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        if batch is None:
            continue
        site        = batch["tensor"].to(device)
        graphs      = [g.to(device) for g in batch["graphs"]]
        voxel_sizes = batch["voxel_sizes"].to(device)
        preds       = model(site, graphs, voxel_sizes)
        losses = cvae_loss(preds, beta=beta)
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
    # geração (Etapa 8) possa reconstruir o tensor de um sítio novo com a
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
            "config manual da interface (Etapa 8)."
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
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
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

        train_losses = train_epoch(model, train_loader, optimizer, device, beta)
        val_losses   = eval_epoch (model, val_loader,              device, beta)
        scheduler.step()

        # Termos separados no log. Só `recon` e `kl` agregados escondiam o
        # que estava acontecendo no treino anterior: o termo de posição
        # ficou travado em ~1,0 (variância explicada zero) por 111 épocas
        # sem que isso aparecesse em lugar nenhum. `pos` está em Å.
        def _linha(rot, L):
            return (f"  {rot} → total={L['total']:.4f}  recon={L['recon']:.4f}  "
                    f"kl={L['kl']:.4f}  |  atom={L['atom']:.4f}  pos={L['pos']:.4f}Å  "
                    f"exist={L['exist']:.4f}  bond={L['bond']:.4f}  "
                    f"stop={L['stop']:.4f}  charge={L['charge']:.4f}")
        log.info(_linha("Train", train_losses))
        log.info(_linha("Val  ", val_losses))

        row = {"epoch": epoch, "beta": beta}
        row.update({f"train_{k}": v for k, v in train_losses.items()})
        row.update({f"val_{k}":   v for k, v in val_losses.items()})
        history.append(row)

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
    return model


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
        checkpoint_dir=args.checkpoint_dir,
        tensor_dir=args.tensor_dir,
        resume=args.resume,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
        min_delta=args.min_delta,
    )
