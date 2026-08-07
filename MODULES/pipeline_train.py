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
    def _random_rotate90(arr: np.ndarray) -> np.ndarray:
        """Rotação aleatória de 90° em torno de um dos 3 eixos espaciais.

        O tensor tem shape (C, X, Y, Z). Rotacionamos apenas os eixos
        espaciais (1,2,3), preservando a ordem dos canais.
        Cada chamada aplica 0-3 rotações de 90° em um eixo aleatório,
        gerando até 24 orientações distintas por sítio.
        """
        axis_pairs = [(1,2), (1,3), (2,3)]  # pares de eixos espaciais
        ax = axis_pairs[np.random.randint(3)]
        k  = np.random.randint(4)           # 0, 90, 180 ou 270 graus
        return np.rot90(arr, k=k, axes=ax).copy()

    def __getitem__(self, idx: int) -> Optional[dict]:
        row     = self.records.iloc[idx]
        pair_id = row["pair_id"]
        pair_dir = self.tensor_dir / pair_id

        # ── Tensor do sítio ──────────────────────────────────────────────────
        tensor_path = pair_dir / "site_tensor.npz"
        npz  = np.load(tensor_path, allow_pickle=True)
        arr  = npz["tensor"]                         # (X, Y, Z, C)
        arr  = np.transpose(arr, (3, 0, 1, 2))       # → (C, X, Y, Z)

        # Data augmentation: rotações aleatórias de 90° nos 3 eixos (apenas treino)
        # Multiplica o dataset efetivo por até 24x sem custo de armazenamento.
        # As rotações preservam a geometria vdW e o MEP do sítio.
        if self.split == "train":
            arr = self._random_rotate90(arr)

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

        return {
            "tensor":     site,
            "graph":      graph,
            "pair_id":    pair_id,
            "smiles":     row.get("ligand_smiles", ""),
            "voxel_size": float(row.get("voxel_size_A", 1.0)),
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
        """Agrupa itens do Dataset em mini-batches, filtrando e sanitizando NaN/Inf."""
        batch = [b for b in batch if b is not None]
        # Filtra tensores de sítio corrompidos
        batch = [b for b in batch if torch.isfinite(b["tensor"]).all()]
        if not batch:
            return None

        tensors = torch.stack([b["tensor"] for b in batch])   # (B,C,X,Y,Z)
        graphs  = Batch.from_data_list([b["graph"] for b in batch])

        # Sanitiza graph.x após Batch.from_data_list — a concatenação pode
        # preservar inf/NaN de grafos individuais mesmo após sanitização prévia
        if not torch.isfinite(graphs.x).all():
            graphs.x = torch.nan_to_num(graphs.x, nan=0.0, posinf=0.0, neginf=0.0)

        # Descarta o batch inteiro se ainda houver NaN (ex: pos corrompido)
        if not torch.isfinite(graphs.x).all():
            return None

        return {
            "tensor":      tensors,
            "graph":       graphs,
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

class GNNDecoder(nn.Module):
    """Decoder GNN que converte vetor latente z → predição sobre o grafo.

    Modo de operação:
      - Durante treino: o grafo real é fornecido (teacher forcing)
        → prediz propriedades de cada nó e aresta
      - Durante inferência: geração auto-regressiva (ver LigandGenerator)

    Saídas por nó:
      - tipo de átomo (softmax, 13 classes)
      - carga parcial (regressão)
      - posição 3D relativa ao centro do sítio (regressão)

    Saídas por aresta:
      - tipo de ligação (softmax, 4 tipos)
      - existe ligação? (sigmoid)
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
        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.use_ecfp4    = use_ecfp4
        self.ecfp4_bits   = ecfp4_bits

        # Dimensão de entrada do z_proj:
        #   sem ECFP4: latent_dim
        #   com ECFP4: latent_dim + ecfp4_bits  (z concatenado ao fingerprint)
        #
        # ECFP4 DESATIVADO por padrão (use_ecfp4=False).
        # Quando ativo, ancora o decoder em fingerprints já vistos no treino,
        # reduzindo a diversidade dos ligantes gerados — indesejável para
        # descoberta de novos scaffolds. Ativar apenas para tarefas de
        # otimização de lead (quando diversidade é menos prioritária).
        z_input_dim = latent_dim + ecfp4_bits if use_ecfp4 else latent_dim

        # Projeta z (+ ECFP4 opcional) no espaço do grafo
        self.z_proj = nn.Sequential(
            nn.Linear(z_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Projeta features dos nós para hidden_dim
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Camadas GNN (GAT com múltiplas cabeças)
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4,
                    dropout=dropout, concat=True)
            for _ in range(n_gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_gnn_layers)
        ])

        # Cabeças de predição por nó
        self.head_atom_type = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, n_atom_types)
        )
        self.head_charge = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.head_pos = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(),
            nn.Linear(128, 3)
        )

        # Cabeça de predição por aresta (concatena features dos dois nós)
        self.head_bond_type = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bond_types)
        )
        self.head_bond_exist = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        z: torch.Tensor,          # (B, latent_dim)
        graph: "PyGData",         # batch de grafos PyG
    ) -> dict:
        """
        Returns:
            dict com predições por nó e por aresta
        """
        batch_vec = graph.batch  # (total_nodes,) — qual grafo cada nó pertence

        # Concatena ECFP4 ao z se ativo — fornece impressão digital global
        # do ligante de referência ao decoder (ver nota em __init__ sobre
        # o trade-off diversidade vs fidelidade)
        if self.use_ecfp4 and hasattr(graph, "ecfp4") and graph.ecfp4 is not None:
            # graph.ecfp4 tem shape (total_nodes, ecfp4_bits) quando em batch
            # ou (ecfp4_bits,) por grafo — normaliza para (B, ecfp4_bits)
            ecfp4_global = global_mean_pool(graph.ecfp4, batch_vec)  # (B, bits)
            z_input = torch.cat([z, ecfp4_global], dim=-1)           # (B, latent+bits)
        else:
            z_input = z

        # Projeta z (+ ECFP4 opcional) e expande para cada nó do batch
        z_node = self.z_proj(z_input)[batch_vec]  # (total_nodes, hidden_dim)

        # Embedding inicial dos nós
        h = self.node_embed(graph.x) + z_node # (total_nodes, hidden_dim)

        # Passagens GNN
        edge_index = graph.edge_index
        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            h = ln(h + gnn(h, edge_index))    # residual + normalização

        # Predições por nó
        atom_logits = self.head_atom_type(h)  # (total_nodes, n_atom_types)
        charges     = self.head_charge(h)     # (total_nodes, 1)
        positions   = self.head_pos(h)        # (total_nodes, 3)

        # Predições por aresta
        i, j = edge_index
        edge_h = torch.cat([h[i], h[j]], dim=-1)    # (total_edges, 2*hidden)
        bond_logits = self.head_bond_type(edge_h)   # (total_edges, n_bond_types)
        bond_exist  = self.head_bond_exist(edge_h)  # (total_edges, 1)

        return {
            "atom_logits":  atom_logits,
            "charges":      charges.squeeze(-1),
            "positions":    positions,
            "bond_logits":  bond_logits,
            "bond_exist":   bond_exist.squeeze(-1),
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

    def forward(self, site_tensor: torch.Tensor, graph: "PyGData",
            voxel_sizes: torch.Tensor) -> dict:
        """Passagem completa (treino com teacher forcing).

        Returns dict com predições + z_mean + z_logvar para calcular a loss.
        """
        z_mean, z_logvar = self.encoder(site_tensor, voxel_sizes)
        # Clipa z_logvar para prevenir exp(logvar) → inf
        z_logvar = torch.clamp(z_logvar, -10, 10)
        z = reparametrize(z_mean, z_logvar)
        preds = self.decoder(z, graph)
        preds["z_mean"]   = z_mean
        preds["z_logvar"] = z_logvar
        return preds

    @torch.no_grad()
    def sample(
        self,
        site_tensor:    torch.Tensor,
        template_graph: "PyGData",
        n_samples:      int   = 1,
        temperature:    float = 1.0,
        voxel_sizes:    torch.Tensor = None,  
    ) -> list:
        """Gera novos ligantes condicionados no sítio.

        Args:
            site_tensor   : (1, C, X, Y, Z) — tensor do sítio alvo
            template_graph: grafo modelo (define n_atoms e topologia inicial)
            n_samples     : quantas amostras gerar
            temperature   : controla diversidade (> 1 = mais diverso)

        Returns:
            Lista de dicts com predições para cada amostra
        """
        self.eval()
        if voxel_sizes is None:
            voxel_sizes = torch.ones(site_tensor.shape[0],
                                    device=site_tensor.device)
        z_mean, z_logvar = self.encoder(site_tensor, voxel_sizes)
        results = []
        for _ in range(n_samples):
            z = z_mean + temperature * torch.randn_like(z_mean)
            preds = self.decoder(z, template_graph)
            results.append(preds)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE LOSS
# ══════════════════════════════════════════════════════════════════════════════

def cvae_loss(
    preds: dict,
    graph: "PyGData",
    beta: float = 1.0,
    pos_weight: float = 2.0,
) -> dict:
    """Loss total do CVAE = reconstruction loss + β·KL divergence.

    Componentes da reconstruction loss:
      L_atom : cross-entropy no tipo de átomo
      L_bond : cross-entropy no tipo de ligação
      L_exist: BCE na existência de ligação
      L_pos  : MSE nas coordenadas 3D (normalizado pela caixa)
      L_charge: MSE na carga parcial

    KL divergence:
      L_KL = -0.5 · Σ (1 + logvar - mean² - exp(logvar))

    O hiperparâmetro beta controla o trade-off reconstrução vs. regularização.
    beta=1 → VAE padrão; beta<1 → mais ênfase em reconstrução.
    """
    # Tipo de átomo: o voxelize codifica 13 classes no início de graph.x
    atom_target = graph.x[:, :ATOM_TYPE_DIM].argmax(dim=-1)   # (total_nodes,)
    L_atom = F.cross_entropy(preds["atom_logits"], atom_target)

    # Tipo de ligação: o voxelize codifica 4 classes no início de edge_attr
    bond_target = graph.edge_attr[:, :BOND_TYPE_DIM].argmax(dim=-1)  # (total_edges,)
    L_bond = F.cross_entropy(preds["bond_logits"], bond_target)

    # Existência de ligação (todos os pares no grafo existem → target=1)
    bond_exist_target = torch.ones(preds["bond_exist"].shape[0],
                                   device=preds["bond_exist"].device)
    L_exist = F.binary_cross_entropy_with_logits(preds["bond_exist"], bond_exist_target)

    # Posições 3D (normaliza por desvio padrão para escalar)
    if hasattr(graph, "pos") and graph.pos is not None:
        pos_target = graph.pos
        pos_scale  = pos_target.std() + 1e-6
        L_pos = F.mse_loss(preds["positions"] / pos_scale,
                           pos_target        / pos_scale)
    else:
        L_pos = torch.tensor(0.0)

    # Cargas parciais
    if graph.x.shape[1] > 19:
        charge_target = graph.x[:, 19]  # índice da carga Gasteiger no vetor
        L_charge = F.mse_loss(preds["charges"], charge_target)
    else:
        L_charge = torch.tensor(0.0)

    # KL divergence
    z_mean   = preds["z_mean"]
    z_logvar = preds["z_logvar"]
    L_KL = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    # Total
    L_recon = L_atom + L_bond + L_exist + pos_weight * L_pos + 0.5 * L_charge
    L_total = L_recon + beta * L_KL

    return {
        "total":  L_total,
        "recon":  L_recon,
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
    totals = {k: 0.0 for k in ["total","recon","atom","bond","exist","pos","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if batch is None:
            continue
        site   = batch["tensor"].to(device)
        graph  = batch["graph"].to(device)
        voxel_sizes = batch["voxel_sizes"].to(device)

        optimizer.zero_grad()
        preds  = model(site, graph, voxel_sizes)
        losses = cvae_loss(preds, graph, beta=beta)

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
    totals = {k: 0.0 for k in ["total","recon","atom","bond","exist","pos","kl"]}
    n_batches = 0

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        if batch is None:
            continue
        site        = batch["tensor"].to(device)
        graph       = batch["graph"].to(device)
        voxel_sizes = batch["voxel_sizes"].to(device)
        preds       = model(site, graph, voxel_sizes)
        losses = cvae_loss(preds, graph, beta=beta)
        # Pula batch com loss NaN (modelo corrompido ou dado degenerado)
        if not torch.isfinite(losses["total"]):
            continue
        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train(
    latent_dim:    int   = 128,
    hidden_dim:    int   = 128,
    n_gnn_layers:  int   = 3,
    batch_size:    int   = 8,
    epochs:        int   = 200,   # máximo — early stopping interrompe antes
    lr:            float = 3e-4,
    beta_max:      float = 0.5,
    beta_warmup:   int   = 30,
    dropout:       float = 0.35,
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

        log.info(f"  Train → total={train_losses['total']:.4f}  "
                 f"recon={train_losses['recon']:.4f}  kl={train_losses['kl']:.4f}")
        log.info(f"  Val   → total={val_losses['total']:.4f}  "
                 f"recon={val_losses['recon']:.4f}  kl={val_losses['kl']:.4f}")

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
                "config": {
                    "latent_dim":   latent_dim,
                    "hidden_dim":   hidden_dim,
                    "n_gnn_layers": n_gnn_layers,
                },
            }, ckpt_dir / "best_model.pt")
            log.info(f"  ★ Melhor modelo salvo (val_loss={best_val:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            log.info(f"  Sem melhora: {patience_count}/{patience} epochs")

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
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--batch",          type=int,   default=8)
    parser.add_argument("--latent",         type=int,   default=128)
    parser.add_argument("--hidden",         type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--gnn-layers",     type=int,   default=3)
    parser.add_argument("--dropout",        type=float, default=0.35)
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
