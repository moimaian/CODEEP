# -*- coding: utf-8 -*-
"""
pipeline_hybrid.py — ARCH 6: decodificador HÍBRIDO (autorregressivo + equivariante) com a
ELETRÔNICA do sítio vinda do encoder 3D-CNN do CVAE.

Divisão de papéis (decisão de projeto):
  • Grafo do sítio  → SÓ GEOMETRIA: posição + raio de vdW do elemento (sem aminoácido, sem backbone).
                      Serve de repulsão estérica e de orientação da direção de crescimento.
  • Rede equivariante (estilo PaiNN) → ONDE colocar o átomo e as ligações: átomo focal, posição
                      (mistura de gaussianas relativa ao focal; a média sai do canal vetorial) e ligações.
  • Encoder 3D-CNN do CVAE (11 canais: elemento, Crippen, aromaticidade, Gasteiger, MEP, HOMO, LUMO;
                      os 21 canais de aminoácido NÃO entram) → O QUE crescer, por duas fontes de contexto:
        – z global (CVAE, com KL): tendência geral do sítio;
        – features LOCAIS: interpolação trilinear dos canais brutos e de 3 mapas de ativação da CNN na
          posição candidata do novo átomo (tipo) e nas posições dos átomos já colocados (parada).
    A eletrônica entra APENAS nas cabeças de tipo e de parada.

Ordem de construção (a do Pocket2Mol): parada → focal → posição → tipo e ligações lidos num nó de
CONSULTA colocado na posição sorteada.

Referencial: tudo em Å a partir da ORIGEM DA GRADE do par (centro do voxel 0). A rotação de 90° do
aumento de dados usa L = (N−1)·voxel — a distância entre os centros do primeiro e do último voxel.

Uso (linha de comando):
    python -m MODULES.pipeline_hybrid selftest
    python -m MODULES.pipeline_hybrid cache  --tensor-dir data/tensors
    python -m MODULES.pipeline_hybrid train  --tensor-dir data/tensors --checkpoint-dir checkpoints/hybrid
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

log = logging.getLogger(__name__)

HYBRID_ARCH_VERSION = 6
CKPT_NAME = "best_model_hybrid.pt"

# ── Constantes ────────────────────────────────────────────────────────────────
N_CAN = 11            # canais de entrada do encoder: 0..10 do tensor do sítio (sem os 21 de aminoácido)
N_TIPOS = 13          # ATOM_TYPE_DIM do pipeline_train
N_LIG = 4             # tipos de ligação: simples, dupla, tripla, aromática
QUERY = 19            # token do nó de consulta (6 elementos do bolso + 13 tipos do ligante → índice 19)
K_VIZ, R_CUT, R_CROP = 16, 10.0, 12.0   # kNN do grafo, corte de mensagens, recorte do bolso (Å)
R_LIG = 4.5           # candidatos a ligação: átomos já colocados a menos disso do novo átomo
EL_BOLSO = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}        # demais (metais, halogênios) → 5
RAIO = torch.tensor([1.70, 1.55, 1.52, 1.80, 1.80, 1.70])  # C N O S P outros (raio de vdW do bolso)
RAIO_BOLSO_NP = RAIO.numpy().astype(np.float32)
RAIO_LIG_GENERICO = 1.55   # o tipo ainda não é conhecido quando a posição é sorteada
MAXVAL = np.array([4, 3, 2, 6, 5, 1, 1, 1, 1, 6, 4, 2, 4], np.float32)   # valência máxima por tipo do ligante
ORDEM = np.array([1.0, 2.0, 3.0, 1.5], np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DADOS
# ══════════════════════════════════════════════════════════════════════════════
def _parse_bolso(path) -> tuple[np.ndarray, np.ndarray]:
    """Átomos PESADOS do bolso (binding_site.pdb): posição (Å, referencial cristalográfico) e classe 0..5."""
    pos, el = [], []
    for ln in open(path, encoding="utf-8"):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        e = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        if e in ("H", "D"):
            continue
        pos.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        el.append(EL_BOLSO.get(e, 5))
    return np.array(pos, np.float32), np.array(el, np.int64)


def _registro_par(args) -> Optional[dict]:
    """Um par → registro com bolso e ligante no referencial da grade, na ordem de construção do treino.
    O focal de cada passo é o vizinho ligado colocado mais cedo (-1: primeiro átomo / reinício)."""
    pair_id, tensor_dir, curated_dir = args
    try:
        tdir = Path(tensor_dir) / pair_id
        g = torch.load(tdir / "ligand_graph.pt", weights_only=False)
        if getattr(g, "atom_order", None) is None:
            return None
        npz = np.load(tdir / "site_tensor.npz", allow_pickle=True)
        origin = np.asarray(npz["origin"], np.float32)
        vs = float(np.asarray(npz["voxel_size"]).reshape(-1)[0])
        N = int(npz["tensor"].shape[0])
        order = g.atom_order.numpy(); n = len(order)
        pos = g.pos.numpy().astype(np.float32)[order]
        el = g.x[:, :N_TIPOS].argmax(1).numpy()[order].astype(np.int64)
        inv = np.empty(n, np.int64); inv[order] = np.arange(n)
        ei = g.edge_index.numpy(); viz = [[] for _ in range(n)]
        for a, b in zip(ei[0], ei[1]):
            viz[inv[a]].append(inv[b])
        parent = np.array([min([v for v in viz[t] if v < t], default=-1) for t in range(n)], np.int64)
        ea = g.edge_attr[:, :N_LIG].argmax(1).numpy(); lig = {}
        for k, (a_, b_) in enumerate(zip(ei[0], ei[1])):
            i_, j_ = sorted((int(inv[a_]), int(inv[b_]))); lig[(i_, j_)] = int(ea[k])
        bonds = np.array([(i_, j_, t_) for (i_, j_), t_ in sorted(lig.items())], np.int64).reshape(-1, 3)
        pk, pk_el = _parse_bolso(Path(curated_dir) / pair_id / "binding_site.pdb")
        if len(pk) < 20 or not np.isfinite(pos).all():
            return None
        if np.linalg.norm(pos.mean(0) - pk.mean(0)) > 30.0:      # ligante fora do bolso = dado ruim
            return None
        pk_g, lg_g = pk - origin, pos - origin
        box = (N - 1) * vs
        if bool(((lg_g < -2.0) | (lg_g > box + 2.0)).any()):     # ligante não cabe na própria grade
            return None
        return dict(pair=pair_id, pk_pos=pk_g.astype(np.float32), pk_el=pk_el, lg_pos=lg_g.astype(np.float32),
                    lg_el=el, parent=parent, bonds=bonds, origin=origin, vs=vs, N=N)
    except Exception:
        return None


def construir_cache(tensor_dir, curated_dir=None, split: str = "train", cache_dir=None,
                    max_atoms: int = 60, n_proc: Optional[int] = None, refazer: bool = False) -> Path:
    """Cria (ou reaproveita) cache/<split>.pt com os registros do divisão `split` do SiteLigandDataset
    (a MESMA divisão por cluster PDB do CVAE: os pares de teste nunca foram vistos no treino)."""
    from multiprocessing import Pool
    from MODULES.pipeline_train import SiteLigandDataset
    tensor_dir = Path(tensor_dir)
    curated_dir = Path(curated_dir) if curated_dir else tensor_dir.parent / "curated"
    cache_dir = Path(cache_dir) if cache_dir else tensor_dir.parent / "hybrid_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    saida = cache_dir / f"{split}.pt"
    if saida.exists() and not refazer:
        return saida
    ids = SiteLigandDataset(tensor_dir=tensor_dir, split=split, max_atoms=max_atoms).records["pair_id"].tolist()
    from MODULES import platform_utils
    with Pool(n_proc or max(1, min(16, (os.cpu_count() or 2) - 1) if not platform_utils.IS_WINDOWS else min(6, os.cpu_count() or 2))) as p:
        res = p.map(_registro_par, [(i, str(tensor_dir), str(curated_dir)) for i in ids], chunksize=64)
    ok = [r for r in res if r is not None]
    torch.save(ok, saida)
    log.info(f"  cache híbrido '{split}': {len(ok)}/{len(ids)} pares → {saida}")
    return saida


def _rot_pos(pos: np.ndarray, ax: tuple, k: int, L: float) -> np.ndarray:
    """Mesma rotação que np.rot90(grade, k, axes=ax) aplica à grade; L = (N−1)·voxel."""
    if k == 0:
        return pos
    a, b = ax[0] - 1, ax[1] - 1; out = pos.copy()
    for _ in range(k):
        ua, ub = out[:, a].copy(), out[:, b].copy(); out[:, a] = L - ub; out[:, b] = ua
    return out


def _carrega_site(tensor_dir, pair) -> np.ndarray:
    t = np.load(Path(tensor_dir) / pair / "site_tensor.npz", allow_pickle=True)["tensor"][..., :N_CAN]
    return np.ascontiguousarray(np.transpose(t, (3, 0, 1, 2)), dtype=np.float32)      # (C,X,Y,Z)


# ── grafos ────────────────────────────────────────────────────────────────────
def _grafo(pos, el):
    D = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)); np.fill_diagonal(D, 1e9)
    k = min(K_VIZ, len(pos) - 1); nb = np.argsort(D, 1)[:, :k]
    dst = np.repeat(np.arange(len(pos)), k); src = nb.reshape(-1); keep = D[dst, src] < R_CUT
    return dict(pos=pos.astype(np.float32), el=el, src=src[keep], dst=dst[keep])


def _recorte(pk, lg_parcial):
    if len(lg_parcial) == 0:
        return np.arange(len(pk))
    d = np.sqrt(((pk[:, None] - lg_parcial[None]) ** 2).sum(-1)).min(1); sel = np.where(d < R_CROP)[0]
    return sel if len(sel) >= 8 else np.argsort(d)[:8]


def _montar_A(pk_el_all, pk, lg, lg_el, t):
    """grafo A = bolso (recortado) + ligante parcial lg[:t]."""
    sel = _recorte(pk, lg[:t])
    pos = np.concatenate([pk[sel], lg[:t]], 0); el = np.concatenate([pk_el_all[sel], 6 + lg_el[:t]])
    return sel, _grafo(pos, el)


def _add_consulta(g, ponto):
    return _grafo(np.concatenate([g["pos"], ponto[None]], 0), np.concatenate([g["el"], [QUERY]]))


def _cat(grafos):
    off, P, EL, S, D, G = 0, [], [], [], [], []
    for gi, g in enumerate(grafos):
        n = len(g["pos"]); P.append(g["pos"]); EL.append(g["el"]); S.append(g["src"] + off); D.append(g["dst"] + off)
        G.append(np.full(n, gi)); off += n
    c = np.concatenate; el = torch.from_numpy(c(EL))
    return dict(pos=torch.from_numpy(c(P)), el=el, src=torch.from_numpy(c(S)), dst=torch.from_numpy(c(D)),
                graf=torch.from_numpy(c(G)), n_graf=len(grafos), is_pk=el < 6)


def _offsets(grafos):
    o, r = 0, []
    for g in grafos:
        r.append(o); o += len(g["pos"])
    return r


class PassosHibrido(torch.utils.data.Dataset):
    """Amostra (par, t), t = 0..n (t = n é o passo de PARADA).
    treino=True: t sorteado (p0 = P(passo 0), pstop = P(passo de parada)), rotação de 90° da grade e das
    posições, ruído na posição de consulta (σ~U(0,sig_q)) e no contexto ligante do passo B (σ~U(0,sig_ctx)),
    que dá robustez à posição SORTEADA na geração (a cabeça de tipo é sensível a ela — ver relatório)."""

    def __init__(self, cache, tensor_dir, itens=None, p0=0.2, pstop=0.15, treino=True, sig_q=0.5, sig_ctx=0.5):
        self.c, self.tdir, self.itens, self.p0, self.pstop, self.treino = cache, Path(tensor_dir), itens, p0, pstop, treino
        self.sig_q, self.sig_ctx = sig_q, sig_ctx

    def __len__(self):
        return len(self.itens) if self.itens is not None else len(self.c)

    def _t(self, d):
        n = len(d["lg_el"]); v = [t for t in range(1, n) if d["parent"][t] >= 0]; r = np.random.rand()
        if r < self.p0 or not v: return 0
        if r < self.p0 + self.pstop: return n
        return v[np.random.randint(len(v))]

    def __getitem__(self, i):
        if self.itens is not None: ip, t = self.itens[i]
        else: ip = i; t = self._t(self.c[i])
        d = self.c[ip]; n = len(d["lg_el"]); N, vs = d["N"], d["vs"]
        site = _carrega_site(self.tdir, d["pair"]); pk, lg = d["pk_pos"], d["lg_pos"]
        if self.treino:
            ax = [(1, 2), (1, 3), (2, 3)][np.random.randint(3)]; k = np.random.randint(4)
            if k:
                site = np.ascontiguousarray(np.rot90(site, k=k, axes=ax)); L = (N - 1) * vs
                pk, lg = _rot_pos(pk, ax, k, L), _rot_pos(lg, ax, k, L)
        sel, A = _montar_A(d["pk_el"], pk, lg, d["lg_el"], t); n_pk = len(sel)
        it = dict(A=A, B=None, n_pk=n_pk, t=t, ip=ip, stop=-1.0, foc=-1, foc_lig=-1, t0=(t == 0), has_pos=False,
                  alvo=np.zeros(3, np.float32), tipo=-1, cand=np.zeros(0, np.int64), cand_lab=np.zeros(0, np.int64),
                  site=torch.from_numpy(site), vs=vs)
        if t >= 1: it["stop"] = float(t == n)
        if t < n:
            it["alvo"] = lg[t].astype(np.float32); it["tipo"] = int(d["lg_el"][t])
            if t == 0:
                it["foc"] = int(np.argmin(((pk - lg[0]) ** 2).sum(1))); it["has_pos"] = True
            elif d["parent"][t] >= 0:
                it["foc_lig"] = n_pk + int(d["parent"][t]); it["foc"] = it["foc_lig"]; it["has_pos"] = True
            ponto = lg[t].astype(np.float32)
            if self.treino:
                ponto = ponto + np.random.randn(3).astype(np.float32) * (np.random.rand() * self.sig_q)
            pos, el = A["pos"].copy(), A["el"].copy()
            if self.treino and t > 0:
                pos[n_pk:n_pk + t] += np.random.randn(t, 3).astype(np.float32) * (np.random.rand() * self.sig_ctx)
            it["B"] = _add_consulta(_grafo(pos, el), ponto)
            if t >= 1:
                dist = np.sqrt(((pos[n_pk:n_pk + t] - ponto) ** 2).sum(1)); cand = np.where(dist < R_LIG)[0]
                lig = {(int(i_), int(j_)): int(b_) for i_, j_, b_ in d["bonds"]}
                it["cand"] = n_pk + cand; it["cand_lab"] = np.array([lig.get((int(j), t), -1) for j in cand], np.int64)
        return it


def collate_hibrido(lst):
    A = _cat([it["A"] for it in lst]); offA = _offsets([it["A"] for it in lst]); nb = len(lst)
    A["focal"] = torch.tensor([it["foc"] + o if it["foc"] >= 0 else 0 for it, o in zip(lst, offA)])
    A["has_pos"] = torch.tensor([it["has_pos"] for it in lst])
    A["foc_lig"] = torch.tensor([it["foc_lig"] + o if it["foc_lig"] >= 0 else -1 for it, o in zip(lst, offA)])
    A["t0"] = torch.tensor([it["t0"] for it in lst]); A["stop"] = torch.tensor([it["stop"] for it in lst])
    A["alvo"] = torch.from_numpy(np.stack([it["alvo"] for it in lst])); A["t"] = torch.tensor([it["t"] for it in lst])
    A["sid"] = torch.arange(nb)
    idxB = [k for k, it in enumerate(lst) if it["B"] is not None]; B = None
    if idxB:
        gB = [lst[k]["B"] for k in idxB]; B = _cat(gB); offB = _offsets(gB)
        B["amostra"] = torch.tensor(idxB); B["sid"] = B["amostra"].clone()
        B["q"] = torch.tensor([o + len(g["pos"]) - 1 for g, o in zip(gB, offB)]); B["tipo"] = torch.tensor([lst[k]["tipo"] for k in idxB])
        cj, cq, cl, cg = [], [], [], []
        for gi, (k, o) in enumerate(zip(idxB, offB)):
            c = lst[k]["cand"]; cj.append(c + o); cq.append(np.full(len(c), B["q"][gi].item())); cl.append(lst[k]["cand_lab"]); cg.append(np.full(len(c), gi))
        B["cand_j"] = torch.from_numpy(np.concatenate(cj).astype(np.int64)); B["cand_q"] = torch.from_numpy(np.concatenate(cq).astype(np.int64))
        B["cand_lab"] = torch.from_numpy(np.concatenate(cl).astype(np.int64)); B["cand_g"] = torch.from_numpy(np.concatenate(cg).astype(np.int64))
    return dict(A=A, B=B, site=torch.stack([it["site"] for it in lst]),
                vs=torch.tensor([it["vs"] for it in lst], dtype=torch.float32))


def _para(b, dev):
    mv = lambda d: None if d is None else {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    return dict(A=mv(b["A"]), B=mv(b["B"]), site=b["site"].to(dev), vs=b["vs"].to(dev))


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════
class RBF(nn.Module):
    def __init__(self, n=16, rc=10.0):
        super().__init__()
        self.register_buffer("c", torch.linspace(0, rc, n)); self.g = (n / rc) ** 2 * 0.5

    def forward(self, d):
        return torch.exp(-self.g * (d.unsqueeze(-1) - self.c) ** 2)


class Camada(nn.Module):
    """camada de passagem de mensagens equivariante (escalares s e vetores v), estilo PaiNN."""
    def __init__(self, F, n_rbf, rc):
        super().__init__()
        self.rc = rc; self.ln = nn.LayerNorm(F)
        self.phi = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F)); self.W = nn.Linear(n_rbf, 3 * F)
        self.U = nn.Linear(F, F, bias=False); self.V = nn.Linear(F, F, bias=False)
        self.upd = nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F))

    def forward(self, s, v, src, dst, rbf, rhat, fc):
        x = self.phi(self.ln(s))[src] * (self.W(rbf) * fc.unsqueeze(-1)); ds, dvr, dvv = x.chunk(3, -1)
        mv = v[src] * dvv.unsqueeze(-1) + dvr.unsqueeze(-1) * rhat.unsqueeze(1)
        s = s + torch.zeros_like(s).index_add_(0, dst, ds); v = v + torch.zeros_like(v).index_add_(0, dst, mv)
        Uv = self.U(v.transpose(1, 2)).transpose(1, 2); Vv = self.V(v.transpose(1, 2)).transpose(1, 2)
        a = self.upd(torch.cat([s, torch.sqrt((Vv ** 2).sum(-1) + 1e-8)], -1)); avv, asv, ass = a.chunk(3, -1)
        v = v + avv.unsqueeze(-1) * Uv; s = s + asv * (Uv * Vv).sum(-1) + ass
        return s, v


def trilin(maps, mb, u):
    """Interpolação trilinear. maps (B,C,X,Y,Z); mb (M,) índice do mapa de cada ponto;
    u (M,3) em unidades de índice do mapa → (M,C). Borda replicada (fora da caixa usa o voxel da borda)."""
    B, C, X, Y, Z = maps.shape; mp = maps.permute(0, 2, 3, 4, 1)
    dim = torch.tensor([X, Y, Z], device=u.device, dtype=u.dtype)
    u = torch.minimum(torch.maximum(u, torch.zeros_like(u)), dim - 1)
    i0 = torch.minimum(u.floor().long(), (dim - 2).long()); f = u - i0; out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w.unsqueeze(-1) * mp[mb, i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


class Res3D(nn.Module):
    def __init__(self, c, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.GELU(), nn.Dropout3d(p),
                               nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c))

    def forward(self, x):
        return Fn.gelu(x + self.f(x))


class SiteEnc(nn.Module):
    """3D-CNN do CVAE: (z_mean, z_logvar) + mapas pré-pooling [32³×16, 16³×32, 8³×64]."""
    def __init__(self, in_ch=N_CAN, latent=256, p=0.1):
        super().__init__()
        def bloco(i, o): return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.GELU(), Res3D(o, p))
        self.b1, self.b2, self.b3 = bloco(in_ch, 16), bloco(16, 32), bloco(32, 64); self.pool = nn.MaxPool3d(2)
        self.fc = nn.Sequential(nn.Linear(64 * 4 ** 3 + 1, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
                                nn.Linear(256, 256), nn.GELU())
        self.mean, self.logvar = nn.Linear(256, latent), nn.Linear(256, latent)

    def forward(self, x, vs):
        f1 = self.b1(x); f2 = self.b2(self.pool(f1)); f3 = self.b3(self.pool(f2)); h = self.pool(f3).flatten(1)
        h = self.fc(torch.cat([h, vs[:, None]], 1))
        return self.mean(h), self.logvar(h).clamp(-10, 10), [f1, f2, f3]


class Local(nn.Module):
    """features locais numa posição: canais brutos + mapas da CNN (trilinear) → F."""
    ESTR = (1, 1, 2, 4)      # passo de cada mapa em voxels da grade original

    def __init__(self, F):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(N_CAN + 16 + 32 + 64 + 1, F), nn.SiLU(), nn.Linear(F, F))

    def forward(self, maps, mb, pos, vs):
        u = pos / vs.unsqueeze(-1); N = maps[0].shape[-1]
        fs = [trilin(m, mb, (u - (s - 1) / 2) / s) for m, s in zip(maps, self.ESTR)]     # centro da célula s: (s−1)/2
        dentro = ((u >= -0.5) & (u <= N - 0.5)).all(-1, keepdim=True).float()
        return self.mlp(torch.cat(fs + [dentro], -1))


class Rede(nn.Module):
    """codificador equivariante GEOMÉTRICO: raio de vdW no bolso; tipo no ligante; token de consulta."""
    def __init__(self, F, L, n_rbf, rc):
        super().__init__()
        self.F, self.rc = F, rc
        self.rad = nn.Sequential(nn.Linear(1, F), nn.SiLU(), nn.Linear(F, F))
        self.e_lig = nn.Embedding(N_TIPOS + 1, F); self.e_kind = nn.Embedding(3, F)
        self.register_buffer("raio", RAIO.clone()); self.rbf = RBF(n_rbf, rc)
        self.camadas = nn.ModuleList([Camada(F, n_rbf, rc) for _ in range(L)])

    def forward(self, b):
        el = b["el"]; q = el == QUERY; pk = el < 6; kind = torch.where(pk, 0, torch.where(q, 2, 1))
        sp = self.rad(((self.raio[el.clamp(max=5)] - 1.65) / 0.15).unsqueeze(-1))
        sl = self.e_lig(torch.where(q, torch.full_like(el, N_TIPOS), (el - 6).clamp(0, N_TIPOS - 1)))
        s = torch.where(pk.unsqueeze(-1), sp, sl) + self.e_kind(kind)
        v = torch.zeros(s.shape[0], self.F, 3, device=s.device)
        d3 = b["pos"][b["src"]] - b["pos"][b["dst"]]; d = torch.sqrt((d3 ** 2).sum(-1) + 1e-12); rhat = d3 / d.unsqueeze(-1)
        rbf = self.rbf(d); fc = 0.5 * (torch.cos(math.pi * d / self.rc) + 1.0) * (d < self.rc)
        for c in self.camadas: s, v = c(s, v, b["src"], b["dst"], rbf, rhat, fc)
        return s, v


def _seg_softmax(x, idx):
    from torch_geometric.utils import softmax as seg_softmax
    return seg_softmax(x, idx)


class ModeloHibrido(nn.Module):
    def __init__(self, F=64, L=4, K=6, n_rbf=16, rc=10.0, latent=256, sem_eletro=False):
        super().__init__()
        self.F, self.K, self.sem_eletro = F, K, sem_eletro
        self.enc = SiteEnc(N_CAN, latent); self.zp = nn.Sequential(nn.Linear(latent, F), nn.SiLU())
        self.loc_stop, self.loc_tipo = Local(F), Local(F)
        self.redeA, self.redeB = Rede(F, L, n_rbf, rc), Rede(F, L, n_rbf, rc); self.rbf = RBF(n_rbf, rc)
        self.focal = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 1))
        self.focal_lig = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 1))
        self.mu = nn.Linear(F, K, bias=False)
        self.sig = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, K))
        self.pi = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, K))
        self.stop = nn.Sequential(nn.Linear(4 * F, F), nn.SiLU(), nn.Linear(F, 1))
        self.tipo = nn.Sequential(nn.Linear(3 * F, F), nn.SiLU(), nn.Linear(F, N_TIPOS))
        self.e_tipo = nn.Embedding(N_TIPOS, F)
        self.par = nn.Sequential(nn.Linear(3 * F + n_rbf, F), nn.SiLU(), nn.Linear(F, F), nn.SiLU())
        self.liga = nn.Linear(F, 1); self.liga_tipo = nn.Linear(F, N_LIG)
        self.abl = set()     # {"z","local"}: troca pela média do lote (remove a informação POR bolso) — diagnóstico

    # ── contexto eletrônico ──
    def contexto(self, site, vs, amostrar=True):
        zm, lv, maps = self.enc(site / 5.0, vs)
        z = zm + torch.exp(0.5 * lv) * torch.randn_like(zm) if amostrar else zm
        kl = 0.5 * (zm ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
        return dict(zf=self.zp(z), maps=[site / 5.0] + maps, mb=torch.arange(len(site), device=site.device), vs=vs, kl=kl)

    def _abl(self, x, nome):
        return x.mean(0, keepdim=True).expand_as(x) if nome in self.abl else x

    def _loc(self, mod, ctx, sid, pos):
        if self.sem_eletro: return torch.zeros(len(pos), self.F, device=pos.device)
        return self._abl(mod(ctx["maps"], ctx["mb"][sid], pos, ctx["vs"][sid]), "local")

    # ── geometria ──
    def mdn(self, s, v, idx):
        vf, sf = v[idx], s[idx]
        return (self.mu(vf.transpose(1, 2)).transpose(1, 2), Fn.softplus(self.sig(sf)) + 0.05, Fn.log_softmax(self.pi(sf), -1))

    @staticmethod
    def nll(mu, sigma, logpi, alvo_rel):
        d2 = ((alvo_rel.unsqueeze(1) - mu) ** 2).sum(-1)
        return -torch.logsumexp(logpi - 3 * torch.log(sigma) - d2 / (2 * sigma ** 2) - 1.5 * math.log(2 * math.pi), -1)

    def focal_logp(self, s, A):
        lg = self.focal(s).squeeze(-1).masked_fill(~A["is_pk"], -1e9); p = _seg_softmax(lg, A["graf"])
        return torch.log(p.clamp_min(1e-12)), p

    def focal_lig_logp(self, s, A):
        lg = self.focal_lig(s).squeeze(-1).masked_fill(~(A["el"] >= 6), -1e9)
        return torch.log(_seg_softmax(lg, A["graf"]).clamp_min(1e-12))

    def passo_A(self, A, ctx):
        """parada + estado (s, v) para focal/posição."""
        s, v = self.redeA(A); nG, F = A["n_graf"], self.F; gA = A["graf"]; eh = A["el"] >= 6
        cl = torch.zeros(nG, device=s.device).index_add_(0, gA, eh.float()).clamp_min(1)
        cp = torch.zeros(nG, device=s.device).index_add_(0, gA, (~eh).float()).clamp_min(1)
        sl = torch.zeros(nG, F, device=s.device).index_add_(0, gA, s * eh.unsqueeze(-1))
        sp = torch.zeros(nG, F, device=s.device).index_add_(0, gA, s * (~eh).unsqueeze(-1))
        idx = eh.nonzero().squeeze(-1); loc = torch.zeros(nG, F, device=s.device)
        if len(idx):
            lf = self._loc(self.loc_stop, ctx, A["sid"][gA[idx]], A["pos"][idx])
            loc = loc.index_add_(0, gA[idx], lf) / cl.unsqueeze(-1)
        zf = torch.zeros_like(sl) if self.sem_eletro else self._abl(ctx["zf"][A["sid"]], "z")
        stop = self.stop(torch.cat([sl / cl.unsqueeze(-1), sp / cp.unsqueeze(-1), zf, loc], -1)).squeeze(-1)
        return s, v, stop

    def passo_B(self, B, ctx):
        """tipo (geometria do nó de consulta + z global + eletrônica local na posição candidata)."""
        s, v = self.redeB(B); q = B["q"]
        zf = torch.zeros(len(q), self.F, device=s.device) if self.sem_eletro else self._abl(ctx["zf"][B["sid"]], "z")
        loc = self._loc(self.loc_tipo, ctx, B["sid"], B["pos"][q])
        return s, self.tipo(torch.cat([s[q], zf, loc], -1))

    def ligacoes(self, B, s, tipo_idx):
        sj, sq = s[B["cand_j"]], s[B["cand_q"]]
        d3 = B["pos"][B["cand_j"]] - B["pos"][B["cand_q"]]; d = torch.sqrt((d3 ** 2).sum(-1) + 1e-12)
        h = self.par(torch.cat([sq, self.e_tipo(tipo_idx[B["cand_g"]]), sj, self.rbf(d)], -1))
        return self.liga(h).squeeze(-1), self.liga_tipo(h)

    def forward(self, b, amostrar=None):
        amostrar = self.training if amostrar is None else amostrar
        ctx = self.contexto(b["site"], b["vs"], amostrar); A, B = b["A"], b["B"]
        s, v, stop = self.passo_A(A, ctx); out = dict(stop=stop, kl=ctx["kl"])
        mu, sigma, logpi = self.mdn(s, v, A["focal"]); out["nll"] = self.nll(mu, sigma, logpi, A["alvo"] - A["pos"][A["focal"]])
        if A["t0"].any(): out["foc0_logp"] = self.focal_logp(s, A)[0]
        if (A["foc_lig"] >= 0).any(): out["foc_lig_logp"] = self.focal_lig_logp(s, A)
        if B is not None:
            sB, tl = self.passo_B(B, ctx); out["tipo_logits"] = tl; out["lig_exist"], out["lig_tipo"] = self.ligacoes(B, sB, B["tipo"])
        return out


# ══════════════════════════════════════════════════════════════════════════════
# PERDAS E AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def perdas_hibrido(o, b, beta):
    A, B = b["A"], b["B"]; L = {}; z0 = torch.zeros((), device=A["pos"].device)
    L["pos"] = o["nll"][A["has_pos"]].mean(); t0 = A["t0"]
    L["foc0"] = -o["foc0_logp"][A["focal"][t0]].mean() if t0.any() else z0
    fl = A["foc_lig"] >= 0; L["foclig"] = -o["foc_lig_logp"][A["foc_lig"][fl]].mean() if fl.any() else z0
    sv = A["stop"] >= 0; L["stop"] = Fn.binary_cross_entropy_with_logits(o["stop"][sv], A["stop"][sv]) if sv.any() else z0
    if B is not None:
        L["tipo"] = Fn.cross_entropy(o["tipo_logits"], B["tipo"])
        if len(B["cand_lab"]):
            p = B["cand_lab"] >= 0; L["exist"] = Fn.binary_cross_entropy_with_logits(o["lig_exist"], p.float())
            L["btipo"] = Fn.cross_entropy(o["lig_tipo"][p], B["cand_lab"][p]) if p.any() else z0
        else: L["exist"] = L["btipo"] = z0
    else: L["tipo"] = L["exist"] = L["btipo"] = z0
    L["kl"] = o["kl"]
    L["total"] = L["pos"] + 0.5 * L["foc0"] + L["foclig"] + L["stop"] + L["tipo"] + L["exist"] + 0.5 * L["btipo"] + beta * L["kl"]
    return L


@torch.no_grad()
def avaliar_hibrido(m, cache, tensor_dir, dev, n_pairs=400, seed=0, bs=64):
    """métricas de validação (teacher forcing, posições exatas) + ABLAÇÕES da eletrônica:
    sem_z / sem_local / sem_ambos (troca pela média do lote) e troca_bolso (grade de outro par)."""
    m.eval(); rng = np.random.RandomState(seed); itens = []
    for i in range(min(n_pairs, len(cache))):
        d = cache[i]; n = len(d["lg_el"]); v = [t for t in range(1, n) if d["parent"][t] >= 0]
        itens += [(i, 0), (i, n)] + ([(i, int(t)) for t in rng.choice(v, min(2, len(v)), replace=False)] if v else [])
    ds = PassosHibrido(cache, tensor_dir, itens=itens, treino=False)
    acc = {k: [] for k in ["pos", "fl", "ex_p", "ex_y", "bt"]}
    abl = {n: dict(tipo=[], ce=[], st_p=[], st_y=[]) for n in ["real", "sem_z", "sem_local", "sem_ambos", "troca_bolso"]}
    for k in range(0, len(ds), bs):
        b = _para(collate_hibrido([ds[j] for j in range(k, min(k + bs, len(ds)))]), dev)
        for nome in abl:
            m.abl = {"real": set(), "sem_z": {"z"}, "sem_local": {"local"}, "sem_ambos": {"z", "local"}, "troca_bolso": set()}[nome]
            bb = dict(b, site=torch.roll(b["site"], 1, 0), vs=torch.roll(b["vs"], 1, 0)) if nome == "troca_bolso" else b
            o = m(bb, amostrar=False); A, B = b["A"], b["B"]; sv = A["stop"] >= 0
            abl[nome]["st_p"].append(torch.sigmoid(o["stop"][sv]).cpu().numpy()); abl[nome]["st_y"].append(A["stop"][sv].cpu().numpy())
            if B is not None:
                abl[nome]["tipo"].append((o["tipo_logits"].argmax(-1) == B["tipo"]).float().cpu().numpy())
                abl[nome]["ce"].append(Fn.cross_entropy(o["tipo_logits"], B["tipo"], reduction="none").cpu().numpy())
            if nome != "real": continue
            acc["pos"].append(o["nll"][A["has_pos"]].cpu().numpy()); fl = A["foc_lig"] >= 0
            if fl.any():
                for g in fl.nonzero().squeeze(-1).tolist():
                    msk = (A["graf"] == g).nonzero().squeeze(-1); acc["fl"].append(float(msk[o["foc_lig_logp"][msk].argmax()] == A["foc_lig"][g]))
            if B is not None and len(B["cand_lab"]):
                acc["ex_p"].append(torch.sigmoid(o["lig_exist"]).cpu().numpy()); acc["ex_y"].append((B["cand_lab"] >= 0).cpu().numpy()); p = B["cand_lab"] >= 0
                if p.any(): acc["bt"].append((o["lig_tipo"][p].argmax(-1) == B["cand_lab"][p]).float().cpu().numpy())
    m.abl = set(); c = np.concatenate; r = {}
    ep, ey = c(acc["ex_p"]), c(acc["ex_y"]); tp = float(((ep > 0.5) & ey).sum()); pr = tp / max(float((ep > 0.5).sum()), 1); rc = tp / max(float(ey.sum()), 1)
    r.update(nll_pos=float(c(acc["pos"]).mean()), foc_lig_acc=float(np.mean(acc["fl"])), btipo_acc=float(c(acc["bt"]).mean()), lig_f1=2 * pr * rc / max(pr + rc, 1e-9))
    for nome, a in abl.items():
        sp, sy = np.clip(c(a["st_p"]), 1e-6, 1 - 1e-6), c(a["st_y"])
        r[f"tipo_acc|{nome}"] = float(c(a["tipo"]).mean()); r[f"tipo_ce|{nome}"] = float(c(a["ce"]).mean())
        r[f"stop_acc|{nome}"] = float(((sp > 0.5) == (sy > 0.5)).mean()); r[f"stop_bce|{nome}"] = float(-(sy * np.log(sp) + (1 - sy) * np.log(1 - sp)).mean())
    return r


# ══════════════════════════════════════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════════════════════════════════════
def train_hybrid(
    tensor_dir,
    checkpoint_dir: str = "checkpoints/hybrid",
    curated_dir=None,
    cache_dir=None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    beta: float = 0.02,           # peso do KL do z global
    beta_warmup: int = 3,
    hidden_dim: int = 64,         # F da rede equivariante
    n_layers: int = 4,
    latent_dim: int = 256,
    sem_eletro: bool = False,     # controle: cabeças de tipo/parada SEM z e SEM features locais
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = "auto",
    workers: Optional[int] = None,       # None: 14 no Linux, 4 no Windows (spawn reimporta o programa)
    n_train: int = 0,
    max_iter: int = 0,
    seed: int = 0,
    epoch_callback=None,
) -> Path:
    """Treina o modelo híbrido. Devolve o caminho do melhor checkpoint (best_model_hybrid.pt)."""
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    tensor_dir = Path(tensor_dir); ck_dir = Path(checkpoint_dir); ck_dir.mkdir(parents=True, exist_ok=True)
    from MODULES import platform_utils
    workers = platform_utils.default_workers() if workers is None else workers
    tr = torch.load(construir_cache(tensor_dir, curated_dir, "train", cache_dir), weights_only=False)
    va = torch.load(construir_cache(tensor_dir, curated_dir, "val", cache_dir), weights_only=False)
    if n_train: tr = tr[:n_train]
    torch.manual_seed(seed); np.random.seed(seed)
    dl = torch.utils.data.DataLoader(PassosHibrido(tr, tensor_dir), batch_size=batch_size, shuffle=True, num_workers=workers,
                                     collate_fn=collate_hibrido, persistent_workers=workers > 0, drop_last=True,
                                     prefetch_factor=4 if workers > 0 else None)
    m = ModeloHibrido(F=hidden_dim, L=n_layers, latent=latent_dim, sem_eletro=sem_eletro).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    total = max(epochs * len(dl), 1) if not max_iter else max_iter
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    cfg = dict(arch_version=HYBRID_ARCH_VERSION, hybrid=True, F=hidden_dim, L=n_layers, latent_dim=latent_dim, sem_eletro=sem_eletro,
               n_can=N_CAN, pos_frame="grid_origin_angstrom", rot_L="(N-1)*voxel", beta=beta)
    log.info(f"  híbrido: treino={len(tr)} pares | iter/época={len(dl)} | params={sum(p.numel() for p in m.parameters())} | {dev}")
    melhor, it, hist = 1e9, 0, []
    best_path = ck_dir / CKPT_NAME
    for ep in range(1, epochs + 1):
        m.train(); t0 = time.time(); acc = {}; b_ep = beta * min(1.0, ep / max(beta_warmup, 1))
        for b in dl:
            b = _para(b, dev); L = perdas_hibrido(m(b), b, b_ep)
            opt.zero_grad(set_to_none=True); L["total"].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
            opt.step(); sched.step(); it += 1
            for k, v in L.items(): acc.setdefault(k, []).append(float(v))
            if max_iter and it >= max_iter: break
        r = avaliar_hibrido(m, va, tensor_dir, dev); tr_l = {k: float(np.mean(v)) for k, v in acc.items()}
        score = r["nll_pos"] + (1 - r["tipo_acc|real"]) + (1 - r["foc_lig_acc"]) + (1 - r["lig_f1"]) + (1 - r["stop_acc|real"])
        hist.append(dict(epoch=ep, beta=b_ep, train=tr_l, val=r, score=score))
        log.info(f"  ép {ep:02d}/{epochs} β={b_ep:.3f} | " + " ".join(f"{k}={v:.2f}" for k, v in tr_l.items())
                 + f" | val nll_pos={r['nll_pos']:.3f} tipo_acc={r['tipo_acc|real']:.3f} stop_acc={r['stop_acc|real']:.3f} | {time.time()-t0:.0f}s")
        estado = dict(model_state=m.state_dict(), config=cfg, epoch=ep, val_loss=score, val=r, history=hist)
        if score < melhor:
            melhor = score; torch.save(estado, best_path)
        torch.save(estado, ck_dir / "last_model_hybrid.pt")
        if epoch_callback is not None: epoch_callback(ep, tr_l, r)
        if max_iter and it >= max_iter: break
    return best_path


# ══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def load_hybrid(path, device="auto"):
    """Carrega best_model_hybrid.pt (arquivo ou diretório) → (modelo em eval, config)."""
    p = Path(path); p = p / CKPT_NAME if p.is_dir() else p
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    ck = torch.load(p, map_location=dev, weights_only=False); cfg = ck["config"]
    m = ModeloHibrido(F=cfg["F"], L=cfg["L"], latent=cfg.get("latent_dim", 256), sem_eletro=cfg.get("sem_eletro", False))
    m.load_state_dict(ck["model_state"]); m.to(dev).eval()
    return m, cfg


def _batch(grafos, dev):
    b = _cat(grafos); return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def gerar_rollout(model, pk_pos, pk_el, site, vs, N, dev, guarda=True, stop_thr=0.5, bond_thr=0.5, max_atoms=60,
                  n_try=60, viola=None, seed=0, temp_tipo=1.0, tau=1.0, abl=(), clash_factor=0.75):
    """Rollout em lockstep de N moléculas.

    pk_pos/pk_el : átomos pesados do bolso no REFERENCIAL DA GRADE (Å desde a origem) e classes 0..5.
    site         : tensor (C,X,Y,Z) do sítio (só os primeiros N_CAN canais são usados), vs: voxel (Å).
    guarda       : rejeita e refaz o sorteio que colide com o bolso (clash_factor × soma dos raios de vdW) ou
                   viola a vizinhança 1-3 (`viola`, por padrão pipeline_train._viola_vizinhanca).
    stop_thr     : limiar da probabilidade de parada (calibrar por modelo; ~0,5–0,7).
    tau          : escala do ruído de z (z = z_mean + tau·σ·ε, por molécula).
    Devolve lista de candidatos (atom_types, positions [grade], bonds) — somar a origem para o referencial do PDB."""
    if viola is None:
        from MODULES.pipeline_train import _viola_vizinhanca as viola
    model.eval(); model.abl = set(abl)
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    pk = np.asarray(pk_pos, np.float32); pk_el = np.asarray(pk_el, np.int64)
    lim_pk = clash_factor * (RAIO_LIG_GENERICO + RAIO_BOLSO_NP[pk_el])
    site_t = torch.as_tensor(np.asarray(site)[:N_CAN], dtype=torch.float32, device=dev)[None] / 5.0
    vs_t = torch.tensor([vs], dtype=torch.float32, device=dev)
    zm, lv, maps = model.enc(site_t, vs_t)
    zf = model.zp(zm + tau * torch.exp(0.5 * lv) * torch.randn(N, zm.shape[1], device=dev))
    ctx = dict(zf=zf, maps=[site_t] + maps, mb=torch.zeros(N, dtype=torch.long, device=dev), vs=vs_t.repeat(N))
    est = [dict(pos=[], el=[], bonds=[], uso=[], vivo=True) for _ in range(N)]

    def colide(p): return bool((np.sqrt(((pk - p) ** 2).sum(1)) < lim_pk).any())

    for t in range(max_atoms):
        ativos = [k for k in range(N) if est[k]["vivo"]]
        if not ativos: break
        novos = {}
        if t == 0:
            _, gA = _montar_A(pk_el, pk, np.zeros((0, 3), np.float32), np.zeros(0, np.int64), 0)
            A = _batch([gA], dev); A["sid"] = torch.zeros(1, dtype=torch.long, device=dev)
            s, v, _ = model.passo_A(A, ctx); _, p = model.focal_logp(s, A); pp = p.cpu().numpy()
            mu, sg, lp = model.mdn(s, v, torch.arange(len(pk), device=dev))
            mu, sg, pi = mu.cpu().numpy(), sg.cpu().numpy(), lp.exp().cpu().numpy()
            for k in ativos:
                for _ in range(n_try if guarda else 1):
                    j = rng.choice(len(pp), p=pp / pp.sum()); c = rng.choice(mu.shape[1], p=pi[j] / pi[j].sum())
                    pos = pk[j] + mu[j, c] + sg[j, c] * rng.randn(3)
                    if not guarda or not colide(pos): break
                novos[k] = (-1, pos.astype(np.float32))
        else:
            gAs, info = [], {}
            for k in ativos:
                lg = np.array(est[k]["pos"], np.float32); lge = np.array(est[k]["el"], np.int64)
                sel, g = _montar_A(pk_el, pk, lg, lge, t); gAs.append(g); info[k] = (len(sel), len(lg))
            A = _batch(gAs, dev); offs = _offsets(gAs); A["sid"] = torch.tensor(ativos, device=dev)
            s, v, stop = model.passo_A(A, ctx); pst = torch.sigmoid(stop).cpu().numpy()
            idx_lig = (A["el"] >= 6).nonzero().squeeze(-1)
            mu, sg, lp = model.mdn(s, v, idx_lig); mu, sg, pi = mu.cpu().numpy(), sg.cpu().numpy(), lp.exp().cpu().numpy()
            lgf = model.focal_lig(s).squeeze(-1).cpu().numpy(); pos_nodes = A["pos"].cpu().numpy()
            pos_lig_idx = {int(i): m_ for m_, i in enumerate(idx_lig.cpu().numpy())}
            for gi, k in enumerate(ativos):
                if pst[gi] > stop_thr: est[k]["vivo"] = False; continue
                n_sel, n_l = info[k]; base = offs[gi] + n_sel
                livre = np.array([MAXVAL[est[k]["el"][i]] - est[k]["uso"][i] >= 1.0 for i in range(n_l)])
                if not livre.any(): est[k]["vivo"] = False; continue
                z = np.where(livre, lgf[base:base + n_l], -1e9); pf = np.exp(z - z.max()); pf /= pf.sum()
                for _ in range(n_try if guarda else 1):
                    f = rng.choice(n_l, p=pf); m_ = pos_lig_idx[base + f]; c = rng.choice(mu.shape[1], p=pi[m_] / pi[m_].sum())
                    pos = pos_nodes[base + f] + mu[m_, c] + sg[m_, c] * rng.randn(3)
                    if not guarda: break
                    if colide(pos): continue
                    if viola is not None and viola(est[k]["pos"], est[k]["bonds"], f, pos): continue
                    break
                novos[k] = (f, pos.astype(np.float32))
        if not novos: continue
        # ── tipo e ligações, lidos no nó de consulta ──
        ks = list(novos.keys()); gBs = []
        for k in ks:
            if t == 0: g0 = _montar_A(pk_el, pk, np.zeros((0, 3), np.float32), np.zeros(0, np.int64), 0)[1]
            else: g0 = _montar_A(pk_el, pk, np.array(est[k]["pos"], np.float32), np.array(est[k]["el"], np.int64), t)[1]
            gBs.append(_add_consulta(g0, novos[k][1]))
        B = _batch(gBs, dev); offB = _offsets(gBs); B["sid"] = torch.tensor(ks, device=dev)
        q = torch.tensor([o + len(g["pos"]) - 1 for g, o in zip(gBs, offB)], device=dev); B["q"] = q
        sB, tl = model.passo_B(B, ctx); tl = tl / temp_tipo
        tipos = torch.multinomial(torch.softmax(tl, -1), 1).squeeze(-1)
        cj, cq, cg, cmap = [], [], [], []
        for gi, k in enumerate(ks):
            if t == 0: continue
            n_l = len(est[k]["pos"]); n_tot = len(gBs[gi]["pos"]); n_sel = n_tot - 1 - n_l
            d = np.sqrt(((np.array(est[k]["pos"]) - novos[k][1]) ** 2).sum(1))
            for j in np.where(d < R_LIG)[0]:
                cj.append(offB[gi] + n_sel + int(j)); cq.append(int(q[gi])); cg.append(gi); cmap.append((k, int(j)))
        if cj:
            B["cand_j"] = torch.tensor(cj, device=dev); B["cand_q"] = torch.tensor(cq, device=dev); B["cand_g"] = torch.tensor(cg, device=dev)
            ex, bt = model.ligacoes(B, sB, tipos); ex = torch.sigmoid(ex).cpu().numpy(); bt = bt.argmax(-1).cpu().numpy()
        by_k = {}
        for m_, (k, j) in enumerate(cmap): by_k.setdefault(k, []).append((j, float(ex[m_]), int(bt[m_])))
        for gi, k in enumerate(ks):
            f, pos = novos[k]; tipo = int(tipos[gi]); e = est[k]; novo_i = len(e["pos"]); uso_novo = 0.0; lig = []
            if t >= 1:
                for j, pe, tb in sorted(by_k.get(k, []), key=lambda x: -x[1]):
                    if j != f and pe < bond_thr: continue        # a ligação ao focal é forçada
                    ordem = ORDEM[tb]
                    for o_ in ([ordem, 1.0] if ordem > 1.0 else [ordem]):    # cai para simples se faltar valência
                        if e["uso"][j] + o_ <= MAXVAL[e["el"][j]] and uso_novo + o_ <= MAXVAL[tipo]:
                            lig.append((j, novo_i, tb if o_ == ordem else 0)); e["uso"][j] += o_; uso_novo += o_; break
            e["pos"].append(pos); e["el"].append(tipo); e["uso"].append(uso_novo); e["bonds"].extend(lig)
    model.abl = set(); out = []
    for e in est:
        if len(e["el"]) == 0: continue
        out.append(dict(atom_types=torch.tensor(e["el"], dtype=torch.long), charges=torch.zeros(len(e["el"])),
                        formal_charges=torch.zeros(len(e["el"]), dtype=torch.long),
                        positions=torch.tensor(np.array(e["pos"]), dtype=torch.float32), bonds=list(e["bonds"])))
    return out


def generate_candidates_hybrid(model, site_tensor, voxel_size, origin, site_coords, site_elements,
                               n_samples=100, site_mol=None, site_is_backbone=None, final_ff="none",
                               stop_thr=0.5, max_atoms=60, guard=True, tau=1.0, seed=0, device=None, ligand_charge_method="gas"):
    """Mesmo contrato de pipeline_generate.generate_candidates: lista de dicts {idx, mol, smiles, métricas…}.

    site_tensor : array (X,Y,Z,C) como gravado em site_tensor.npz; origin/voxel_size: da mesma grade.
    site_coords/site_elements : átomos do sítio no referencial do PDB (hidrogênios são ignorados no grafo).
    Obs.: final_ff aceita "none" e "mmff_uff"; GROMACS não é suportado neste caminho."""
    import MODULES.pipeline_generate as G
    from rdkit import Chem
    dev = device or next(model.parameters()).device
    origin = np.asarray(origin, np.float32); sc = np.asarray(site_coords, np.float32)
    pesado = np.array([str(e).upper() not in ("H", "D") for e in site_elements])
    pk_pos = sc[pesado] - origin; pk_el = np.array([EL_BOLSO.get(str(e).upper(), 5) for e in np.asarray(site_elements)[pesado]], np.int64)
    arr = np.ascontiguousarray(np.transpose(np.asarray(site_tensor), (3, 0, 1, 2)), dtype=np.float32)
    samples = gerar_rollout(model, pk_pos, pk_el, arr, voxel_size, n_samples, dev, guarda=guard, stop_thr=stop_thr,
                            max_atoms=max_atoms, tau=tau, seed=seed)
    for s_ in samples: s_["positions"] = s_["positions"] + torch.as_tensor(origin)
    cands, falhas = [], 0
    for i, c in enumerate(samples):
        try:
            mol = G.preds_to_mol(c, site_coords=site_coords, site_elements=site_elements, site_mol=site_mol,
                                 site_is_backbone=site_is_backbone, final_ff=final_ff, ligand_charge_method=ligand_charge_method)
            if mol is None or mol.GetNumAtoms() < 3: falhas += 1; continue
            smi = Chem.MolToSmiles(mol); met = G.compute_drug_metrics(mol, smi)
            if not smi or met is None: falhas += 1; continue
            hc = mol.GetBoolProp("has_pocket_clash") if mol.HasProp("has_pocket_clash") else None
            nc = mol.GetIntProp("n_pocket_clashes") if mol.HasProp("n_pocket_clashes") else None
            cands.append({"idx": i, "mol": mol, "smiles": smi, "has_pocket_clash": hc, "n_pocket_clashes": nc, **met})
        except Exception:
            falhas += 1
    log.info(f"  Gerados (híbrido): {len(cands)} válidos | {falhas} inválidos de {n_samples}")
    return cands


# ══════════════════════════════════════════════════════════════════════════════
# AUTOTESTE E CLI
# ══════════════════════════════════════════════════════════════════════════════
def selftest() -> dict:
    """Verificações sem dados: convenção de rotação, trilinear, equivariância da rede geométrica e passo A/B."""
    res = {}
    # 1) rotação: posição rodada == onde o impulso cai após np.rot90 (L=(N-1)*voxel); com N*voxel falha
    N, vs = 32, 0.9; rng = np.random.RandomState(0); e_ok, e_old = [], []
    for _ in range(100):
        ijk = rng.randint(2, N - 2, 3); t = np.zeros((1, N, N, N)); t[0, ijk[0], ijk[1], ijk[2]] = 1
        ax = [(1, 2), (1, 3), (2, 3)][rng.randint(3)]; k = rng.randint(1, 4)
        novo = np.array(np.unravel_index(np.rot90(t, k=k, axes=ax)[0].argmax(), (N, N, N))) * vs; p = (ijk * vs)[None].astype(float)
        e_ok.append(np.linalg.norm(_rot_pos(p, ax, k, (N - 1) * vs)[0] - novo)); e_old.append(np.linalg.norm(_rot_pos(p, ax, k, N * vs)[0] - novo))
    res["rot_erro_(N-1)vs"] = float(np.max(e_ok)); res["rot_erro_N*vs"] = float(np.mean(e_old)); assert res["rot_erro_(N-1)vs"] < 1e-6
    # 2) trilinear
    x = torch.zeros(1, 1, 8, 8, 8); x[0, 0, 3, 4, 5] = 1.0; z = torch.zeros(1, dtype=torch.long)
    assert abs(float(trilin(x, z, torch.tensor([[3., 4., 5.]]))) - 1) < 1e-6 and abs(float(trilin(x, z, torch.tensor([[3.5, 4., 5.]]))) - .5) < 1e-6
    # 3) equivariância: rodar e transladar o sistema roda a média da MDN e não altera os escalares
    torch.manual_seed(0); m = ModeloHibrido(F=32, L=2).eval(); pos = np.random.randn(30, 3).astype(np.float32) * 4
    el = np.array([0, 1, 2, 3] * 6 + [6 + 1, 6 + 4, 6 + 0, 6 + 2, 6 + 0, 6 + 1][:6], np.int64)[:30]
    Q, _ = np.linalg.qr(np.random.randn(3, 3)); Q = Q.astype(np.float32) * np.sign(np.linalg.det(Q)).astype(np.float32); tr = np.array([3., -2., 5.], np.float32)
    outs = []
    for P in (pos, pos @ Q.T + tr):
        b = _cat([_grafo(P, el)]); s, v = m.redeA(b); outs.append((s, v))
    d_s = float((outs[0][0] - outs[1][0]).abs().max()); d_v = float((outs[0][1] @ torch.tensor(Q.T) - outs[1][1]).abs().max())
    res["equiv_escalar"], res["equiv_vetor"] = d_s, d_v; assert d_s < 1e-3 and d_v < 1e-3
    res["ok"] = True
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ARCH 6 — decodificador híbrido com eletrônica")
    ap.add_argument("cmd", choices=["selftest", "cache", "train"])
    ap.add_argument("--tensor-dir", default="data/tensors"); ap.add_argument("--curated-dir", default=None)
    ap.add_argument("--checkpoint-dir", default="checkpoints/hybrid"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.02); ap.add_argument("--sem-eletro", action="store_true")
    ap.add_argument("--workers", type=int, default=None); ap.add_argument("--device", default="auto")
    a = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "selftest": print(json.dumps(selftest(), indent=1))
    elif a.cmd == "cache":
        for sp in ("train", "val", "test"): construir_cache(a.tensor_dir, a.curated_dir, sp)
    else:
        train_hybrid(a.tensor_dir, a.checkpoint_dir, a.curated_dir, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, beta=a.beta,
                     sem_eletro=a.sem_eletro, workers=a.workers, device=a.device)
