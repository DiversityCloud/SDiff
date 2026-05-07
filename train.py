# -*- coding: utf-8 -*-
"""Model definition, training loop, and evaluation for SDiff-GCN."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from gcn_encoder import FullGraphGCNUserEncoder, GCNConditionalEncoder
from utils import DatasetBundle, pad_2d


class UserItemTransformer(nn.Module):
    """Sequence encoder for user historical items.

    Dropout is controlled by the `dropout` argument. It is applied in three
    places: item+position embedding dropout, Transformer layer dropout, and
    output dropout. In main.py, this value is exposed as `--dropout`.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        *,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        _, seq_len = item_seq.shape
        pos = torch.arange(seq_len, device=item_seq.device)
        x = self.item_embedding(item_seq) + self.pos_embedding(pos)
        x = self.embedding_dropout(x)
        h = self.encoder(x)
        return self.output_dropout(h[:, -1])


class CrossAttentionItemEncoder(nn.Module):
    def __init__(self, embed_dim: int, *, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, neighbor_user_emb: torch.Tensor, user_seq_rep: torch.Tensor) -> torch.Tensor:
        key_value = user_seq_rep.unsqueeze(1)
        attn_out, _ = self.cross_attn(neighbor_user_emb, key_value, key_value)
        return attn_out.mean(dim=1)


class SimpleDiffusion(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.002,
        default_steps: int = 10,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.default_steps = default_steps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.t_embedding = nn.Embedding(timesteps, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.denoise_out = nn.Linear(embed_dim, embed_dim)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        return a.gather(0, t).to(x.device).reshape(-1, *((1,) * (x.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.alphas_cumprod.sqrt(), t, x0)
        sqrt_1mab = self._extract((1.0 - self.alphas_cumprod).sqrt(), t, x0)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def pred_eps(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        tokens = torch.stack([x_t + self.t_embedding(t), cond], dim=1)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        return self.denoise_out(attn_out.mean(dim=1))

    def loss(self, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.pred_eps(x_t, t, cond)
        return F.mse_loss(eps, noise)

    def _f_ode(self, x_t: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor):
        beta_t = self._extract(self.betas, t_idx, x_t)
        alphac = self._extract(self.alphas_cumprod, t_idx, x_t)
        return -beta_t / torch.sqrt(1.0 - alphac) * eps

    @torch.no_grad()
    def _sample_heun2(self, shape, cond: torch.Tensor, steps: int):
        device = cond.device
        batch = shape[0]
        x = torch.randn(shape, device=device)
        t_seq = torch.linspace(self.timesteps - 1, 0, steps + 1, device=device)
        for i in range(steps):
            t0, t1 = t_seq[i], t_seq[i + 1]
            h = t1 - t0
            t0_idx = torch.full((batch,), int(t0), dtype=torch.long, device=device)
            t1_idx = torch.full_like(t0_idx, int(t1))
            eps0 = self.pred_eps(x, t0_idx, cond)
            f0 = self._f_ode(x, t0_idx, eps0)
            x_eu = x + h * f0
            eps1 = self.pred_eps(x_eu, t1_idx, cond)
            f1 = self._f_ode(x_eu, t1_idx, eps1)
            x = x + h * 0.5 * (f0 + f1)
        return x

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, cond: torch.Tensor, steps: int | None = None):
        return self._sample_heun2(x0.shape, cond, steps or self.default_steps)


class SocialRecModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        edge_index_tensor: torch.Tensor | None = None,
        gcn_layers: int = 2,
        diffusion_steps: int = 10,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        cl_weight: float = 0.1,
        rec_weight: float = 1.0,
        diff_weight: float = 1e-2,
        var_weight: float = 1.0,
    ):
        super().__init__()
        self.cl_weight = cl_weight
        self.rec_weight = rec_weight
        self.diff_weight = diff_weight
        self.var_weight = var_weight

        self.user_trans = UserItemTransformer(
            num_items,
            embed_dim,
            n_layers=transformer_layers,
            dropout=dropout,
        )
        self.cross_enc = CrossAttentionItemEncoder(embed_dim)
        gcn_encoder = FullGraphGCNUserEncoder(num_users, embed_dim, num_layers=gcn_layers, dropout=dropout)
        self.hop_agg = GCNConditionalEncoder(gcn_encoder, edge_index_tensor)

        # Share embeddings to keep the optimized lite design.
        self.neighbor_user_embedding = self.hop_agg.gcn_encoder.user_embedding
        self.item_embed = self.user_trans.item_embedding
        self.diffusion = SimpleDiffusion(embed_dim, default_steps=diffusion_steps)
        self.norm_item = nn.LayerNorm(embed_dim)
        self.norm_social = nn.LayerNorm(embed_dim)

    def _encode_item_and_social(self, user_ids: torch.Tensor, item_seq: torch.Tensor, neighbor_users: torch.Tensor):
        u_rep = self.user_trans(item_seq)
        neigh_emb = self.neighbor_user_embedding(neighbor_users)
        item_rep = self.norm_item(self.cross_enc(neigh_emb, u_rep))
        c_friends = self.norm_social(self.hop_agg(user_ids))
        return u_rep, item_rep, c_friends

    @staticmethod
    def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07):
        a_norm = F.normalize(a + 1e-6, dim=-1)
        b_norm = F.normalize(b + 1e-6, dim=-1)
        logits = a_norm @ b_norm.T
        labels = torch.arange(a.size(0), device=a.device)
        return F.cross_entropy(logits / temp, labels)

    def forward(self, user_ids, item_seq, neighbor_users, target_items=None):
        _, item_rep, c_friends = self._encode_item_and_social(user_ids, item_seq, neighbor_users)
        cl_loss = self.contrastive_loss(item_rep, c_friends)
        var_loss = 0.01 * (-torch.var(item_rep, dim=0).mean() - torch.var(c_friends, dim=0).mean())
        t_rand = torch.randint(0, self.diffusion.t_embedding.num_embeddings, (item_rep.size(0),), device=item_rep.device)
        diff_loss = self.diffusion.loss(item_rep, t_rand, c_friends)
        rec_loss = torch.tensor(0.0, device=item_rep.device)
        if target_items is not None:
            rec_loss = F.mse_loss(item_rep, self.item_embed(target_items))
        total_loss = (
            self.cl_weight * cl_loss
            + self.rec_weight * rec_loss
            + self.diff_weight * diff_loss
            + self.var_weight * var_loss
        )
        return total_loss

    @torch.no_grad()
    def encode_for_eval(self, user_ids, item_seq, neighbor_users):
        u_rep, item_rep, c_friends = self._encode_item_and_social(user_ids, item_seq, neighbor_users)
        item_rep_denoised = self.diffusion.sample(item_rep.detach(), cond=c_friends.detach())
        user_vec = 0.5 * (u_rep + c_friends)
        return user_vec, item_rep_denoised


def build_model(args, bundle: DatasetBundle, device: torch.device) -> SocialRecModel:
    model = SocialRecModel(
        bundle.num_users,
        bundle.num_items,
        args.embed_dim,
        edge_index_tensor=bundle.edge_index,
        gcn_layers=args.gcn_layers,
        diffusion_steps=args.diffusion_steps,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        cl_weight=args.cl_weight,
        rec_weight=args.rec_weight,
        diff_weight=args.diff_weight,
        var_weight=args.var_weight,
    )
    torch.set_float32_matmul_precision("high")
    return model.to(device)


def make_optimizer(model: nn.Module, lr: float):
    return optim.Adam(model.parameters(), lr=lr)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler, device: torch.device, grad_clip: float):
    model.train()
    total_loss = 0.0
    n_batches = 0
    nan_batches = 0
    for batch in loader:
        uids, seq, nei, tgt = [x.to(device, non_blocking=True) for x in batch]
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss = model(uids, seq, nei, target_items=tgt)
        if not torch.isfinite(loss):
            nan_batches += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        n_batches += 1
    return {"train_loss": total_loss / max(n_batches, 1), "nan_batches": nan_batches}


def clone_state_dict_to_cpu(model: nn.Module) -> dict:
    """Keep the best model in memory instead of writing best_model.pt."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sample_negatives(rng: random.Random, all_items_list: list[int], positive_item: int, num_neg: int):
    pool = [it for it in all_items_list if it != positive_item]
    if len(pool) <= num_neg:
        return pool
    return rng.sample(pool, num_neg)


@torch.no_grad()
def evaluate(
    model: SocialRecModel,
    data: list[tuple[int, int]],
    user_item: dict,
    all_items: set,
    device: torch.device,
    item_user: dict,
    ks=(10, 20),
    batch_size: int = 1024,
    num_neg: int = 100,
    rng_seed: int = 42,
):
    model.eval()
    rng = random.Random(rng_seed)
    ks = tuple(sorted(ks))
    metrics = {f"Recall@{k}": 0.0 for k in ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})
    all_items_list = list(all_items)
    total_samples = len(data)

    for start in range(0, total_samples, batch_size):
        batch = data[start : min(start + batch_size, total_samples)]
        users, seqs, targ_items, neg_lists = [], [], [], []
        for u, pos in batch:
            users.append(u)
            seqs.append(user_item[u][-20:])
            targ_items.append(pos)
            neg_lists.append(sample_negatives(rng, all_items_list, pos, num_neg))

        uids = torch.tensor(users, dtype=torch.long, device=device)
        seq_pad = pad_2d(seqs).to(device, non_blocking=True)
        nei_pad = pad_2d([item_user.get(it, []) for it in targ_items]).to(device, non_blocking=True)
        user_vec, item_rep_denoised = model.encode_for_eval(uids, seq_pad, nei_pad)

        cand_lists = [[pos, *negs] for pos, negs in zip(targ_items, neg_lists)]
        cand_tensor = torch.tensor(cand_lists, device=device)
        cand_emb = model.item_embed(cand_tensor)
        cand_emb[:, 0, :] = item_rep_denoised

        scores = torch.bmm(cand_emb, user_vec.unsqueeze(2)).squeeze(2)
    
        ranked_idx = scores.argsort(dim=1, descending=True)
        pos_rank = (ranked_idx == 0).nonzero(as_tuple=True)[1]

        for k in ks:
            hits = pos_rank < k
            metrics[f"Recall@{k}"] += hits.float().sum().item()
            ndcg = torch.where(
                hits,
                1.0 / torch.log2(pos_rank.float() + 2.0),
                torch.zeros_like(pos_rank, dtype=torch.float),
            )
            metrics[f"NDCG@{k}"] += ndcg.sum().item()

    for key in metrics:
        metrics[key] /= max(total_samples, 1)
    return metrics
