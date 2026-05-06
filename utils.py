# -*- coding: utf-8 -*-
"""General utilities for SDiff-GCN: data loading, batching, logging, and metrics."""
from __future__ import annotations

import functools
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_FILES = ["user-item.pkl", "item-user.pkl", "train.pkl", "val.pkl", "test.pkl"]
EDGE_FILE_CANDIDATES = {
    "Ciao": ["ciao_gcn_edge_index_train.pkl", "gcn_edge_index_train.pkl"],
    "Epinions": ["epinions_gcn_edge_index_train.pkl", "gcn_edge_index_train.pkl"],
    "Dianping": ["dianping_gcn_edge_index_train.pkl", "gcn_edge_index_train.pkl"],
}


@dataclass
class DatasetBundle:
    dataset_dir: Path
    user_item: dict
    item_user: dict
    user_item_train: dict
    train_data: list
    val_data: list
    test_data: list
    edge_index: torch.Tensor
    all_items: set
    num_users: int
    num_items: int


class SocialDataset(Dataset):
    """Dataset containing (user, historical sequence, target item)."""

    def __init__(self, data: Iterable[tuple[int, int]], user_item: dict, min_history: int = 5):
        self.user_item = user_item
        self.data = [(u, it) for u, it in data if len(self.user_item.get(u, [])) >= min_history]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        u, it = self.data[idx]
        return u, self.user_item[u][-20:], it


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_dir(data_root: str, dataset: str) -> Path:
    root = Path(data_root)
    candidates = [root / dataset / "src_data_friend", root / dataset, root]
    for cand in candidates:
        if all((cand / name).exists() for name in REQUIRED_FILES):
            if any((cand / edge_name).exists() for edge_name in EDGE_FILE_CANDIDATES[dataset]):
                return cand
    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Could not locate dataset directory for {dataset}. Tried:\n{tried}\n"
        f"Need files: {REQUIRED_FILES} and one of {EDGE_FILE_CANDIDATES[dataset]}"
    )


def load_edge_index(dataset_dir: Path, dataset: str) -> torch.Tensor:
    for edge_name in EDGE_FILE_CANDIDATES[dataset]:
        edge_path = dataset_dir / edge_name
        if edge_path.exists():
            return torch.as_tensor(pickle.load(open(edge_path, "rb")), dtype=torch.long)
    raise FileNotFoundError(f"No edge index file found in {dataset_dir} for dataset={dataset}")


def _max_from_nested_values(item_user: dict) -> int:
    max_id = -1
    for users in item_user.values():
        if users:
            max_id = max(max_id, max(users))
    return max_id


def load_dataset_bundle(data_root: str, dataset: str) -> DatasetBundle:
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    user_item = pickle.load(open(dataset_dir / "user-item.pkl", "rb"))
    item_user = pickle.load(open(dataset_dir / "item-user.pkl", "rb"))
    train_data = pickle.load(open(dataset_dir / "train.pkl", "rb"))
    val_data = pickle.load(open(dataset_dir / "val.pkl", "rb"))
    test_data = pickle.load(open(dataset_dir / "test.pkl", "rb"))
    edge_index = load_edge_index(dataset_dir, dataset)

    train_items = {it for _, it in train_data}
    all_items = set(train_items)
    user_item_train = {u: [i for i in items if i in train_items] for u, items in user_item.items()}

    all_pairs = list(train_data) + list(val_data) + list(test_data)
    max_user_id = max(
        [max(user_item.keys(), default=-1), _max_from_nested_values(item_user), int(edge_index.max().item())]
    )
    max_item_id = max([max((it for _, it in all_pairs), default=-1), max(item_user.keys(), default=-1)])

    return DatasetBundle(
        dataset_dir=dataset_dir,
        user_item=user_item,
        item_user=item_user,
        user_item_train=user_item_train,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        edge_index=edge_index,
        all_items=all_items,
        num_users=max_user_id + 1,
        num_items=max_item_id + 1,
    )


def pad_2d(list_of_lists, pad_val: int = 0) -> torch.Tensor:
    max_len = max((len(x) for x in list_of_lists), default=1)
    max_len = max(max_len, 1)
    padded = [[*lst, *[pad_val] * (max_len - len(lst))] for lst in list_of_lists]
    return torch.tensor(padded, dtype=torch.long)


def collate_fn(batch, item_user: dict):
    users, seqs, tgt_items = zip(*batch)
    seq_pad = pad_2d(seqs)
    nei_pad = pad_2d([item_user.get(it, []) for it in tgt_items])
    return (
        torch.tensor(users, dtype=torch.long),
        seq_pad,
        nei_pad,
        torch.tensor(tgt_items, dtype=torch.long),
    )


def make_train_loader(bundle: DatasetBundle, batch_size: int, num_workers: int, min_history: int = 5):
    train_ds = SocialDataset(bundle.train_data, bundle.user_item_train, min_history=min_history)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_fn, item_user=bundle.item_user),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_ds, loader


def format_metrics(metrics: dict, ks=(10, 20)) -> str:
    parts = []
    for k in ks:
        parts.append(f"Recall@{k}={metrics[f'Recall@{k}']:.4f}")
        parts.append(f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f}")
    return " ".join(parts)


def prepare_result_dir(result_root: str, dataset: str) -> Path:
    """Create a clean dataset-level result directory that will contain only train_log.txt."""
    result_dir = Path(result_root) / dataset
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def write_log_header(log_path: Path, lines: list[str]) -> None:
    log_path.write_text("".join(lines), encoding="utf-8")


def append_log_line(log_path: Path, line: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
