# -*- coding: utf-8 -*-
"""Main entry for SDiff-GCN.

Outputs:
- only train_log.txt is saved;
- no csv, summary, checkpoint, efficiency file, time log, or memory log is saved.

Dropout:
- adjust it from the command line with `--dropout`;
- it is passed into both the Transformer sequence encoder and GCN user encoder.
"""
from __future__ import annotations

import argparse

import torch

from train import build_model, clone_state_dict_to_cpu, evaluate, make_optimizer, train_one_epoch
from utils import (
    SocialDataset,
    append_log_line,
    format_metrics,
    load_dataset_bundle,
    make_train_loader,
    prepare_result_dir,
    set_seed,
    write_log_header,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SDiff-GCN four-file refactor")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="Ciao", choices=["Ciao", "Epinions", "Dianping"])
    parser.add_argument("--result_root", type=str, default="./Results")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval_batch", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--diffusion_steps", type=int, default=10)

    # Dropout is adjusted here. Example: --dropout 0.3
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--diff_weight", type=float, default=1e-2)
    parser.add_argument("--var_weight", type=float, default=1.0)
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_dataset_bundle(args.data_root, args.dataset)
    _, train_loader = make_train_loader(bundle, batch_size=args.batch, num_workers=args.num_workers)
    val_ds = SocialDataset(bundle.val_data, bundle.user_item_train)
    test_ds = SocialDataset(bundle.test_data, bundle.user_item_train)

    model = build_model(args, bundle, device)
    optimizer = make_optimizer(model, args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    result_dir = prepare_result_dir(args.result_root, args.dataset)
    log_path = result_dir / "train_log.txt"
    ks = (10, 20)

    write_log_header(
        log_path,
        [
            f"Dataset: {args.dataset}\n",
            f"DataDir: {bundle.dataset_dir}\n",
            f"Epochs: {args.epochs}\n",
            f"LogInterval: {args.log_interval}\n",
            f"Dropout: {args.dropout}\n",
            f"Metrics: Recall@10 NDCG@10 Recall@20 NDCG@20\n",
            "\n",
        ],
    )

    best_score = -1.0
    best_epoch = -1
    best_state = None

    print(f"[Info] Dataset={args.dataset} DataDir={bundle.dataset_dir}")
    print(f"[Info] Dropout={args.dropout}  # change it with --dropout")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, args.grad_clip)

        should_log = (epoch % args.log_interval == 0) or (epoch == args.epochs)
        if not should_log:
            continue

        val_metrics = evaluate(
            model,
            val_ds.data,
            bundle.user_item_train,
            bundle.all_items,
            device,
            bundle.item_user,
            ks=ks,
            batch_size=args.eval_batch,
            num_neg=args.num_neg,
            rng_seed=args.seed,
        )

        score = val_metrics["Recall@10"] + val_metrics["NDCG@10"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)

        line = f"Epoch {epoch:03d} | TrainLoss={train_stats['train_loss']:.6f} | {format_metrics(val_metrics, ks)}"
        print(line)
        append_log_line(log_path, line)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_metrics = evaluate(
        model,
        test_ds.data,
        bundle.user_item_train,
        bundle.all_items,
        device,
        bundle.item_user,
        ks=ks,
        batch_size=args.eval_batch,
        num_neg=args.num_neg,
        rng_seed=args.seed,
    )

    append_log_line(log_path, "")
    append_log_line(log_path, "[Test]")
    append_log_line(log_path, format_metrics(test_metrics, ks))
    append_log_line(log_path, f"BestEpoch={best_epoch}")

    print(f"[Test] {format_metrics(test_metrics, ks)}")
    print(f"[Done] Only train_log.txt is saved in {result_dir}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
