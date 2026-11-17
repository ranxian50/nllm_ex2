#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 OPUS-100 中英语料上对 MarianMT (Helsinki-NLP/opus-mt-zh-en) 做领域调优。

用法:
    python train.py --train data/train.tsv --valid data/valid.tsv \
        --model-dir models/opus-mt-zh-en --output models/opus-mt-zh-en-ft
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


def read_tsv(path: Path):
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch, tokenizer, max_src=128, max_tgt=128):
    src, tgt = zip(*batch)
    model_inputs = tokenizer(
        list(src), max_length=max_src, truncation=True, padding=True, return_tensors="pt"
    )
    labels = tokenizer(
        text_target=list(tgt), max_length=max_tgt, truncation=True,
        padding=True, return_tensors="pt"
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.tsv")
    parser.add_argument("--valid", default="data/valid.tsv")
    parser.add_argument("--model-dir", default="models/opus-mt-zh-en")
    parser.add_argument("--output", default="models/opus-mt-zh-en-ft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-train", type=int, default=0, help="0 表示使用全部")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train_pairs = read_tsv(Path(args.train))
    if args.max_train:
        train_pairs = train_pairs[: args.max_train]
    valid_pairs = read_tsv(Path(args.valid))
    print(f"train={len(train_pairs)} valid={len(valid_pairs)} threads={torch.get_num_threads()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    train_loader = DataLoader(
        TranslationDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer),
        num_workers=0,
    )
    valid_loader = DataLoader(
        TranslationDataset(valid_pairs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps
    )

    best_loss = float("inf")
    start = time.time()
    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running += loss.item()
            step += 1
            if step % 50 == 0:
                avg = running / 50
                running = 0.0
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch + 1} step {step}/{total_steps} "
                    f"loss {avg:.4f} lr {lr:.2e} "
                    f"elapsed {time.time() - start:.0f}s",
                    flush=True,
                )
            if step % 1000 == 0:
                val_loss = evaluate(model, valid_loader, device)
                print(f"[eval] step {step} valid_loss {val_loss:.4f}", flush=True)
                best_loss = min(best_loss, val_loss)
            if args.max_train and step * args.batch_size >= args.max_train:
                break

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"训练完成，模型已保存到 {out}，耗时 {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
