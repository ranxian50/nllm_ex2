#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 OPUS-100 en-zh 平行语料中抽取训练/验证集，保存为 TSV。

用法:
    python prepare_data.py --parquet data/opus100_train.parquet \
        --out-dir data --train-size 20000 --valid-size 1000
"""

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq


def extract_pairs(parquet_path: Path):
    table = pq.read_table(parquet_path)
    col = table.column("translation").to_pylist()
    pairs = []
    for item in col:
        if not item:
            continue
        zh, en = item.get("zh"), item.get("en")
        if not zh or not en:
            continue
        zh, en = zh.strip(), en.strip()
        # 过滤过长/过短、以及疑似非中文的样本
        if not (2 <= len(zh) <= 120 and 2 <= len(en) <= 300):
            continue
        if sum("\u4e00" <= c <= "\u9fff" for c in zh) < 2:
            continue
        pairs.append((zh.replace("\t", " "), en.replace("\t", " ")))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/opus100_train.parquet")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = extract_pairs(Path(args.parquet))
    random.Random(args.seed).shuffle(pairs)
    need = args.train_size + args.valid_size
    pairs = pairs[:need]
    train, valid = pairs[args.valid_size:], pairs[:args.valid_size]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.tsv", train), ("valid.tsv", valid)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for zh, en in data:
                f.write(f"{zh}\t{en}\n")
        print(f"{name}: {len(data)} pairs")
    print("示例:", train[0])


if __name__ == "__main__":
    main()
