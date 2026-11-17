#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用（微调后的）MarianMT 模型翻译《红楼梦》选段。

用法:
    python translate.py --model models/opus-mt-zh-en-ft \
        --input data/hongloumeng_sample.txt --output output/translation_ft.txt
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SENT_SPLIT = re.compile(r"(?<=[。！？!?；;.])")
PUNCT_MAP = str.maketrans({
    ",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；",
})


def split_text(text: str, max_chars: int = 90):
    """按句切分，并将过短的句子合并，避免逐句翻译丢失上下文。"""
    # 《红楼梦》电子文本多用半角标点，先归一化到中文标点
    text = text.translate(PUNCT_MAP)
    text = re.sub(r"\s+", "", text)
    units = [s for s in SENT_SPLIT.split(text) if s]
    chunks, buf = [], ""
    for u in units:
        if len(buf) + len(u) <= max_chars:
            buf += u
        else:
            if buf:
                chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/opus-mt-zh-en-ft")
    parser.add_argument("--input", default="data/hongloumeng_sample.txt")
    parser.add_argument("--output", default="output/translation.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()

    chunks = split_text(Path(args.input).read_text(encoding="utf-8"))
    translations = []
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i: i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=192,
                num_beams=args.num_beams,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    lines = ["# 原文", "", Path(args.input).read_text(encoding="utf-8").strip(), "", "# 译文", ""]
    for src, tgt in zip(chunks, translations):
        lines.append(f"原文: {src}")
        lines.append(f"译文: {tgt}")
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    for t in translations:
        print(t)


if __name__ == "__main__":
    main()
