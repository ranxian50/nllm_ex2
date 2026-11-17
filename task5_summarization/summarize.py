#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 TextRank、法律领域词库与关键信息抽取的中文判决书自动摘要。

思路:
1. 用 THUOCL 法律词库增强 jieba 分词，突出“合同、违约、上诉、验收”等领域词；
2. 以句子为节点、TF-IDF 余弦相似度为边构建 TextRank 图，得到句子重要性；
3. 叠加判决书结构先验（首段、本院认为、判决如下、驳回上诉等位置权重）；
4. 用 MMR 去除冗余句，按原文顺序输出抽取式摘要；
5. 额外用正则抽取案号、当事人、案由、裁判结果等关键信息。

用法:
    python summarize.py --input data/judgment.txt --output output/summary.txt
"""

import argparse
import math
import re
from pathlib import Path

import jieba
import numpy as np

STOPWORDS = set("""
的 了 和 与 及 或 是 在 为 对 中 上 下 也 就 都 而 及其 以及 其 该 此 本 本案
一 二 三 四 五 六 七 八 九 十 第 条 款 项 之 等 于 由 被 将 向 从 至 按 按照
根据 依据 关于 对于 由于 因此 所以 但是 但 并 并且 且 若 如果 因 因为 所
有 无 未 不 已 已经 可以 应当 应 能够 能 会 系 即 亦 又 再 还 仍 均 皆
原告公司 被告公司 上诉人 被上诉人 原审 本院 一审 二审 涉案
""".split())

SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*")
CLEAN_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")
HEADER_RE = re.compile(
    r"^(中华人民共和国|民\s*事\s*判\s*决\s*书|行\s*政\s*判\s*决\s*书|"
    r"（\d{4}）|\(\d{4}\)|审判长|审判员|法官助理|书记员|二〇|本判决为终审判决)"
)
HEADING_RE = re.compile(r"^[（(]?[一二三四五六七八九十]+[）)、]")
CLAUSE_RE = re.compile(r"^[0-9]+[）).、]")
PARTY_RE = re.compile(r"^(上诉人|被上诉人|住所地|法定代表人|委托诉讼代理人)")


def load_lexicon(path: Path) -> dict:
    lexicon = {}
    if not path.exists():
        print(f"[warn] 词库不存在: {path}")
        return lexicon
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                lexicon[parts[0]] = int(parts[1])
    return lexicon


def build_jieba(lexicon: dict) -> None:
    for word, freq in lexicon.items():
        jieba.add_word(word, freq=min(int(math.log1p(freq) * 10), 100000))


def is_heading(s: str) -> bool:
    return bool(HEADING_RE.match(s)) and len(s) < 40 and "。" not in s


def split_sentences(text: str) -> list:
    sentences = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or HEADER_RE.match(raw):
            continue
        for s in SENT_SPLIT_RE.split(raw):
            s = s.strip()
            if len(CLEAN_RE.sub("", s)) < 8:
                continue
            if s.endswith(("：", ":")):
                continue
            if is_heading(s) or CLAUSE_RE.match(s) or PARTY_RE.match(s):
                continue
            sentences.append(s)
    return sentences


def tokenize(sentence: str) -> list:
    return [
        w for w in jieba.lcut(sentence)
        if len(w) > 1 and w not in STOPWORDS and CLEAN_RE.search(w)
    ]


def tfidf_vectors(sentences: list):
    docs = [tokenize(s) for s in sentences]
    vocab = {}
    for doc in docs:
        for w in doc:
            vocab.setdefault(w, len(vocab))
    tf = np.zeros((len(sentences), len(vocab)))
    for i, doc in enumerate(docs):
        for w in doc:
            tf[i, vocab[w]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(sentences)) / (1 + df)) + 1
    vec = tf * idf
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vec / norm


def textrank(vectors: np.ndarray, window: int = 3, damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n = len(vectors)
    sim = vectors @ vectors.T
    mask = np.zeros_like(sim, dtype=bool)
    for i in range(n):
        mask[i, max(0, i - window): min(n, i + window + 1)] = True
    sim = sim * mask
    np.fill_diagonal(sim, 0.0)
    row_sum = sim.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    trans = sim / row_sum

    score = np.ones(n) / n
    for _ in range(max_iter):
        new_score = (1 - damping) / n + damping * trans.T @ score
        if np.abs(new_score - score).sum() < tol:
            score = new_score
            break
        score = new_score
    return score


def role_bonus(sentences: list) -> np.ndarray:
    """判决书结构先验：不同位置的句子对摘要的贡献不同。"""
    bonus = np.zeros(len(sentences))
    for i, s in enumerate(sentences):
        if i < 3:
            bonus[i] += 1.0
        if "本院认为" in s[:10]:
            bonus[i] += 2.5
        if "判决如下" in s or "裁定如下" in s:
            bonus[i] += 3.0
        if "驳回上诉" in s or "维持原判" in s:
            bonus[i] += 3.0
        if "争议焦点" in s:
            bonus[i] += 2.0
        if "本院查明" in s[:10] or "原审法院认定" in s[:10]:
            bonus[i] += 1.2
        # 程序性描述对摘要价值较低
        if any(k in s for k in ("立案", "合议庭", "到庭参加", "审理终结", "询问当事人")):
            bonus[i] -= 2.0
        # 合同付款条款、金额罗列通常不是摘要重点
        if s.count("元") >= 3 or "合约总费用的" in s:
            bonus[i] -= 1.5
        if "民法典" in s and "时间效力" in s:
            bonus[i] -= 3.0
        if len(s) > 200:
            bonus[i] -= 0.5
    return bonus


def select_sentences(vectors, scores, max_sentences=9, lam=0.6) -> list:
    """MMR：在相关性和冗余度之间折中。"""
    n = len(scores)
    selected = []
    candidates = set(range(n))
    while candidates and len(selected) < max_sentences:
        best_idx, best_val = None, -1e9
        for i in candidates:
            redundancy = max((float(vectors[i] @ vectors[j]) for j in selected), default=0.0)
            val = scores[i] - lam * redundancy
            if val > best_val:
                best_idx, best_val = i, val
        selected.append(best_idx)
        candidates.remove(best_idx)
    return sorted(selected)


def extract_key_facts(text: str) -> dict:
    facts = {}
    m = re.search(r"（\d{4}）[^\s，。]{2,30}号", text)
    if m:
        facts["案号"] = m.group(0)
    m = re.search(r"上诉人（原审被告）：([^。\n]+)", text)
    if m:
        facts["上诉人"] = m.group(1).strip()
    m = re.search(r"被上诉人（原审原告）：([^。\n]+)", text)
    if m:
        facts["被上诉人"] = m.group(1).strip()
    m = re.search(r"因与被上诉人[^，。]*?([\u4e00-\u9fff]{2,20}纠纷)一案", text)
    if m:
        facts["案由"] = m.group(1)
    m = re.search(r"判决如下：\s*\n?\s*([^\n]{2,60})", text)
    if not m:
        m = re.search(r"(驳回上诉，维持原判)", text)
    if m:
        facts["裁判结果"] = m.group(1).strip()
    return facts


def summarize(text: str, lexicon_path: Path | None, max_sentences: int = 9) -> str:
    if lexicon_path:
        build_jieba(load_lexicon(lexicon_path))

    sentences = split_sentences(text)
    if not sentences:
        return ""

    vectors = tfidf_vectors(sentences)
    rank = textrank(vectors)
    rank = rank / (rank.max() + 1e-9)
    scores = rank + role_bonus(sentences)

    chosen = select_sentences(vectors, scores, max_sentences)
    facts = extract_key_facts(text)

    lines = ["## 关键信息", ""]
    for key in ("案号", "上诉人", "被上诉人", "案由", "裁判结果"):
        if key in facts:
            lines.append(f"- {key}：{facts[key]}")
    lines += ["", "## 抽取式摘要", ""]
    for i, idx in enumerate(chosen, 1):
        lines.append(f"{i}. {sentences[idx]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/judgment.txt")
    parser.add_argument("--output", default="output/summary.txt")
    parser.add_argument(
        "--lexicon",
        default="/home/ranxian/bak/nllm/work2/THUOCL：清华大学开放中文词库/THUOCL_law.txt",
    )
    parser.add_argument("--max-sentences", type=int, default=9)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    summary = summarize(text, Path(args.lexicon), args.max_sentences)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\n[info] 摘要已写入 {out}")


if __name__ == "__main__":
    main()
