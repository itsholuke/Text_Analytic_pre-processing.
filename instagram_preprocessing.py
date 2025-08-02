"""Reusable helpers for Instagram caption tokenization & rolling context."""
from __future__ import annotations
import re, string
from typing import List
import pandas as pd

_HASHTAG_RE       = re.compile(r"#\w+")
_PUNCT_ONLY_RE    = re.compile(rf"^[\s{re.escape(string.punctuation)}]+$")
_SENT_SPLIT_RE    = re.compile(r"(?<=[.!?])\s+|\n")

def _sent_split_core(chunk: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(chunk.replace("\r", "\n"))
    return [p for p in parts if p]

def split_sentences(text: str | None) -> List[str]:
    """Return clean sentences, keep hashtags, and handle NaN/None inputs safely."""
    if text is None or pd.isna(text):
        return []
    text_str = str(text)
    if not text_str.strip():
        return []

    out: List[str] = []
    pos = 0
    for m in _HASHTAG_RE.finditer(text_str):
        pre = text_str[pos:m.start()]
        if pre.strip():
            out.extend(_sent_split_core(pre))
        out.append(m.group())
        pos = m.end()
    tail = text_str[pos:]
    if tail.strip():
        out.extend(_sent_split_core(tail))

    return [s.strip() for s in out if s.strip() and not _PUNCT_ONLY_RE.match(s)]

def sentence_tokenize_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if {"shortcode", "caption"}.difference(df_raw.columns):
        raise ValueError("Columns 'shortcode' and 'caption' not found.")
    df = df_raw.rename(columns={"shortcode": "ID", "caption": "Context"})
    rows, sid = [], 1
    for _, row in df.iterrows():
        for sent in split_sentences(row.Context):
            rows.append({"ID": row.ID, "Context": row.Context, "Statement": sent, "Sentence ID": sid})
            sid += 1
    return pd.DataFrame(rows)

def add_rolling(df_sent: pd.DataFrame) -> pd.DataFrame:
    df = df_sent.sort_values("Sentence ID").copy()
    roll, prev_id, prev_stmt = [], None, None
    for _, r in df.iterrows():
        roll.append(f"{prev_stmt} {r.Statement}" if r.ID == prev_id else r.Statement)
        prev_id, prev_stmt = r.ID, r.Statement
    df["Rolling_Context"] = roll
    return df