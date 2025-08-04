# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, dual‑file GT build)
#  ----------------------------------------------------------
#  • Accept *two* inputs:
#       1. Tokenised/rolling‑context CSV (sentences)
#       2. Original caption‑level CSV with `mode_researcher`, likes, comments
#  • Join them on `ID` to restore the ground‑truth
#  • Build / edit tactic‑aware dictionary
#  • Classify sentences and aggregate back to post‑level
#  • Compute precision, recall, F1 at both sentence‑ and post‑level
#  • Correlate post‑level flags with likes / comments
#  • Download results
# ───────────────────────────────────────────────────────────
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ──────────────── Streamlit page setup ─────────────────────
st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Sentence‑level Classifier with Post‑level Ground Truth")

# ────────────────── built‑in dictionaries ──────────────────
DEFAULT_TACTICS = {
    "urgency_marketing": ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof": ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance","heritage","sophistication","refined","timeless","grace","legacy","opulence","bespoke","tailored","understated","prestige","quality","craftsmanship","heirloom","classic","tradition","iconic","enduring","rich","authentic","luxury","fine","pure","exclusive","elite","mastery","immaculate","flawless","distinction","noble","chic","serene","clean","minimal","poised","balanced","eternal","neutral","subtle","grand","timelessness","tasteful","quiet","sublime",
    ],
}

# ───────────────────────── helpers ─────────────────────────

def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())


def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]


def safe_bool(x, tac):
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", tac.lower()}
    return False

# ─────────── STEP 0 – choose tactic ────────────────────────
st.sidebar.header("Setup")
tactic = st.sidebar.selectbox("🎯 Tactic", list(DEFAULT_TACTICS.keys()))

# ───────────────────────── session init ────────────────────
STATE_DEFAULTS = {
    "token_df": pd.DataFrame(),
    "gt_df": pd.DataFrame(),
    "merged_df": pd.DataFrame(),
    "dictionary": {},
    "dict_ready": False,
}
for k, v in STATE_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ───────── STEP 1 – upload files ───────────────────────────
st.header("Step 1 — Upload data files")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    """Try utf‑8, then latin‑1 as fallback to avoid UnicodeDecodeError."""
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            uploaded_file.seek(0)  # reset pointer for next attempt
    st.error("Cannot decode CSV with utf‑8 or latin‑1. Please save the file with a standard encoding.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    token_file = st.file_uploader("📄 Tokenised CSV (sentence‑level)", type="csv", key="token")
with col2:
    gt_file = st.file_uploader("📄 Original captions CSV (with mode_researcher, likes, comments)", type="csv", key="gt")

if token_file:
    st.session_state.token_df = load_csv(token_file)
    if "ID" not in st.session_state.token_df.columns:
        st.error("Tokenised file must contain an 'ID' column.")
        st.stop()
    st.success(f"Tokenised file loaded — {len(st.session_state.token_df)} rows")

if gt_file:
    st.session_state.gt_df = load_csv(gt_file)
    if "mode_researcher" not in st.session_state.gt_df.columns:
        st.error("Ground‑truth file must contain 'mode_researcher'.")
        st.stop()
    st.success(f"Ground‑truth file loaded — {len(st.session_state.gt_df)} rows")

if st.session_state.token_df.empty or st.session_state.gt_df.empty:
    st.stop()

# ───────── STEP 2 – merge & inspect ──────────────────────── ────────────────────────
st.header("Step 2 — Merge sentence‑ and post‑level data")

if st.button("🔗 Merge on ID"):
    df = st.session_state.token_df.copy()
    gt = st.session_state.gt_df[["ID", "mode_researcher", "likes", "comments"]].copy()
    st.session_state.merged_df = df.merge(gt, on="ID", how="left", indicator=True)

    missing = st.session_state.merged_df["_merge"].eq("left_only").sum()
    if missing:
        st.warning(f"{missing} sentences had no matching ground‑truth.")
    st.session_state.merged_df.drop(columns="_merge", inplace=True)

if st.session_state.merged_df.empty:
    st.stop()

st.dataframe(st.session_state.merged_df.head())

text_col = st.selectbox("Text column", [c for c in st.session_state.merged_df.columns if st.session_state.merged_df[c].dtype == object], index=st.session_state.merged_df.columns.get_loc("Statement") if "Statement" in st.session_state.merged_df.columns else 0)

# ───────── STEP 3 – dictionary build ───────────────────────
st.header("Step 3 — Build / refine dictionary")

if st.button("🧠 Auto‑generate keywords"):
    df = st.session_state.merged_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[tactic])
    df["hit"] = df["cleaned"].apply(lambda x: any(tok in x.split() for tok in base_terms))
    pos_df = df[df["hit"]]

    stop_words = {"the","is","in","on","and","a","for","you","i","are","of","your","to","my","with","it","me","this","that","or"}

    word_freq = pos_df["cleaned"].str.split(expand=True).stack().value_counts()
    contextual_terms = [w for w in word_freq.index if w not in stop_words and w not in base_terms][:30]

    auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

    st.dataframe(word_freq.head(20).rename("Freq"))
    dict_text = st.text_area("✏️ Edit / confirm dictionary", value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_text)
        st.success("Dictionary saved.")
        st.session_state.dict_ready = True
    except Exception:
        st.error("Invalid dict syntax — not saved.")
        st.session_state.dict_ready = False

# ───────── STEP 4 – classify sentences ─────────────────────
st.header("Step 4 — Classify sentences")

if st.button("🔹 Run sentence‑level classification", disabled=not st.session_state.dict_ready):
    df = st.session_state.merged_df.copy()
    df["cleaned"] = df[text_col].apply(clean)
    dct = st.session_state.dictionary

    df["categories"] = df["cleaned"].apply(lambda x: classify(x, dct))
    df["pred_flag"] = df["categories"].apply(lambda lst: tactic in lst)

    # ground‑truth flag (post‑level) propagated to each sentence
    df["gt_flag"] = df["mode_researcher"].apply(lambda x: safe_bool(x, tactic))

    st.session_state.merged_df = df
    st.success("Sentence‑level predictions added.")

if st.session_state.merged_df.empty or "pred_flag" not in st.session_state.merged_df.columns:
    st.stop()

# ───────── STEP 5 – metrics ───────────────────────────────
st.header("Step 5 — Metrics")

from sklearn.metrics import precision_recall_fscore_support

# sentence‑level
sent_prec, sent_rec, sent_f1, _ = precision_recall_fscore_support(
    st.session_state.merged_df["gt_flag"], st.session_state.merged_df["pred_flag"], pos_label=True, average="binary"
)

st.subheader("Sentence‑level")
st.write(
    f"Precision **{sent_prec:.3f}**    Recall **{sent_rec:.3f}**    F1 **{sent_f1:.3f}**"
)

# post‑level aggregation
post_df = (
    st.session_state.merged_df
    .groupby("ID")
    .agg(pred_flag=("pred_flag", "max"), gt_flag=("gt_flag", "max"), likes=("likes", "first"), comments=("comments", "first"))
    .reset_index()
)
post_prec, post_rec, post_f1, _ = precision_recall_fscore_support(post_df["gt_flag"], post_df["pred_flag"], pos_label=True, average="binary")

st.subheader("Post‑level (caption)")
st.write(
    f"Precision **{post_prec:.3f}**    Recall **{post_rec:.3f}**    F1 **{post_f1:.3f}**"
)

# correlation with engagement
if {"likes", "comments"}.issubset(post_df.columns):
    st.subheader("Correlation with engagement")
    st.dataframe(post_df[["pred_flag", "gt_flag", "likes", "comments"]].corr().round(3))

# ───────── STEP 6 – download ───────────────────────────────
st.header("Step 6 — Download")

st.download_button(
    "Download sentence‑level.csv",
    st.session_state.merged_df.to_csv(index=False).encode(),
    file_name="sentence_level_results.csv",
    mime="text/csv",
)

st.download_button(
    "Download post‑level.csv",
    post_df.to_csv(index=False).encode(),
    file_name="post_level_results.csv",
    mime="text/csv",
)
