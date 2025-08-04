# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, GT‑only build)
#  ----------------------------------------------------------
#  • Two uploads: sentence‑level CSV + caption‑level CSV
#  • User picks any ground‑truth column (default: mode_researcher)
#  • Likes / comments removed — focus purely on classification metrics
# ───────────────────────────────────────────────────────────
import ast
import re

import pandas as pd
import streamlit as st

# ──────────────── Streamlit page setup ─────────────────────
st.set_page_config(page_title="📊 Tactic Classifier", layout="wide")
st.title("📊 Sentence‑level Classifier (Ground Truth required)")

# ────────────────── built‑in dictionaries ──────────────────
DEFAULT_TACTICS = {
    "urgency_marketing": ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof": ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance", "heritage", "sophistication", "refined", "timeless", "grace", "legacy", "opulence", "bespoke", "tailored", "understated", "prestige", "quality", "craftsmanship", "heirloom", "classic", "tradition", "iconic", "enduring", "rich", "authentic", "luxury", "fine", "pure", "exclusive", "elite", "mastery", "immaculate", "flawless", "distinction", "noble", "chic", "serene", "clean", "minimal", "poised", "balanced", "eternal", "neutral", "subtle", "grand", "timelessness", "tasteful", "quiet", "sublime",
    ],
}

# ───────────────────────── helpers ─────────────────────────

def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())


def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or [
        "uncategorized"
    ]


def safe_bool(x, tac):
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", tac.lower()}
    return False


def prec_rec_f1(gt, pred):
    TP = int(((gt) & (pred)).sum())
    FP = int((~gt & pred).sum())
    FN = int((gt & ~pred).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

@st.cache_data(show_spinner=False)
def load_csv(file):
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            file.seek(0)
    st.error("Could not decode CSV. Save with UTF‑8 or Latin‑1 encoding.")
    st.stop()

# ─────────── STEP 0 – choose tactic ────────────────────────
tactic = st.sidebar.selectbox("🎯 Choose tactic", list(DEFAULT_TACTICS.keys()))

# ─────────── STEP 1 – uploads ─────────────────────────────
st.header("Step 1 — Upload files")
col1, col2 = st.columns(2)
with col1:
    sent_file = st.file_uploader("Sentence‑level CSV", type="csv")
with col2:
    cap_file = st.file_uploader("Caption‑level CSV (with ground truth)", type="csv")

if not sent_file or not cap_file:
    st.info("Upload both files to continue.")
    st.stop()

sent_df = load_csv(sent_file)
cap_df  = load_csv(cap_file)

if "ID" not in sent_df.columns or "ID" not in cap_df.columns:
    st.error("Both files must have an 'ID' column.")
    st.stop()

# ─────────── STEP 2 – column selection ────────────────────
st.header("Step 2 — Select columns")

text_col = st.selectbox(
    "Sentence text column",
    [c for c in sent_df.columns if sent_df[c].dtype == object],
    index=sent_df.columns.get_loc("Statement") if "Statement" in sent_df.columns else 0,
)

possible_gt = [c for c in cap_df.columns if c != "ID"]
if "mode_researcher" in possible_gt:
    default_gt_idx = possible_gt.index("mode_researcher")
else:
    default_gt_idx = 0

gt_col = st.selectbox("Ground‑truth column", possible_gt, index=default_gt_idx)

# ─────────── STEP 3 – merge ────────────────────────────────
st.header("Step 3 — Merge on ID")

if not st.button("🔗 Merge & preview"):
    st.stop()

cap_sub = cap_df[["ID", gt_col]].copy()
merged = sent_df.merge(cap_sub, on="ID", how="left")

missing = merged[gt_col].isna().sum()
if missing:
    st.warning(f"{missing} sentences have no ground‑truth label.")

st.dataframe(merged.head())

# ─────────── STEP 4 – dictionary ───────────────────────────
st.header("Step 4 — Dictionary")

base_terms = DEFAULT_TACTICS[tactic]
auto_dict = {tactic: base_terms}

dict_text = st.text_area("Edit dictionary", value=str(auto_dict), height=150)
try:
    user_dict = ast.literal_eval(dict_text)
except Exception:
    st.error("Dictionary syntax error.")
    st.stop()

# ─────────── STEP 5 – classify ─────────────────────────────
st.header("Step 5 — Classify & score")

if not st.button("🔹 Run classification"):
    st.stop()

df = merged.copy()
df["cleaned"] = df[text_col].apply(clean)
df["categories"] = df["cleaned"].apply(lambda x: classify(x, user_dict))
df["pred_flag"] = df["categories"].apply(lambda lst: tactic in lst)
df["gt_flag"] = df[gt_col].apply(lambda x: safe_bool(x, tactic))

s_prec, s_rec, s_f1 = prec_rec_f1(df["gt_flag"], df["pred_flag"])

post_df = df.groupby("ID").agg(pred_flag=("pred_flag", "max"), gt_flag=("gt_flag", "max")).reset_index()
p_prec, p_rec, p_f1 = prec_rec_f1(post_df["gt_flag"], post_df["pred_flag"])

st.subheader("Sentence‑level metrics")
st.write(f"Precision **{s_prec:.3f}**   Recall **{s_rec:.3f}**   F1 **{s_f1:.3f}**")

st.subheader("Post‑level metrics")
st.write(f"Precision **{p_prec:.3f}**   Recall **{p_rec:.3f}**   F1 **{p_f1:.3f}**")

# ─────────── Downloads ─────────────────────────────────────
st.download_button("Download sentence‑level.csv", df.to_csv(index=False).encode(), "sentence_level_results.csv", mime="text/csv")
st.download_button("Download post‑level.csv", post_df.to_csv(index=False).encode(), "post_level_results.csv", mime="text/csv")
