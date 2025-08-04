# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, no‑sklearn build)
#  ----------------------------------------------------------
#  Same features as before BUT without scikit‑learn.
#  • Computes precision / recall / F1 manually to avoid missing‑package errors.
# ───────────────────────────────────────────────────────────
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ──────────────── Streamlit page setup ─────────────────────
st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Sentence‑level Classifier with Selectable Ground Truth (no‑sklearn)")

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

def precision_recall_f1(gt_series, pred_series):
    TP = int(((gt_series) & (pred_series)).sum())
    FP = int((~gt_series & pred_series).sum())
    FN = int((gt_series & ~pred_series).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1   = 2*prec*rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

@st.cache_data(show_spinner=False)
def load_csv(file):
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            file.seek(0)
    st.error("Could not decode CSV. Save the file with UTF‑8 or Latin‑1 encoding.")
    st.stop()

# ─────────── STEP 0 – choose tactic ────────────────────────
tactic = st.sidebar.selectbox("🎯 Tactic", list(DEFAULT_TACTICS.keys()))

# ─────────── STEP 1 – upload files ─────────────────────────
st.header("Step 1 — Upload data files")
col1, col2 = st.columns(2)
with col1:
    token_file = st.file_uploader("📄 Tokenised CSV (sentence‑level)", type="csv", key="token")
with col2:
    gt_file = st.file_uploader("📄 Original captions CSV (ground truth & engagement)", type="csv", key="gt")

if not token_file or not gt_file:
    st.info("Please upload both files.")
    st.stop()

sent_df = load_csv(token_file)
gt_df   = load_csv(gt_file)

if "ID" not in sent_df.columns:
    st.error("Tokenised file must contain an 'ID' column.")
    st.stop()

# ─────────── choose columns ────────────────────────────────
st.header("Step 2 — Select columns")

text_col = st.selectbox(
    "Sentence text column",
    [c for c in sent_df.columns if sent_df[c].dtype == object],
    index=sent_df.columns.get_loc("Statement") if "Statement" in sent_df.columns else 0,
)

obj_cols = [c for c in gt_df.columns if gt_df[c].dtype == object]
if "mode_researcher" in obj_cols:
    default_gt_index = obj_cols.index("mode_researcher")
else:
    default_gt_index = 0

gt_col = st.selectbox("Ground‑truth column", obj_cols, index=default_gt_index)

likes_col = st.selectbox("Likes column (optional)", ["None"] + gt_df.columns.tolist(), index=(gt_df.columns.tolist().index("likes") + 1) if "likes" in gt_df.columns else 0)
comments_col = st.selectbox("Comments column (optional)", ["None"] + gt_df.columns.tolist(), index=(gt_df.columns.tolist().index("comments") + 1) if "comments" in gt_df.columns else 0)

# ─────────── STEP 3 – merge ────────────────────────────────
st.header("Step 3 — Merge on ID")

if not st.button("🔗 Merge files"):
    st.stop()

present_cols = ["ID", gt_col]
if likes_col != "None":
    present_cols.append(likes_col)
if comments_col != "None":
    present_cols.append(comments_col)

missing_cols = {"ID", gt_col} - set(gt_df.columns)
if missing_cols:
    st.error(f"Ground‑truth file missing columns: {missing_cols}")
    st.stop()

gt_sub = gt_df[present_cols].copy()
merged_df = sent_df.merge(gt_sub, on="ID", how="left", indicator=False)

st.dataframe(merged_df.head())

# ─────────── STEP 4 – dictionary ───────────────────────────
st.header("Step 4 — Build / refine dictionary")

if st.button("🧠 Auto‑generate keywords"):
    tmp = merged_df.copy()
    tmp["cleaned"] = tmp[text_col].apply(clean)
    base_terms = set(DEFAULT_TACTICS[tactic])
    hit = tmp[tmp["cleaned"].apply(lambda x: any(tok in x.split() for tok in base_terms))]
    freq = hit["cleaned"].str.split(expand=True).stack().value_counts()
    contextual = [w for w in freq.index if w not in base_terms][:30]
    auto_dict = {tactic: sorted(base_terms.union(contextual))}
else:
    auto_dict = {tactic: DEFAULT_TACTICS[tactic]}

dict_text = st.text_area("✏️ Dictionary", value=str(auto_dict), height=150)
try:
    user_dict = ast.literal_eval(dict_text)
    dict_ok = True
except Exception:
    st.error("Invalid dictionary syntax.")
    dict_ok = False

# ─────────── STEP 5 – classify sentences ───────────────────
st.header("Step 5 — Classify & score")

if st.button("🔹 Run classification", disabled=not dict_ok):
    df = merged_df.copy()
    df["cleaned"] = df[text_col].apply(clean)
    df["categories"] = df["cleaned"].apply(lambda x: classify(x, user_dict))
    df["pred_flag"] = df["categories"].apply(lambda lst: tactic in lst)
    df["gt_flag"] = df[gt_col].apply(lambda x: safe_bool(x, tactic))

    s_prec, s_rec, s_f1 = precision_recall_f1(df["gt_flag"], df["pred_flag"])

    # aggregate to post level
    agg = {"pred_flag": "max", "gt_flag": "max"}
    if likes_col != "None":
        agg[likes_col] = "first"
    if comments_col != "None":
        agg[comments_col] = "first"
    post_df = df.groupby("ID").agg(**agg).reset_index()
    p_prec, p_rec, p_f1 = precision_recall_f1(post_df["gt_flag"], post_df["pred_flag"])

    st.subheader("Sentence‑level metrics")
    st.write(f"Precision **{s_prec:.3f}**   Recall **{s_rec:.3f}**   F1 **{s_f1:.3f}**")

    st.subheader("Post‑level metrics")
    st.write(f"Precision **{p_prec:.3f}**   Recall **{p_rec:.3f}**   F1 **{p_f1:.3f}**")

    if likes_col != "None" and comments_col != "None":
        st.subheader("Correlation with engagement")
        st.dataframe(post_df[["pred_flag", "gt_flag", likes_col, comments_col]].corr().round(3))

    st.download_button("Download sentence‑level.csv", df.to_csv(index=False).encode(), "sentence_level_results.csv", mime="text/csv")
    st.download_button("Download post‑level.csv", post_df.to_csv(index=False).encode(), "post_level_results.csv", mime="text/csv")
