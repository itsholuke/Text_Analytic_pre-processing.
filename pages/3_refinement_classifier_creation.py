# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, state‑safe build)
#  ----------------------------------------------------------
#  Keeps state between reruns so the app doesn’t jump back a step.
#  • Two uploads: sentence‑level CSV + caption‑level CSV
#  • User picks any GT column (default: mode_researcher)
#  • Dictionary editing preserved in session
#  • Merge / classify buttons set flags in session_state
#  • Likes/comments removed
# ───────────────────────────────────────────────────────────
import ast
import re
import pandas as pd
import streamlit as st

# ──────────── page config ─────────────────────────────────
st.set_page_config(page_title="📊 Tactic Classifier", layout="wide")
st.title("📊 Sentence‑level Classifier (Ground Truth required)")

# ─────────── default dictionaries ─────────────────────────
DEFAULT_TACTICS = {
    "urgency_marketing": ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof": ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance", "heritage", "sophistication", "refined", "timeless", "grace", "legacy", "opulence", "bespoke", "tailored", "understated", "prestige", "quality", "craftsmanship", "heirloom", "classic", "tradition", "iconic", "enduring", "rich", "authentic", "luxury", "fine", "pure", "exclusive", "elite", "mastery", "immaculate", "flawless", "distinction", "noble", "chic", "serene", "clean", "minimal", "poised", "balanced", "eternal", "neutral", "subtle", "grand", "timelessness", "tasteful", "quiet", "sublime",
    ],
}

# ─────────── helpers ──────────────────────────────────────

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
    st.error("Could not decode CSV.")
    st.stop()

# ─────────── session defaults ─────────────────────────────
SS_DEF = {
    "sent_df": pd.DataFrame(),
    "cap_df": pd.DataFrame(),
    "merged_df": pd.DataFrame(),
    "user_dict": {},
    "merge_ready": False,
    "dict_ready": False,
    "results_ready": False,
}
for k, v in SS_DEF.items():
    st.session_state.setdefault(k, v)

# ─────────── choose tactic ────────────────────────────────
tactic = st.sidebar.selectbox("🎯 Choose tactic", list(DEFAULT_TACTICS.keys()))

# ─────────── STEP 1 – uploads ─────────────────────────────
st.header("Step 1 — Upload files")
col1, col2 = st.columns(2)
with col1:
    sent_file = st.file_uploader("Sentence‑level CSV", type="csv", key="sent_upload")
with col2:
    cap_file = st.file_uploader("Caption‑level CSV", type="csv", key="cap_upload")

if sent_file:
    st.session_state.sent_df = load_csv(sent_file)
if cap_file:
    st.session_state.cap_df = load_csv(cap_file)

if st.session_state.sent_df.empty or st.session_state.cap_df.empty:
    st.stop()

# ─────────── STEP 2 – column selection ────────────────────
st.header("Step 2 — Select columns")

text_col = st.selectbox(
    "Sentence text column",
    [c for c in st.session_state.sent_df.columns if st.session_state.sent_df[c].dtype == object],
    index=st.session_state.sent_df.columns.get_loc("Statement") if "Statement" in st.session_state.sent_df.columns else 0,
)

possible_gt = [c for c in st.session_state.cap_df.columns if c != "ID"]
if not possible_gt:
    st.error("Caption file has no candidate GT columns.")
    st.stop()

gt_col = st.selectbox("Ground‑truth column", possible_gt, index=possible_gt.index("mode_researcher") if "mode_researcher" in possible_gt else 0)

# ─────────── STEP 3 – merge ───────────────────────────────
st.header("Step 3 — Merge on ID")

if st.button("🔗 Merge & preview"):
    cap_sub = st.session_state.cap_df[["ID", gt_col]].copy()
    merged = st.session_state.sent_df.merge(cap_sub, on="ID", how="left")
    st.session_state.merged_df = merged
    st.session_state.merge_ready = True

if not st.session_state.merge_ready:
    st.stop()

st.dataframe(st.session_state.merged_df.head())

# ─────────── STEP 4 – dictionary ──────────────────────────
st.header("Step 4 — Dictionary")

dict_text = st.text_area(
    "Edit dictionary", value=str(st.session_state.user_dict or {tactic: DEFAULT_TACTICS[tactic]}), height=150
)

if st.button("💾 Save dictionary"):
    try:
        st.session_state.user_dict = ast.literal_eval(dict_text)
        st.session_state.dict_ready = True
        st.success("Dictionary saved.")
    except Exception:
        st.error("Invalid dict.")
        st.session_state.dict_ready = False

if not st.session_state.dict_ready:
    st.stop()

# ─────────── STEP 5 – classify ────────────────────────────
st.header("Step 5 — Classify & score")

if st.button("🔹 Run classification"):
    df = st.session_state.merged_df.copy()
    df["cleaned"] = df[text_col].apply(clean)
    df["categories"] = df["cleaned"].apply(lambda x: classify(x, st.session_state.user_dict))
    df["pred_flag"] = df["categories"].apply(lambda lst: tactic in lst)
    df["gt_flag"] = df[gt_col].apply(lambda x: safe_bool(x, tactic))

    s_prec, s_rec, s_f1 = prec_rec_f1(df["gt_flag"], df["pred_flag"])

    post_df = (
        df.groupby("ID").agg(pred_flag=("pred_flag", "max"), gt_flag=("gt_flag", "max")).reset_index()
    )
    p_prec, p_rec, p_f1 = prec_rec_f1(post_df["gt_flag"], post_df["pred_flag"])

    st.session_state.results_ready = True
    st.session_state.sentence_results = df
    st.session_state.post_results = post_df
    st.session_state.metrics = {
        "s_prec": s_prec,
        "s_rec": s_rec,
        "s_f1": s_f1,
        "p_prec": p_prec,
        "p_rec": p_rec,
        "p_f1": p_f1,
    }

if not st.session_state.results_ready:
    st.stop()

m = st.session_state.metrics
st.subheader("Sentence‑level metrics")
st.write(f"Precision **{m['s_prec']:.3f}**   Recall **{m['s_rec']:.3f}**   F1 **{m['s_f1']:.3f}**")

st.subheader("Post‑level metrics")
st.write(f"Precision **{m['p_prec']:.3f}**   Recall **{m['p_rec']:.3f}**   F1 **{m['p_f1']:.3f}**")

st.download_button("Download sentence‑level.csv", st.session_state.sentence_results.to_csv(index=False).encode(), "sentence_level_results.csv", mime="text/csv")

st.download_button("Download post‑level.csv", st.session_state.post_results.to_csv(index=False).encode(), "post_level_results.csv", mime="text/csv")
