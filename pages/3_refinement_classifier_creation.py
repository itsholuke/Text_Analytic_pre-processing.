# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, GT‑required build)
#  ----------------------------------------------------------
#  • Build / edit tactic‑aware dictionary
#  • Classify text and create 0/1 tactic_flag
#  • Ground‑truth *mandatory* — taken from `mode_researcher` column in the raw CSV
#  • Compute precision, recall, F1
#  • Show correlation of ground‑truth & predictions with likes / comments
#  • Download single CSV with predictions + truth
# ───────────────────────────────────────────────────────────
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ──────────────── Streamlit page setup ─────────────────────
st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing‑Tactic Text Classifier + Metrics — GT mandatory")

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
    """Lower‑case & strip punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())


def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or [
        "uncategorized"
    ]


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def safe_bool(x):
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return False

# ─────────── STEP 0 – choose tactic ────────────────────────
st.header("Step 0 — Choose tactic")
selected_tactic = st.selectbox("🎯 Select marketing tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: *{selected_tactic}*")

# ───────────────────────── session init ────────────────────

defaults = {
    "dict_ready": False,
    "dictionary": {},
    "raw_df": pd.DataFrame(),
    "pred_df": pd.DataFrame(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────── STEP 1 – upload raw CSV (must contain GT) ────
st.header("Step 1 — Upload raw captions CSV (must include *mode_researcher* column)")
raw_file = st.file_uploader("📁 Upload pre‑processed CSV", type="csv")
if raw_file is not None:
    st.session_state.raw_df = pd.read_csv(raw_file)
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0, "ID", st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

if st.session_state.raw_df.empty:
    st.stop()

if "mode_researcher" not in st.session_state.raw_df.columns:
    st.error("Uploaded CSV must include a *mode_researcher* column (ground‑truth label).")
    st.stop()

text_col = st.selectbox("Select text column", st.session_state.raw_df.columns, index=st.session_state.raw_df.columns.get_loc("cleaned") if "cleaned" in st.session_state.raw_df.columns else 0)

# ───────── STEP 2 – generate / refine dictionary ───────────
st.header("Step 2 — Generate / refine dictionary")

if st.button("🧠 Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned_tmp"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[selected_tactic])
    df["row_matches_tactic"] = df["cleaned_tmp"].apply(lambda x: any(tok in x.split() for tok in base_terms))
    pos_df = df[df["row_matches_tactic"]]

    stop_words = {"the","is","in","on","and","a","for","you","i","are","of","your","to","my","with","it","me","this","that","or"}

    if pos_df.empty:
        contextual_terms, contextual_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; using default list only.")
    else:
        word_freq = pos_df["cleaned_tmp"].str.split(expand=True).stack().value_counts()
        contextual_terms = [w for w in word_freq.index if w not in stop_words and w not in base_terms][:30]
        contextual_freq = word_freq.loc[contextual_terms]

    auto_dict = {selected_tactic: sorted(base_terms.union(contextual_terms))}

    st.subheader("Contextual keywords")
    if not contextual_freq.empty:
        st.dataframe(contextual_freq.rename("Freq"))
        fig, ax = plt.subplots()
        contextual_freq.head(15).plot.bar(ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Top contextual words")
        st.pyplot(fig)
    else:
        st.write("— none found —")

    dict_text = st.text_area("✏️ Edit dictionary (Python dict)", value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_text)
        st.success("Dictionary saved.")
    except Exception:
        st.session_state.dictionary = auto_dict
        st.error("Bad format → reverted to auto dictionary.")

    st.session_state.dict_ready = True

# ───────── STEP 3 – run classification ─────────────────────
st.header("Step 3 — Run classification")

if st.button("🔹 Run Classification", disabled=not st.session_state.dict_ready):
    df = st.session_state.raw_df.copy()
    df["cleaned_tmp"] = df[text_col].apply(clean)

    dct = st.session_state.dictionary
    df["categories"] = df["cleaned_tmp"].apply(lambda x: classify(x, dct))
    df["tactic_flag"] = df["categories"].apply(lambda cats: int(selected_tactic in cats))

    # ground‑truth flag
    df["gt_flag"] = df["mode_researcher"].apply(lambda x: safe_bool(x) or (str(x).strip().lower() == selected_tactic.lower()))
    df["true_label"] = df["gt_flag"].apply(lambda x: [selected_tactic] if x else [])

    st.session_state.pred_df = df.copy()
    st.success("Predictions stored.")
    st.dataframe(df.head())

if not st.session_state.pred_df.empty:
    counts = pd.Series([c for cats in st.session_state.pred_df["categories"] for c in cats]).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ───────── STEP 4 – compute metrics ────────────────────────
st.header("Step 4 — Compute metrics & correlations")

if st.button("🔹 Compute Metrics & Correlations", disabled=st.session_state.pred_df.empty):
    df_pred = st.session_state.pred_df.copy()

    df_pred["_gt_list_"] = df_pred["true_label"]
    df_pred["_pred_list_"] = df_pred["categories"]

    rows = []
    for tac in st.session_state.dictionary.keys():
        df_pred["_pred_flag_"] = df_pred["_pred_list_"].apply(lambda lst: tac in lst)
        df_pred["_gt_flag_"] = df_pred["_gt_list_"].apply(lambda lst: tac in lst)

        TP = int((df_pred["_pred_flag_"] & df_pred["_gt_flag_"]).sum())
        FP = int((df_pred["_pred_flag_"] & ~df_pred["_gt_flag_"]).sum())
        FN = int((~df_pred["_pred_flag_"] & df_pred["_gt_flag_"]).sum())

        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        rows.append({"tactic": tac, "TP": TP, "FP": FP, "FN": FN, "precision": prec, "recall": rec, "f1": f1})

    metrics_df = pd.DataFrame(rows).set_index("tactic")
    st.markdown("##### Precision / Recall / F1")
    st.dataframe(metrics_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}))

    # correlations with engagement metrics
    if {"likes", "comments"}.issubset(df_pred.columns):
        st.markdown("##### Correlation with Likes / Comments")
        df_corr = df_pred[["tactic_flag", "gt_flag", "likes", "comments"]].corr()
        st.dataframe(df_corr.round(3))
    else:
        st.info("Columns 'likes' and/or 'comments' not found → skipping correlation.")

    st.session_state.pred_df = df_pred

# ───────── STEP 5 – downloads ──────────────────────────────
if not st.session_state.pred_df.empty:
    st.header("Step 5 — Download results")
    st.download_button(
        label="Download classified_results.csv",
        data=st.session_state.pred_df.to_csv(index=False).encode(),
        file_name="classified_results.csv",
        mime="text/csv",
    )
