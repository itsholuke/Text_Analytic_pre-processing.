# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, stable build)
#  ----------------------------------------------------------
#  • Build / edit tactic‑aware dictionary
#  • Classify text and create 0/1 tactic_flag
#  • Provide ground‑truth via CSV *or* numeric/manual labels
#  • Compute precision, recall, F1
#  • Download single CSV with predictions + truth
# ───────────────────────────────────────────────────────────
import ast
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ──────────────── Streamlit page setup ─────────────────────
st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing‑Tactic Text Classifier + Metrics")

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


def safe_bool(x, tac: str):
    """Treat 1/True/yes **or** tactic‑string itself as True."""
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", tac.lower()}
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
    "gt_df": pd.DataFrame(),
    "gt_flag_col": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────── STEP 1 – upload raw CSV ──────────────────────
st.header("Step 1 — Upload raw captions CSV")
raw_file = st.file_uploader("📁 Upload raw CSV", type="csv")
if raw_file is not None:
    st.session_state.raw_df = pd.read_csv(raw_file)
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0, "ID", st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

if st.session_state.raw_df.empty:
    st.stop()

text_col = st.selectbox("Select text column", st.session_state.raw_df.columns)

# ───────── STEP 2 – generate / refine dictionary ───────────
st.header("Step 2 — Generate / refine dictionary")

if st.button("🧠 Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[selected_tactic])
    df["row_matches_tactic"] = df["cleaned"].apply(lambda x: any(tok in x.split() for tok in base_terms))
    pos_df = df[df["row_matches_tactic"]]

    stop_words = {"the","is","in","on","and","a","for","you","i","are","of","your","to","my","with","it","me","this","that","or"}

    if pos_df.empty:
        contextual_terms, contextual_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; using default list only.")
    else:
        word_freq = pos_df["cleaned"].str.split(expand=True).stack().value_counts()
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
    df["cleaned"] = df[text_col].apply(clean)

    dct = st.session_state.dictionary
    df["categories"] = df["cleaned"].apply(lambda x: classify(x, dct))
    df["tactic_flag"] = df["categories"].apply(lambda cats: int(selected_tactic in cats))

    st.session_state.pred_df = df.copy()
    st.success("Predictions stored.")
    st.dataframe(df.head())

if not st.session_state.pred_df.empty:
    counts = pd.Series([c for cats in st.session_state.pred_df["categories"] for c in cats]).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ───────── STEP 4 – ground‑truth input ─────────────────────
st.header("Step 4 — Provide ground‑truth (optional)")

mode = st.radio("Ground‑truth source", ["None", "Upload CSV", "Manual entry"], horizontal=True)

if mode != "Upload CSV":
    st.session_state.gt_df = pd.DataFrame()
    st.session_state.gt_flag_col = ""

if mode == "Upload CSV":
    gt_file = st.file_uploader("Upload ground‑truth CSV", type="csv", key="gt_upload")
    if gt_file is not None:
        st.session_state.gt_df = pd.read_csv(gt_file)
        st.success("Ground‑truth file loaded.")
        cols = st.session_state.gt_df.columns.tolist()
        preselect = st.session_state.get("gt_flag_col") or next((c for c in cols if c.endswith("_flag") or c.lower().startswith("mode")), cols[0])
        st.session_state.gt_flag_col = st.selectbox("Select ground‑truth column", cols, index=cols.index(preselect))

elif mode == "Manual entry":
    if st.session_state.pred_df.empty:
        st.info("Run classification first, then label rows here.")
    else:
        flag_col = f"{selected_tactic}_flag_gt"
        preview = "_snippet_"
        df_edit = st.session_state.pred_df.copy()
        if flag_col not in df_edit.columns:
            df_edit[flag_col] = 0
        df_edit[flag_col] = pd.to_numeric(df_edit[flag_col], errors="coerce").fillna(0).astype("int64")
        if preview not in df_edit.columns:
            df_edit[preview] = df_edit[text_col].astype(str).str.slice(0, 120)
        edited = st.data_editor(
            df_edit[["ID", preview, flag_col]],
            column_config={
                flag_col: st.column_config.NumberColumn(label=f"1 = *{selected_tactic}*   0 = not", min_value=0, max_value=1, step=1),
                preview: st.column_config.TextColumn(label="Text (first 120 chars)"),
            },
            height=650,
            use_container_width=True,
        )
        st.session_state.pred_df[flag_col] = pd.to_numeric(edited[flag_col], errors="coerce").fillna(0).astype("int64")
        st.session_state.pred_df["true_label"] = st.session_state.pred_df[flag_col].apply(lambda x: [selected_tactic] if x == 1 else [])

# ───────── STEP 5 – compute metrics ────────────────────────
st.header("Step 5 — Compute metrics")

if st.button("🔹 Compute Metrics", disabled=st.session_state.pred_df.empty):
    df_pred = st.session_state.pred_df.copy()

    # merge uploaded ground‑truth if present
    if not st.session_state.gt_df.empty and st.session_state.gt_flag_col:
        gt = st.session_state.gt_df.copy()
        col = st.session_state.gt_flag_col
        if col in gt.columns:
            # numeric / boolean or text label
            if gt[col].apply(lambda x: isinstance(x, (int, float)) or safe_bool(x, selected_tactic)).all():
                gt["true_label"] = gt[col].apply(lambda x: [selected_tactic] if safe_bool(x, selected_tactic) else [])
            else:
                gt["true_label"] = gt[col].apply(lambda x: [selected_tactic] if str(x).strip().lower() == selected_tactic.lower() else [])
        df_pred = df_pred.merge(gt[["ID", "true_label"]], on="ID", how="left", suffixes=("", "_y"))
        if "true_label_y" in df_pred.columns:
            df_pred["true_label"] = df_pred["true_label_y"].combine_first(df_pred["true_label"])
            df_pred.drop(columns=["true_label_y"], inplace=True)

    if "true_label" not in df_pred.columns or df_pred["true_label"].isna().all():
        st.warning("No ground‑truth labels present → cannot compute metrics.")
        st.stop()

    df_pred["_gt_list_"] = df_pred["true_label"].apply(to_list)
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

    st.session_state.pred_df = df_pred

# ───────── STEP 6 – downloads ──────────────────────────────
if not st.session_state.pred_df.empty:
    st.header("Step 6 — Download results")
    st.download_button(
        label="Download classified_results.csv",
        data=st.session_state.pred_df.to_csv(index=False).encode(),
        file_name="classified_results.csv",
        mime="text/csv",
    )
