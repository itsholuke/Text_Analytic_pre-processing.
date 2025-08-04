# ───────────────────────────────────────────────────────────
#  streamlit_app.py          (Aug‑2025, numeric‑flag, FIXED)
#  ----------------------------------------------------------
#  • Build / edit tactic‑aware dictionary
#  • Classify text and create 0/1 tactic_flag
#  • Provide ground‑truth via CSV *or* numeric 0/1 column
#  • Compute precision, recall, F1
#  • Download single CSV with predictions + truth
# ───────────────────────────────────────────────────────────
import ast, re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing‑Tactic Text Classifier + Metrics")

# ────────────────── built‑in dictionaries ──────────────────
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        'elegance', 'heritage', 'sophistication', 'refined', 'timeless', 'grace',
        'legacy', 'opulence', 'bespoke', 'tailored', 'understated', 'prestige',
        'quality', 'craftsmanship', 'heirloom', 'classic', 'tradition', 'iconic',
        'enduring', 'rich', 'authentic', 'luxury', 'fine', 'pure', 'exclusive',
        'elite', 'mastery', 'immaculate', 'flawless', 'distinction', 'noble',
        'chic', 'serene', 'clean', 'minimal', 'poised', 'balanced', 'eternal',
        'neutral', 'subtle', 'grand', 'timelessness', 'tasteful', 'quiet', 'sublime'
    ]
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: *{tactic}*")

# ───────────────────────── helpers ─────────────────────────
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try: return ast.literal_eval(x)
        except Exception: return []
    return []

def safe_bool(x):
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes"}
    return False
# ───────────────────────────────────────────────────────────

# initialise session objects
defaults = {
    "dict_ready": False,
    "dictionary": {},
    "top_words":  pd.Series(dtype=int),
    "raw_df":     pd.DataFrame(),  # uploaded raw data
    "pred_df":    pd.DataFrame(),  # predictions stored here
    "gt_df":      pd.DataFrame()   # ground‑truth (if any)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────── STEP 2 – upload raw CSV ──────────────────────
raw_file = st.file_uploader("📁 Step 2 — upload raw CSV", type="csv")
if raw_file:
    st.session_state.raw_df = pd.read_csv(raw_file)
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0, "ID", st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

# need raw_df for everything else
if st.session_state.raw_df.empty:
    st.stop()

text_col = st.selectbox("📋 Step 3 — select text column",
                        st.session_state.raw_df.columns)

# ─────────── STEP 4 – generate / refine dictionary ─────────
if st.button("🧠 Step 4 — Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[tactic])
    df["row_matches_tactic"] = df["cleaned"].apply(
        lambda x: any(tok in x.split() for tok in base_terms)
    )
    pos_df = df[df["row_matches_tactic"]]

    stop_words = {'the','is','in','on','and','a','for','you','i','are','of',
                  'your','to','my','with','it','me','this','that','or'}

    if pos_df.empty:
        contextual_terms, contextual_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; using default list only.")
    else:
        word_freq = (pos_df["cleaned"]
                     .str.split(expand=True)
                     .stack()
                     .value_counts())
        contextual_terms = [w for w in word_freq.index
                            if w not in stop_words and w not in base_terms][:30]
        contextual_freq  = word_freq.loc[contextual_terms]

    auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

    st.subheader("Contextual keywords")
    if not contextual_freq.empty:
        st.dataframe(contextual_freq.rename("Freq"))
    else:
        st.write("‑‑ none found ‑‑")

    dict_text = st.text_area("✏ Edit dictionary (Python dict syntax)",
                             value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_text)
        st.success("Dictionary saved.")
    except Exception:
        st.session_state.dictionary = auto_dict
        st.error("Bad format → reverted to auto dictionary.")

    st.session_state.top_words  = contextual_freq
    st.session_state.dict_ready = True

# ───────── STEP 5‑A – RUN CLASSIFICATION (predictions) ─────
st.subheader("Step 5‑A — Classification")
if st.button("🔹 1. Run Classification",
             disabled=not st.session_state.dict_ready):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    dct = st.session_state.dictionary
    df["categories"]  = df["cleaned"].apply(lambda x: classify(x, dct))
    df["tactic_flag"] = df["categories"].apply(lambda cats: int(tactic in cats))

    st.session_state.pred_df = df.copy()
    st.success("Predictions generated and stored.")
    st.dataframe(df.head())

# category frequency display
if not st.session_state.pred_df.empty:
    counts = pd.Series(
        [c for cats in st.session_state.pred_df["categories"] for c in cats]
    ).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ───────── STEP 5‑B – GROUND‑TRUTH & METRICS ───────────────
st.subheader("Step 5‑B — Ground‑truth & Metrics (optional)")

gt_source = st.radio(
    "Ground‑truth source",
    ["None", "Upload CSV", "Manual entry"],
    horizontal=True,
    index=0
)

# reset uploaded gt when not in upload mode
if gt_source != "Upload CSV":
    st.session_state.gt_df = pd.DataFrame()

# ---------- option 1: upload csv ----------
if gt_source == "Upload CSV":
    gt_file = st.file_uploader(
        "Upload CSV with ID + true_label *or* ID + <tactic>_flag",
        type="csv",
        key="gt_upload"
    )
    if gt_file:
        st.session_state.gt_df = pd.read_csv(gt_file)
        st.success("Ground‑truth file loaded.")

# ---------- option 2: manual numeric flags ----------
elif gt_source == "Manual entry":
    if st.session_state.pred_df.empty:
        st.info("Run classification first, then you can label rows here.")
    else:
        flag_col = f"{tactic}_flag_gt"
        preview  = "_snippet_"

        df_edit = st.session_state.pred_df.copy()

        # ensure numeric int64 column with no NaN
        if flag_col not in df_edit.columns:
            df_edit[flag_col] = 0
        df_edit[flag_col] = pd.to_numeric(df_edit[flag_col], errors="coerce").fillna(0).astype("int64")

        # shorter preview column
        if preview not in df_edit.columns:
            df_edit[preview] = df_edit[text_col].astype(str).str.slice(0, 120)

        edited = st.data_editor(
            df_edit[["ID", preview, flag_col]],
            column_config={
                flag_col: st.column_config.NumberColumn(
                    label=f"1 = *{tactic}*   0 = not",
                    min_value=0, max_value=1, step=1
                ),
                preview: st.column_config.TextColumn(label="Text (first 120 chars)")
            },
            height=650,
            use_container_width=True,
            num_rows="dynamic",
            key="manual_numeric_gt"
        )

        # write back
        st.session_state.pred_df[flag_col] = pd.to_numeric(
            edited[flag_col], errors="coerce").fillna(0).astype("int64")
        st.session_state.pred_df["true_label"] = st.session_state.pred_df[flag_col].apply(
            lambda x: [tactic] if x == 1 else []
        )

# ---------- COMPUTE METRICS ----------
if st.button("🔹 2. Compute Metrics",
             disabled=st.session_state.pred_df.empty):

    df_pred = st.session_state.pred_df.copy()

    # merge uploaded gt if present
    if not st.session_state.gt_df.empty:
        gt = st.session_state.gt_df.copy()
        col_flag = f"{tactic}_flag"
        if col_flag in gt.columns:
            gt["true_label"] = gt[col_flag].apply(
                lambda x: [tactic] if safe_bool(x) else [])
        elif "true_label" not in gt.columns:
            st.error("Ground‑truth CSV must have "
                     f"'true_label' or '{col_flag}' column.")
            st.stop()
        df_pred = df_pred.merge(gt[["ID", "true_label"]],
                                on="ID", how="left", suffixes=("","_y"))
        if "true_label_y" in df_pred.columns:
            df_pred["true_label"] = df_pred["true_label_y"].combine_first(df_pred["true_label"])
            df_pred.drop(columns=["true_label_y"], inplace=True)

    # ensure truth exists
    if "true_label" not in df_pred.columns or df_pred["true_label"].isna().all():
        st.warning("No ground‑truth labels present → cannot compute metrics.")
    else:
        df_pred["_gt_list_"]   = df_pred["true_label"].apply(to_list)
        df_pred["_pred_list_"] = df_pred["categories"]

        rows = []
        for tac in st.session_state.dictionary.keys():
            df_pred["_pred_flag"] = df_pred["pred_list_"].apply(lambda lst: tac in lst)
            df_pred["_gt_flag"]   = df_pred["gt_list_"].apply(lambda lst: tac in lst)

            TP = int(((df_pred["_pred_flag"]) & (df_pred["gt_flag_"])).sum())
            FP = int(((df_pred["_pred_flag"]) & (~df_pred["gt_flag_"])).sum())
            FN = int((~df_pred["_pred_flag"] & (df_pred["gt_flag_"])).sum())

            prec = TP / (TP + FP) if TP + FP else 0.0
            rec  = TP / (TP + FN) if TP + FN else 0.0
            f1   = 2*prec*rec / (prec + rec) if prec + rec else 0.0

            rows.append({"tactic": tac, "TP": TP, "FP": FP, "FN": FN,
                         "precision": prec, "recall": rec, "f1": f1})

        metrics_df = pd.DataFrame(rows).set_index("tactic")
        st.markdown("##### Precision / Recall / F1")
        st.dataframe(metrics_df.style.format({"precision":"{:.3f}",
                                              "recall":"{:.3f}",
                                              "f1":"{:.3f}"}))

        st.session_state.pred_df = df_pred

# ───────── DOWNLOADS ───────────────────────────────────────
if not st.session_state.pred_df.empty:
    st.markdown("### 📥 Download results")
    st.download_button("classified_results.csv",
                       st.session_state.pred_df.to_csv(index=False).encode(),
                       "classified_results.csv", "text/csv")
