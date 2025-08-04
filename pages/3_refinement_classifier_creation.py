# ───────────────────────────────────────────────────────────
#  /pages/3_refinement_classifier_creation.py  (numeric-flag, fixed)
#  ----------------------------------------------------------
#  • Build / edit tactic-aware dictionary
#  • Classify text and create 0/1 tactic_flag
#  • Provide ground-truth via CSV *or* numeric 0/1 column
#  • Compute precision, recall, F1
#  • Download single CSV with predictions + truth
#  • Keeps its step between reruns
# ───────────────────────────────────────────────────────────
import ast, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing-Tactic Text Classifier + Metrics")

# ─────────── seed dictionaries ─────────────────────────────
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance","heritage","sophistication","refined","timeless","grace","legacy","opulence","bespoke","tailored",
        "understated","prestige","quality","craftsmanship","heirloom","classic","tradition","iconic","enduring","rich",
        "authentic","luxury","fine","pure","exclusive","elite","mastery","immaculate","flawless","distinction","noble",
        "chic","serene","clean","minimal","poised","balanced","eternal","neutral","subtle","grand","timelessness",
        "tasteful","quiet","sublime",
    ],
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: *{tactic}*")

# ─────────── helpers ───────────────────────────────────────
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\\s]", "", str(txt).lower())

def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

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
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes"}
    return False

# ─────────── session flags ────────────────────────────────
FLAGS = {
    "dict_ready": False,
    "pred_ready": False,
    "gt_ready":   False,
}
for k, v in FLAGS.items():
    st.session_state.setdefault(k, v)

# ─────────── STEP 2 – upload raw CSV ───────────────────────
raw_file = st.file_uploader("📁 Step 2 — upload raw CSV", type="csv")
if raw_file:
    st.session_state.raw_df = pd.read_csv(raw_file)
    st.session_state.pred_ready = False  # reset later steps
    st.session_state.gt_ready   = False
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0, "ID", st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

if st.session_state.get("raw_df", pd.DataFrame()).empty:
    st.stop()

text_col = st.selectbox("📋 Step 3 — select text column", st.session_state.raw_df.columns)

# ─────────── STEP 4 – generate / refine dictionary ─────────
if st.button("🧠 Step 4 — Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[tactic])
    pos_df = df[df["cleaned"].apply(lambda x: any(tok in x.split() for tok in base_terms))]

    stop_words = {"the","is","in","on","and","a","for","you","i","are","of",
                  "your","to","my","with","it","me","this","that","or"}

    if pos_df.empty:
        contextual_terms, contextual_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; using default list only.")
    else:
        word_freq = pos_df["cleaned"].str.split(expand=True).stack().value_counts()
        contextual_terms = [w for w in word_freq.index if w not in stop_words and w not in base_terms][:30]
        contextual_freq  = word_freq.loc[contextual_terms]

    auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}
    dict_text = st.text_area("✏ Edit dictionary (Python dict syntax)", value=str(auto_dict), height=150)

    try:
        st.session_state.dictionary = ast.literal_eval(dict_text)
        st.session_state.dict_ready = True
        st.success("Dictionary saved.")
    except Exception:
        st.error("Bad format → dictionary not saved.")
        st.session_state.dict_ready = False

# ───────── STEP 5-A – RUN CLASSIFICATION ───────────────────
st.subheader("Step 5-A — Classification")
if st.button("🔹 1. Run Classification", disabled=not st.session_state.dict_ready):
    df = st.session_state.raw_df.copy()
    df["cleaned"]    = df[text_col].apply(clean)
    df["categories"] = df["cleaned"].apply(lambda x: classify(x, st.session_state.dictionary))
    df["tactic_flag"] = df["categories"].apply(lambda cats: int(tactic in cats))

    st.session_state.pred_df   = df
    st.session_state.pred_ready = True
    st.session_state.gt_ready   = False  # reset GT path
    st.success("Predictions stored.")
    st.dataframe(df.head())

# category frequency display
if st.session_state.get("pred_ready"):
    counts = pd.Series([c for cats in st.session_state.pred_df["categories"] for c in cats]).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ───────── STEP 5-B – GROUND-TRUTH & METRICS ───────────────
st.subheader("Step 5-B — Ground-truth & Metrics")

gt_source = st.radio("Ground-truth source", ["None", "Upload CSV", "Manual entry"], horizontal=True, index=0)

if gt_source != "Upload CSV":
    st.session_state.gt_df = pd.DataFrame()

if gt_source == "Upload CSV":
    gt_file = st.file_uploader("Upload CSV with ID + true_label *or* ID + <tactic>_flag",
                               type="csv", key="gt_upload")
    if gt_file:
        st.session_state.gt_df  = pd.read_csv(gt_file)
        st.session_state.gt_ready = True
        st.success("Ground-truth file loaded.")

elif gt_source == "Manual entry" and st.session_state.get("pred_ready"):
    flag_col = f"{tactic}_flag_gt"
    preview  = "_snippet_"
    df_edit = st.session_state.pred_df.copy()

    if flag_col not in df_edit.columns:
        df_edit[flag_col] = 0
    df_edit[flag_col] = pd.to_numeric(df_edit[flag_col], errors="coerce").fillna(0).astype("int64")

    if preview not in df_edit.columns:
        df_edit[preview] = df_edit[text_col].astype(str).str.slice(0, 120)

    edited = st.data_editor(
        df_edit[["ID", preview, flag_col]],
        column_config={
            flag_col: st.column_config.NumberColumn(label=f"1 = *{tactic}*   0 = not", min_value=0, max_value=1),
            preview:  st.column_config.TextColumn(label="Text (first 120 chars)"),
        },
        height=650, use_container_width=True, num_rows="dynamic", key="manual_gt",
    )
    st.session_state.pred_df[flag_col] = pd.to_numeric(edited[flag_col], errors="coerce").fillna(0).astype("int64")
    st.session_state.pred_df["true_label"] = st.session_state.pred_df[flag_col].apply(lambda x: [tactic] if x else [])
    st.session_state.gt_ready = True

# ---------- COMPUTE METRICS ----------
if st.button("🔹 2. Compute Metrics", disabled=not (st.session_state.get('pred_ready') and st.session_state.get('gt_ready'))):
    df_pred = st.session_state.pred_df.copy()

    if not st.session_state.gt_df.empty:     # merge uploaded GT if present
        gt = st.session_state.gt_df.copy()
        col_flag = f"{tactic}_flag"
        if col_flag in gt.columns:
            gt["true_label"] = gt[col_flag].apply(lambda x: [tactic] if safe_bool(x) else [])
        elif "true_label" not in gt.columns:
            st.error("Ground-truth CSV must have 'true_label' or '{col_flag}'.")
            st.stop()
        df_pred = df_pred.merge(gt[["ID", "true_label"]], on="ID", how="left", suffixes=("","_y"))
        if "true_label_y" in df_pred.columns:
            df_pred["true_label"] = df_pred["true_label_y"].combine_first(df_pred["true_label"])
            df_pred.drop(columns=["true_label_y"], inplace=True)

    if df_pred["true_label"].isna().all():
        st.warning("No ground-truth labels present → cannot compute metrics.")
        st.stop()

    df_pred["gt_list"]   = df_pred["true_label"].apply(to_list)
    df_pred["pred_list"] = df_pred["categories"]

    rows = []
    for tac in st.session_state.dictionary.keys():
        df_pred["pred_flag"] = df_pred["pred_list"].apply(lambda lst: tac in lst)
        df_pred["gt_flag"]   = df_pred["gt_list"].apply(lambda lst: tac in lst)

        TP = int((df_pred.pred_flag & df_pred.gt_flag).sum())
        FP = int((df_pred.pred_flag & ~df_pred.gt_flag).sum())
        FN = int((~df_pred.pred_flag & df_pred.gt_flag).sum())

        prec = TP / (TP + FP) if TP + FP else 0.0
        rec  = TP / (TP + FN) if TP + FN else 0.0
        f1   = 2*prec*rec / (prec + rec) if prec + rec else 0.0

        rows.append(dict(tactic=tac, TP=TP, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1))

    metrics_df = pd.DataFrame(rows).set_index("tactic")
    st.markdown("##### Precision / Recall / F1")
    st.dataframe(metrics_df.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}))

    st.session_state.pred_df = df_pred  # store merged + labels

# ───────── DOWNLOADS ───────────────────────────────────────
if st.session_state.get("pred_ready"):
    st.markdown("### 📥 Download results")
    st.download_button("classified_results.csv",
                       st.session_state.pred_df.to_csv(index=False).encode(),
                       "classified_results.csv", mime="text/csv")
