# ───────────────────────────────────────────────────────────
#  /pages/3_refinement_classifier_creation.py  (numeric-flag, final consolidated)
#  ----------------------------------------------------------
#  • Single raw-caption CSV ➜ auto dictionary ➜ classify ➜
#    optional ground-truth (upload/manual/CSV col select) ➜ metrics ➜ download
#  • Tactic selector always visible in sidebar
#  • Session flags keep state across reruns
# ───────────────────────────────────────────────────────────
import ast, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
# ─────────── TACTIC SELECTOR ─────────────────────────────
DEFAULT_TACTICS = {
    "urgency_marketing": ["now","today","limited","hurry","exclusive"],
    "social_proof":      ["bestseller","popular","trending","recommended"],
    "discount_marketing": ["sale","discount","deal","free","offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance","heritage","sophistication","refined","timeless","grace",
        "legacy","opulence","bespoke","tailored","understated","prestige",
        "quality","craftsmanship","heirloom","classic","tradition","iconic",
        "enduring","rich","authentic","luxury","fine","pure","exclusive",
        "elite","mastery","immaculate","flawless","distinction","noble","chic",
        "serene","clean","minimal","poised","balanced","eternal","neutral",
        "subtle","grand","timelessness","tasteful","quiet","sublime"
    ],
}
st.title("📊 Marketing-Tactic Text Classifier + Metrics")
tactic = st.sidebar.selectbox("🎯 Choose tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: *{tactic}*")

# ─────────── HELPERS ─────────────────────────────────────
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.startswith("["):
        try: return ast.literal_eval(x)
        except: return []
    return []

def safe_bool(x):
    if isinstance(x, (int, float)): return bool(int(x))
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes"}
    return False

# ─────────── SESSION DEFAULTS & FLAGS ────────────────────
DEFAULT_STATE = {
    "raw_df":     pd.DataFrame(),
    "dictionary": {},
    "pred_df":    pd.DataFrame(),
    "gt_df":      pd.DataFrame(),
    "gt_col":     None,
    "dict_ready": False,
    "pred_ready": False,
    "gt_ready":   False,
}
for key, val in DEFAULT_STATE.items():
    st.session_state.setdefault(key, val)
# keep pred_ready synced
st.session_state['pred_ready'] = not st.session_state['pred_df'].empty

# ─────────── STEP 2 — UPLOAD RAW CSV ──────────────────────
raw_file = st.file_uploader("📁 Step 2 — upload raw CSV", type="csv")
if raw_file:
    df_raw = pd.read_csv(raw_file)
    if "ID" not in df_raw.columns:
        df_raw.insert(0, "ID", df_raw.index.astype(str))
    st.session_state.raw_df = df_raw
    # reset downstream flags
    st.session_state.dict_ready = False
    st.session_state.pred_ready = False
    st.session_state.gt_ready = False
    st.session_state.gt_col   = None
    st.dataframe(df_raw.head(), use_container_width=True)

if st.session_state.raw_df.empty:
    st.info("Upload a raw CSV to begin.")
    st.stop()

# ─────────── STEP 3 — SELECT TEXT COLUMN ─────────────────
text_col = st.selectbox("📋 Step 3 — select text column", st.session_state.raw_df.columns)

# ─────────── STEP 4 — DICTIONARY BUILD/REFINE ─────────────
if st.button("🧠 Step 4 — Generate / refine dictionary"):
    df_temp = st.session_state.raw_df.copy()
    df_temp["cleaned"] = df_temp[text_col].apply(clean)
    seeds = set(DEFAULT_TACTICS[tactic])
    pos = df_temp[df_temp["cleaned"].apply(lambda x: any(tok in x.split() for tok in seeds))]
    stop = {"the","is","in","on","and","a","for","you","i","are","of",
            "your","to","my","with","it","me","this","that","or"}
    if pos.empty:
        ctx_terms, ctx_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seeds; using default list.")
    else:
        freq = pos["cleaned"].str.split(expand=True).stack().value_counts()
        ctx_terms = [w for w in freq.index if w not in stop and w not in seeds][:30]
        ctx_freq  = freq.loc[ctx_terms]
        st.dataframe(ctx_freq.rename("Freq"), use_container_width=True)
    auto_dict = {tactic: sorted(seeds.union(ctx_terms))}
    dict_txt   = st.text_area("✏ Edit dictionary (Python dict)", value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_txt)
        st.session_state.dict_ready = True
        st.success("Dictionary saved.")
    except:
        st.error("Invalid dict syntax — not saved.")
        st.session_state.dict_ready = False

# ─────────── STEP 5-A — CLASSIFICATION ────────────────────
st.subheader("Step 5-A — Classification")
if st.button("🔹 1. Run Classification", disabled=not st.session_state.dict_ready):
    df_pred = st.session_state.raw_df.copy()
    df_pred["cleaned"]    = df_pred[text_col].apply(clean)
    df_pred["categories"] = df_pred["cleaned"].apply(lambda x: classify(x, st.session_state.dictionary))
    df_pred["tactic_flag"] = df_pred["categories"].apply(lambda cats: int(tactic in cats))
    st.session_state.pred_df   = df_pred
    st.session_state.pred_ready = True
    st.session_state.gt_ready   = False
    st.session_state.gt_col     = None
    st.dataframe(df_pred.head(), use_container_width=True)
if st.session_state.pred_ready:
    counts = pd.Series([c for cats in st.session_state.pred_df["categories"] for c in cats]).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ─────────── STEP 5-B — GROUND-TRUTH & METRICS ────────────
# Auto-load ground truth if 'mode_researcher' exists
if 'mode_researcher' in st.session_state.raw_df.columns and not st.session_state.gt_ready:
    st.session_state.gt_df    = st.session_state.raw_df[['ID','mode_researcher']].copy()
    st.session_state.gt_col   = 'mode_researcher'
    st.session_state.gt_ready = True
    st.info("Auto ground-truth loaded from 'mode_researcher' column.")

st.subheader("Step 5-B — Ground-truth & Metrics")
mode = st.radio("Ground-truth source", ["None","Upload CSV","Manual entry"], horizontal=True)
# Reset GT if changing mode
if mode != "Upload CSV": st.session_state.gt_df = pd.DataFrame()
# Upload CSV path
if mode == "Upload CSV":
    gt_file = st.file_uploader("Upload CSV with ID + true_label / flag", type="csv", key="gt_up")
    if gt_file:
        df_gt = pd.read_csv(gt_file)
        st.session_state.gt_df  = df_gt
        cols = [c for c in df_gt.columns if c != "ID"]
        st.session_state.gt_col = st.selectbox("Select GT column", cols,
                                               index=cols.index("mode_researcher") if "mode_researcher" in cols else 0)
        st.session_state.gt_ready = True
        st.success("Ground-truth loaded.")
# Manual entry path
elif mode == "Manual entry" and st.session_state.pred_ready:
    flag = f"{tactic}_flag_gt"
    preview = "_snippet_"
    df_e = st.session_state.pred_df.copy()
    # ensure the flag column exists and is numeric
    if flag not in df_e.columns:
        df_e[flag] = 0
    df_e[flag] = pd.to_numeric(df_e[flag], errors="coerce").fillna(0).astype(int)
    if preview not in df_e.columns:
        df_e[preview] = df_e[text_col].astype(str).str.slice(0, 120)
    edited = st.data_editor(
        df_e[["ID", preview, flag]],
        column_config={
            flag:    st.column_config.NumberColumn(label=f"1 = *{tactic}*   0 = not", min_value=0, max_value=1),
            preview: st.column_config.TextColumn(label="Text (first 120 chars)"),
        },
        height=650, use_container_width=True, key="manual_gt"
    )
    # write back
    st.session_state.pred_df[flag] = pd.to_numeric(edited[flag], errors="coerce").fillna(0).astype(int)
    st.session_state.pred_df["true_label"] = st.session_state.pred_df[flag].apply(lambda x: [tactic] if x else [])
    st.session_state.gt_ready = True
# None: skip

# ─────────── COMPUTE METRICS ─────────────────────────────
if st.button("🔹 2. Compute Metrics", disabled=st.session_state.pred_df.empty):
    dfp = st.session_state.pred_df.copy()
    # merge GT if CSV mode
    if mode == "Upload CSV" and st.session_state.gt_ready:
        gt = st.session_state.gt_df.copy()
        col = st.session_state.gt_col
        dfp = dfp.merge(gt[["ID", col]], on="ID", how="left")
        # numeric flags or categorical labels
        if pd.api.types.is_numeric_dtype(gt[col]) or gt[col].dropna().isin([0,1]).all():
            dfp["true_label"] = dfp[col].apply(lambda x: [tactic] if safe_bool(x) else [])
        else:
            dfp["true_label"] = dfp[col].apply(lambda x: [tactic] if str(x)==tactic else [])
    # ensure truth exists
    if "true_label" not in dfp.columns or dfp["true_label"].isna().all():
        st.warning("No ground-truth labels present → cannot compute metrics.")
    else:
        dfp["gt_list"]   = dfp["true_label"].apply(to_list)
        dfp["pred_list"] = dfp["categories"]
        rows = []
        for tac in st.session_state.dictionary.keys():
            dfp["pred_flag"] = dfp["pred_list"].apply(lambda lst: tac in lst)
            dfp["gt_flag"]   = dfp["gt_list"].apply(lambda lst: tac in lst)
            TP = int((dfp.pred_flag & dfp.gt_flag).sum())
            FP = int((dfp.pred_flag & ~dfp.gt_flag).sum())
            FN = int((~dfp.pred_flag & dfp.gt_flag).sum())
            prec = TP/(TP+FP) if TP+FP else 0.0
            rec  = TP/(TP+FN) if TP+FN else 0.0
            f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
            rows.append({"tactic":tac,"TP":TP,"FP":FP,"FN":FN,
                         "precision":prec,"recall":rec,"f1":f1})
        metrics_df = pd.DataFrame(rows).set_index("tactic")
        st.markdown("##### Precision / Recall / F1")
        st.dataframe(metrics_df.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}))
        st.session_state.pred_df = dfp

# ─────────── DOWNLOAD RESULTS ─────────────────────────────
st.markdown("### 📥 Download results")
st.download_button(
    "Download classified_results.csv",
    st.session_state.pred_df.to_csv(index=False).encode(),
    "classified_results.csv",
    mime="text/csv"
)
