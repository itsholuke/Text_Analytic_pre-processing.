# ───────────────────────────────────────────────────────────
#  /pages/3_refinement_classifier_creation.py  (numeric-flag, gated flow)
#  ----------------------------------------------------------
#  • Stepwise UI: hide previous steps after completion (dict, classify, GT)
#  • Choose tactic → upload CSV → refine dict → classify → GT → metrics → download
# ───────────────────────────────────────────────────────────
import ast, re
import pandas as pd
import streamlit as st

# ─────────── HELPERS ────────────────────────────────────
def to_list(x):
    """Ensure values are lists: parse literal or return list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []


st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
# ─────────── TACTIC SELECTION (Step 1) ────────────────────
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
tactic = st.sidebar.selectbox("🎯 Choose tactic", [None] + list(DEFAULT_TACTICS.keys()), format_func=lambda x: "Select a tactic" if x is None else x)
if tactic is None:
    st.warning("Please select a tactic to begin.")
    st.stop()

# ─────────── SESSION STATE & FLAGS ──────────────────────
def init_state():
    defaults = {
        "raw_df": pd.DataFrame(),
        "dictionary": {},
        "pred_df": pd.DataFrame(),
        "gt_df": pd.DataFrame(),
        "gt_col": None,
        "dict_ready": False,
        "pred_ready": False,
        "gt_ready": False,
        "metrics_ready": False
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
init_state()

# Sync pred_ready if pred_df exists
st.session_state.pred_ready = not st.session_state.pred_df.empty

# ─────────── UPLOAD RAW CSV (Step 2) ─────────────────────
if not st.session_state.raw_df.any().any():
    raw = st.file_uploader("📁 Step 2 — upload raw CSV", type="csv")
    if raw:
        df_raw = pd.read_csv(raw)
        if "ID" not in df_raw.columns:
            df_raw.insert(0, "ID", df_raw.index.astype(str))
        st.session_state.raw_df = df_raw
        st.session_state.dict_ready = False
        st.session_state.pred_ready = False
        st.session_state.gt_ready = False
        st.session_state.metrics_ready = False
        st.session_state.gt_col = None
        st.dataframe(df_raw.head(), use_container_width=True)
    else:
        st.stop()

# ─────────── SELECT TEXT COLUMN (Step 3) ─────────────────
text_col = st.selectbox("📋 Step 3 — select text column", st.session_state.raw_df.columns)

# ─────────── DICTIONARY BUILD (Step 4) ───────────────────
if not st.session_state.dict_ready:
    if st.button("🧠 Step 4 — Generate / refine dictionary"):
        df_temp = st.session_state.raw_df.copy()
        df_temp["cleaned"] = df_temp[text_col].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]","",str(x).lower()))
        seeds = set(DEFAULT_TACTICS[tactic])
        pos = df_temp[df_temp["cleaned"].apply(lambda c: any(tok in c.split() for tok in seeds))]
        stop = {"the","is","in","on","and","a","for","you","i","are","of","your","to","my","with","it","me","this","that","or"}
        if not pos.empty:
            freq = pos["cleaned"].str.split(expand=True).stack().value_counts()
            ctx = [w for w in freq.index if w not in stop and w not in seeds][:30]
            st.dataframe(freq.loc[ctx].rename("Freq"), use_container_width=True)
        auto_dict = {tactic: sorted(seeds.union(ctx if 'ctx' in locals() else []))}
        dtxt = st.text_area("✏ Edit dictionary", value=str(auto_dict), height=150)
        try:
            st.session_state.dictionary = ast.literal_eval(dtxt)
            st.session_state.dict_ready = True
            st.success("Dictionary saved.")
        except:
            st.error("Invalid dict syntax.")
    else:
        st.info("Click 'Generate / refine dictionary' to continue.")
        st.stop()

# ─────────── CLASSIFICATION (Step 5A) ────────────────────
if st.session_state.dict_ready and not st.session_state.pred_ready:
    st.subheader("Step 5‑A — Classification")
    if st.button("🔹 Run Classification"):
        dfp = st.session_state.raw_df.copy()
        dfp["cleaned"] = dfp[text_col].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]","",str(x).lower()))
        dfp["categories"] = dfp["cleaned"].apply(lambda c: [cat for cat,terms in st.session_state.dictionary.items() if any(w in c.split() for w in terms)] or ["uncategorized"])
        dfp["tactic_flag"] = dfp["categories"].apply(lambda lst: int(tactic in lst))
        st.session_state.pred_df = dfp
        st.session_state.pred_ready = True
        st.success("Classification done.")
        st.dataframe(dfp.head(), use_container_width=True)
    else:
        st.info("Click 'Run Classification' to get predictions.")
        st.stop()

# Show frequencies after classification
if st.session_state.pred_ready:
    st.markdown("##### Category frequencies")
    st.table(pd.Series([c for cats in st.session_state.pred_df.categories for c in cats]).value_counts())

# ─────────── GROUND-TRUTH (Step 5B) ─────────────────────
if st.session_state.pred_ready and not st.session_state.gt_ready:
    st.subheader("Step 5‑B — Ground-truth")
    mode = st.radio("GT source", ["None","Upload CSV","Manual entry"], horizontal=True)
    if mode == "None":
        st.warning("No ground-truth → metrics will be zeros.")
        st.session_state.gt_ready = True
    elif mode == "Upload CSV":
        gtf = st.file_uploader("Upload GT CSV", type="csv", key="gtup")
        if gtf:
            gtd = pd.read_csv(gtf)
            st.session_state.gt_df = gtd
            cols = [c for c in gtd.columns if c!="ID"]
            st.session_state.gt_col = st.selectbox("Select GT column", cols, index=cols.index("mode_researcher") if "mode_researcher" in cols else 0)
            st.session_state.gt_ready = True
            st.success("GT loaded.")
        else:
            st.info("Upload a GT CSV or choose 'None' to proceed.")
            st.stop()
    else:  # Manual entry
        flag = f"{tactic}_flag_gt"
        dfm = st.session_state.pred_df.copy()
        dfm[flag] = dfm.get(flag,0)
        dfm[flag] = pd.to_numeric(dfm[flag],errors="coerce").fillna(0).astype(int)
        dfm["_snippet_"] = dfm[text_col].astype(str).str.slice(0,120)
        edited = st.data_editor(dfm[["ID","_snippet_",flag]], use_container_width=True, height=600)
        st.session_state.pred_df[flag] = pd.to_numeric(edited[flag],errors="coerce").fillna(0).astype(int)
        st.session_state.pred_df["true_label"] = st.session_state.pred_df[flag].apply(lambda x: [tactic] if x else [])
        st.session_state.gt_ready = True

# ─────────── METRICS (Step 6) ────────────────────────────
if st.session_state.pred_ready and st.session_state.gt_ready:
    st.subheader("Step 6 — Precision / Recall / F1")
    dfp = st.session_state.pred_df.copy()
    if "true_label" not in dfp:
        dfp["true_label"] = dfp["tactic_flag"].apply(lambda x: [tactic] if x else [])
    dfp["gt_list"] = dfp["true_label"].apply(to_list)
    dfp["pred_list"] = dfp["categories"]
    rows=[]
    for tac in st.session_state.dictionary:
        dfp["pred_flag"] = dfp["pred_list"].apply(lambda lst: tac in lst)
        dfp["gt_flag"]   = dfp["gt_list"].apply(lambda lst: tac in lst)
        TP = int((dfp.pred_flag & dfp.gt_flag).sum())
        FP = int((dfp.pred_flag & ~dfp.gt_flag).sum())
        FN = int((~dfp.pred_flag & dfp.gt_flag).sum())
        prec = TP/(TP+FP) if TP+FP else 0
        rec  = TP/(TP+FN) if TP+FN else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        rows.append({"tactic":tac,"TP":TP,"FP":FP,"FN":FN,"precision":prec,"recall":rec,"f1":f1})
    mdf = pd.DataFrame(rows).set_index("tactic")
    st.dataframe(mdf.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}), use_container_width=True)
    st.success("Metrics computed.")

    # ─────────── DOWNLOAD (Step 7) ─────────────────────────
    st.markdown("### 📥 Download results")
    st.download_button("Download CSV", dfp.to_csv(index=False).encode(), "class_results.csv", mime="text/csv")
