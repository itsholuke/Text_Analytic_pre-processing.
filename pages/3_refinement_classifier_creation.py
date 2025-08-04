# pages/3_Refinement_Classifier_Creation.py
# --------------------------------------------------------------------
# Refinement Classifier Creation – tactic dictionary + metrics
# --------------------------------------------------------------------
import re, ast, pandas as pd, streamlit as st

# ── optional plotting (skip gracefully if matplotlib missing) ─────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ---------------------------------------------------------------------

st.set_page_config(page_title="Refinement Classifier Creation", page_icon="🛠️")
st.title("🛠️ Refinement Classifier Creation")

# ---------- upload ---------------------------------------------------
upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

df = pd.read_csv(upload)
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("Text column:", df.columns)

# ---------- safe 0/1 column detection --------------------------------
def is_binary(series: pd.Series) -> bool:
    """True if column holds only 0/1 (or 0.0/1.0); False for un-hashable."""
    try:
        vals = set(series.dropna().unique())
        return vals <= {0,1} or vals <= {0.0,1.0}
    except TypeError:
        return False                   # happens for list/dict cells

bin_cols = [c for c in df.columns if is_binary(df[c])]
gt_col   = st.selectbox("Ground-truth 0/1 column (optional):",
                        ["<none>"] + bin_cols)

# ---------- tactic dictionary ----------------------------------------
example = '{"urgency_marketing": ["now", "today", "hurry"]}'
dict_text = st.text_area("Paste tactic dictionary (ONE key)", example, height=140)

try:
    parsed = ast.literal_eval(dict_text)
    if len(parsed) != 1:
        raise ValueError("Provide exactly ONE tactic")
    tactic_name, keywords = next(iter(parsed.items()))
    keywords = set(map(str.lower, keywords))
except Exception as e:
    st.error(f"Dict parse error → {e}")
    st.stop()

# ---------- helper fns -----------------------------------------------
clean = lambda t: re.sub(r"[^a-zA-Z0-9\\s]","",str(t).lower())
classify_flag = lambda toks: int(any(w in toks for w in keywords))
# ---------------------------------------------------------------------

if st.button("🚀 Run"):
    with st.spinner("Classifying…"):
        df["_clean"]   = df[text_col].apply(clean)
        df["pred_flag"] = df["_clean"].apply(lambda x: classify_flag(x.split()))

    freq = df["pred_flag"].value_counts().rename(index={0:"No",1:"Yes"})
    st.subheader("Predicted tactic flag frequency")
    st.dataframe(freq.rename("Rows"))

    if HAS_PLOT and not freq.empty:
        fig, ax = plt.subplots()
        freq.plot.bar(ax=ax)
        ax.set_ylabel("Rows"); ax.set_title("Predicted tactic flag")
        st.pyplot(fig)
    elif not HAS_PLOT:
        st.info("Matplotlib not installed – skipping bar chart.")

    # ---------- metrics ------------------------------------------------
    if gt_col != "<none>":
        y_true = df[gt_col].astype(int)
        y_pred = df["pred_flag"]
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        precision = tp/(tp+fp) if tp+fp else 0
        recall    = tp/(tp+fn) if tp+fn else 0
        f1        = 2*precision*recall/(precision+recall) if precision+recall else 0
        st.subheader("Classification metrics")
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall",    f"{recall:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")

    # ---------- download ----------------------------------------------
    st.download_button("💾 Download classified CSV",
                       df.drop(columns=\"_clean\").to_csv(index=False).encode(),
                       "refined_results.csv", "text/csv")
