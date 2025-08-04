# pages/3_refinement_classifier_creation.py
# --------------------------------------------------------------------
# Refinement Classifier Creation  (now matplotlib-optional)
# --------------------------------------------------------------------
import streamlit as st
import pandas as pd
import re, ast

# ── optional plotting (skip gracefully if matplotlib is absent) ──────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ---------------------------------------------------------------------

st.set_page_config(page_title="Refinement Classifier Creation", page_icon="🛠️")
st.title("🛠️ Refinement Classifier Creation")

upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None: st.stop()

df = pd.read_csv(upload)
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("Text column:", df.columns)
gt_cols  = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
gt_col   = st.selectbox("Ground-truth 0/1 column (optional):", ["<none>"]+gt_cols)

dict_text = st.text_area("Paste tactic dictionary (Python dict)", height=120)
if not dict_text: st.stop()

try:
    tactic_dict = ast.literal_eval(dict_text)
except Exception as e:
    st.error(f"Dict parse error → {e}")
    st.stop()

def clean(t): return re.sub(r"[^a-zA-Z0-9\\s]","",str(t).lower())
def classify(toks,d): return [k for k,v in d.items() if set(v)&set(toks)] or ["uncategorized"]

if st.button("🚀 Run"):
    df["clean"] = df[text_col].apply(clean)
    df["cats"]  = df["clean"].apply(lambda x: classify(x.split(), tactic_dict))

    counts = pd.Series([c for cats in df["cats"] for c in cats]).value_counts()
    st.subheader("Category frequencies")
    st.dataframe(counts.rename("Posts"))

    # optional bar chart
    if HAS_PLOT and not counts.empty:
        fig, ax = plt.subplots()
        counts.plot.bar(ax=ax); ax.set_ylabel("Posts")
        st.pyplot(fig)
    elif not HAS_PLOT:
        st.info("Matplotlib not installed – skipping bar chart.")

    # precision / recall / F1 if ground-truth supplied
    if gt_col != "<none>":
        y_true = df[gt_col].astype(int)
        target = next(iter(tactic_dict))
        y_pred = df["cats"].apply(lambda cats: int(target in cats))
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall",    f"{rec:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")

    st.download_button("💾 Download classified CSV",
                       df.to_csv(index=False).encode(),
                       "refined_results.csv","text/csv")
