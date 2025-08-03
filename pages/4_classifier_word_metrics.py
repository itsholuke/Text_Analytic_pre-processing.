# ─────────────────────────────────────────────────────────────
# pages/4_classifier_word_metrics.py
# Robust – runs with or without NLTK in the environment
# ─────────────────────────────────────────────────────────────
import re
import pandas as pd
import streamlit as st

# ---------- tokenizer (NLTK-aware fallback) ----------
try:                                # use NLTK if available
    from nltk.tokenize import wordpunct_tokenize   # no extra models needed
except ModuleNotFoundError:         # else minimal drop-in replacement
    def wordpunct_tokenize(text: str):
        # Splits words and punctuation just like NLTK's version
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
# -----------------------------------------------------

st.set_page_config(page_title="Classifier Word Metrics", page_icon="🧮")
st.title("🧮 Classifier Word Metrics")

csv = st.file_uploader("Upload classified_results.csv", type="csv")
if not csv:
    st.info("Upload a CSV created with the main classifier app.")
    st.stop()

df = pd.read_csv(csv)
text_col = st.selectbox("Text column", df.columns, index=0)

# ─── choose metric source ──────────────────────────────
pct_cols = [c for c in df.columns if c.startswith("%_")]
mode = st.radio("Metric source", ("Existing %_ columns", "Ad-hoc keywords"))

if mode == "Existing %_ columns":
    if not pct_cols:
        st.error("No %_ columns found. Re-run the main app first.")
        st.stop()

    col = st.selectbox("Pick %_ column", pct_cols)
    st.subheader("Descriptive stats")
    st.dataframe(df[[col]].describe(), use_container_width=True)

else:  # ad-hoc keyword metric
    kws = st.text_input("Comma-separated keywords (case-insensitive)")
    keywords = {k.strip().lower() for k in kws.split(",") if k.strip()}

    if keywords:
        def pct_hits(text: str) -> float:
            toks = [t.lower() for t in wordpunct_tokenize(str(text))]
            return sum(t in keywords for t in toks) / max(1, len(toks))

        df["pct_custom"] = df[text_col].apply(pct_hits)

        st.subheader("Descriptive stats")
        st.dataframe(df[["pct_custom"]].describe(), use_container_width=True)

        st.download_button("Download pct_custom.csv",
                           df[["pct_custom"]].to_csv(index=False).encode(),
                           "pct_custom.csv",
                           "text/csv")
