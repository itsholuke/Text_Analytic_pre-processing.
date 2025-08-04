import re
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- sentence tokenizer (NLTK if available, regex fallback) ----------
try:
    from nltk.tokenize import sent_tokenize  # uses Punkt once downloaded
except ModuleNotFoundError:
    def sent_tokenize(text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sentence Tokenizer", page_icon="✂️")
st.title("✂️ Sentence Tokenizer")

st.markdown(
    """
    Upload any CSV, choose the **Post‑ID** and **Caption/Text** columns, and I’ll
    split every caption into individual sentences.  The download will reuse your
    original file name with **_tokenised.csv** appended.
    """
)

# ─────────────────────────────────────────────────────────────
# 1 • Upload raw CSV
# ─────────────────────────────────────────────────────────────
upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

raw_df = pd.read_csv(upload)
st.subheader("Preview of uploaded data")
st.dataframe(raw_df.head(), use_container_width=True)

# Determine output file name based on the uploaded file
base_name = Path(upload.name).stem  # drop .csv extension
out_name  = f"{base_name}_tokenised.csv"

# ─────────────────────────────────────────────────────────────
# 2 • Column mapping
# ─────────────────────────────────────────────────────────────
cols = raw_df.columns.tolist()

id_col   = st.selectbox("🆔 Column that uniquely identifies each post", cols)
text_col = st.selectbox("💬 Column that contains the caption / text", cols)

# ─────────────────────────────────────────────────────────────
# 3 • Tokenise
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Sentence Tokenizer"):
    rows = []
    for _, row in raw_df.iterrows():
        pid       = row[id_col]
        caption   = str(row[text_col])
        sentences = sent_tokenize(caption)
        for sid, sent in enumerate(sentences, start=1):
            rows.append({
                "ID":          pid,
                "Context":     caption,
                "Statement":   sent,
                "Sentence ID": sid,
            })

    token_df = pd.DataFrame(rows)

    st.success(f"Tokenised {len(raw_df):,} posts into {len(token_df):,} sentences.")
    st.dataframe(token_df.head(10), use_container_width=True)

    st.download_button(
        f"📥 Download {out_name}",
        token_df.to_csv(index=False).encode(),
        file_name=out_name,
        mime="text/csv",
    )
