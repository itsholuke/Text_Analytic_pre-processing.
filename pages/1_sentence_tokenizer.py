import re, pandas as pd
from pathlib import Path
import streamlit as st

try:
    from nltk.tokenize import sent_tokenize
except ModuleNotFoundError:
    def sent_tokenize(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]

st.set_page_config(page_title="Sentence Tokenizer", page_icon="✂️")
st.title("✂️ Sentence Tokenizer")

upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

df = pd.read_csv(upload)
st.dataframe(df.head(), use_container_width=True)

out_name = f"{Path(upload.name).stem}_tokenised.csv"
cols = df.columns.tolist()
id_col   = st.selectbox("ID column", cols)
text_col = st.selectbox("Caption/Text column", cols)

if st.button("🚀 Run Sentence Tokenizer"):
    rows = []
    for _, r in df.iterrows():
        for sid, sent in enumerate(sent_tokenize(str(r[text_col])), 1):
            rows.append({
                "ID": r[id_col],
                "Context": r[text_col],
                "Statement": sent,
                "Sentence ID": sid,
            })
    tok = pd.DataFrame(rows)
    st.success(f"Created {len(tok):,} rows")
    st.dataframe(tok.head(), use_container_width=True)
    st.download_button(f"Download {out_name}",
                       tok.to_csv(index=False).encode(),
                       out_name, "text/csv")
