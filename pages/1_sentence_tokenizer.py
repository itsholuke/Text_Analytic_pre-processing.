# pages/1_sentence_tokenizer.py
# --------------------------------------------------------------------
# Sentence Tokenizer  –  Professor Step 2
# --------------------------------------------------------------------
# • Accepts ANY CSV
# • User maps Post-ID and Caption/Text columns
# • Outputs:  ID, Context, Statement, Sentence ID
# • Download name: <original>_tokenised.csv
# --------------------------------------------------------------------
import re
from pathlib import Path
import pandas as pd
import streamlit as st

# ── robust CSV reader ─────────────────────────────────────────────────
def safe_read_csv(file_obj) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all tried encodings failed", b"", 0, 0, "")

# ── fallback sentence tokenizer (if NLTK absent) ─────────────────────
try:
    from nltk.tokenize import sent_tokenize
except ModuleNotFoundError:
    def sent_tokenize(text: str):
        return [
            s.strip() for s in
            re.split(r"(?<=[.!?])\s+", str(text).strip())
            if s.strip()
        ]
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sentence Tokenizer", page_icon="✂️")
st.title("✂️ Sentence Tokenizer")

upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

try:
    df = safe_read_csv(upload)
except UnicodeDecodeError as e:
    st.error(f"Unable to decode CSV: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

base_name = Path(upload.name).stem
out_name  = f"{base_name}_tokenised.csv"

id_col   = st.selectbox("🆔 Post-ID column", df.columns)
text_col = st.selectbox("💬 Caption/Text column", df.columns)

if st.button("🚀 Run Sentence Tokenizer"):
    rows = []
    for _, row in df.iterrows():
        caption = str(row[text_col])
        for sid, sent in enumerate(sent_tokenize(caption), 1):
            rows.append({
                "ID":          row[id_col],
                "Context":     caption,
                "Statement":   sent,
                "Sentence ID": sid,
            })
    tok = pd.DataFrame(rows)

    st.success(f"Created {len(tok):,} sentence rows.")
    st.dataframe(tok.head(), use_container_width=True)

    st.download_button(
        label=f"📥 Download {out_name}",
        data=tok.to_csv(index=False).encode(),
        file_name=out_name,
        mime="text/csv",
    )
