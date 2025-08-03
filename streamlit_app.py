# pages/1_sentence_tokenizer.py
# --------------------------------------------------------------------
# Sentence Tokenizer – accepts *any* CSV, lets the user map ID & text
# columns, splits every caption / paragraph into individual sentences,
# and (optionally) builds a two‑sentence rolling context window.
# --------------------------------------------------------------------
import re
import pandas as pd
import streamlit as st

# ---------- sentence tokenizer (with graceful fallback) ----------
try:
    from nltk.tokenize import sent_tokenize  # requires Punkt once per session
except ModuleNotFoundError:
    # Minimal regex fallback – splits on ., ! or ? followed by a space / EOL.
    def sent_tokenize(text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
# -----------------------------------------------------------------

st.set_page_config(page_title="Sentence Tokenizer", page_icon="✂️")
st.title("✂️ Sentence Tokenizer (inline)")

st.markdown("Upload **any** CSV; then tell me which columns are *Post ID* and *Caption / Text*. I’ll split every caption into individual sentences and give you an optional two‑sentence rolling context.")

# ─────────────────────────────────────────────────────────────
# 1 • Upload
# ─────────────────────────────────────────────────────────────
up = st.file_uploader("Upload CSV", type="csv")
if not up:
    st.stop()

orig = pd.read_csv(up)
st.subheader("Preview of uploaded data")
st.dataframe(orig.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2 • Column mapping
# ─────────────────────────────────────────────────────────────
cols = orig.columns.tolist()

id_col   = st.selectbox("Select the column that uniquely identifies the post", cols)
text_col = st.selectbox("Select the column that contains the caption / text", cols)

# Option: include rolling context
roll = st.checkbox("Add two‑sentence rolling context window", value=True)

# ─────────────────────────────────────────────────────────────
# 3 • Tokenize
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Tokenize"):
    rows = []

    for _, row in orig.iterrows():
        post_id = row[id_col]
        caption = str(row[text_col])
        sents   = sent_tokenize(caption)

        for idx, sent in enumerate(sents, start=1):
            if roll:
                start = max(0, idx - 2)  # previous + current sentence
                rolling_ctx = " ".join(sents[start:idx])
            else:
                rolling_ctx = ""

            rows.append({
                "ID":            post_id,
                "Context":       caption,
                "Statement":     sent,
                "Sentence ID":   idx,
                "Rolling_Context": rolling_ctx,
            })

    out = pd.DataFrame(rows)

    st.success(f"Tokenized {len(orig):,} posts into {len(out):,} sentences.")
    st.subheader("Tokenized output (first 10 rows)")
    st.dataframe(out.head(10), use_container_width=True)

    st.download_button(
        "📥 Download tokenized CSV",
        out.to_csv(index=False).encode(),
        "ig_posts_tokenized.csv",
        "text/csv",
    )
