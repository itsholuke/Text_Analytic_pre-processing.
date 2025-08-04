# pages/1_preprocessing.py
# --------------------------------------------------------------------
# Pre‑processing (Sentence Tokenizer + Rolling Context)
# --------------------------------------------------------------------
import re
import pandas as pd
import streamlit as st

# ---------- sentence tokenizer (NLTK if present, regex fallback) ----------
try:
    from nltk.tokenize import sent_tokenize  # requires Punkt once downloaded
except ModuleNotFoundError:
    def sent_tokenize(text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
# -------------------------------------------------------------------------

st.set_page_config(page_title="Pre-processing (Tokenizer & Rolling Context)", page_icon="✂️")
st.title("✂️ Pre-processing (Sentence Tokenizer + Rolling Context)")

st.markdown(
    """
    **Step 1.** Upload any CSV and map the *Post‑ID* and *Caption/Text* columns.  \ 
    **Step 2.** I’ll split each caption into individual sentences and let you
    download the tokenised file.\

    **Step 3 (optional).** Click **Add Rolling Context** to create a two‑sentence
    rolling window column and download the extended file.
    """
)

# ─────────────────────────────────────────────────────────────
# 1 • Upload
# ─────────────────────────────────────────────────────────────
upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

raw_df = pd.read_csv(upload)
st.subheader("Preview of uploaded data")
st.dataframe(raw_df.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2 • Column mapping
# ─────────────────────────────────────────────────────────────
cols = raw_df.columns.tolist()

id_col   = st.selectbox("🆔 Column that uniquely identifies each post", cols)
text_col = st.selectbox("📝 Column that contains the full caption / text", cols)

# ─────────────────────────────────────────────────────────────
# 3 • Tokenise
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Pre‑processing"):
    sent_rows = []
    for _, row in raw_df.iterrows():
        post_id = row[id_col]
        caption = str(row[text_col])
        sentences = sent_tokenize(caption)
        for idx, sent in enumerate(sentences, start=1):
            sent_rows.append({
                "ID":          post_id,
                "Context":     caption,
                "Statement":   sent,
                "Sentence ID": idx,
            })

    token_df = pd.DataFrame(sent_rows)
    st.session_state["token_df"] = token_df  # store for later

    st.success(f"Tokenised {len(raw_df):,} posts into {len(token_df):,} sentences.")
    st.subheader("Tokenised output (first 10 rows)")
    st.dataframe(token_df.head(10), use_container_width=True)

    st.download_button(
        "📥 Download tokenised CSV",
        token_df.to_csv(index=False).encode(),
        "ig_posts_tokenised.csv",
        "text/csv",
    )

# ─────────────────────────────────────────────────────────────
# 4 • Optional Rolling Context
# ─────────────────────────────────────────────────────────────
if st.session_state.get("token_df") is not None:
    st.divider()
    st.header("Optional: Add Rolling Context")
    if st.button("➕ Add two‑sentence rolling context and download"):
        token_df = st.session_state["token_df"].copy()
        token_df["Rolling_Context"] = token_df.groupby("ID")["Statement"].apply(
            lambda s: s.shift(1).fillna("") + " " + s
        )
        token_df["Rolling_Context"] = token_df["Rolling_Context"].str.strip()

        st.success("Rolling context column added.")
        st.dataframe(token_df.head(10), use_container_width=True)

        st.download_button(
            "📥 Download tokenised + context CSV",
            token_df.to_csv(index=False).encode(),
            "ig_posts_tokenised_with_context.csv",
            "text/csv",
        )
