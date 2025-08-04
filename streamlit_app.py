import re
import pandas as pd
import streamlit as st

# ---------- sentence tokenizer (NLTK if available, regex fallback) ----------
try:
    from nltk.tokenize import sent_tokenize  # uses Punkt once downloaded
except ModuleNotFoundError:
    def sent_tokenize(text: str):
        """Regex fallback that splits on . ! ? followed by whitespace"""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pre‑processing (Tokenizer + Rolling Context)", page_icon="🧰")
st.title("🧰 Pre‑processing — Sentence Tokenizer + Rolling Context")

st.markdown(
    """
    ### Workflow
    1️⃣ **Upload** any CSV and choose the *Post‑ID* and *Caption/Text* columns.  
    2️⃣ **Sentence Tokenization** – break down the caption into **single‑sentence** rows.  
    &nbsp;&nbsp;&nbsp;&nbsp;⬇ download **`ig_posts_tokenised.csv`**  
    3️⃣ **Rolling Context Window** – *dyadic conversation* context = **previous + current** sentence.  
    &nbsp;&nbsp;&nbsp;&nbsp;⬇ download **`ig_posts_tokenised_with_context.csv`**  
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

# ─────────────────────────────────────────────────────────────
# Column mapping
# ─────────────────────────────────────────────────────────────
cols = raw_df.columns.tolist()

id_col   = st.selectbox("🆔 Column that uniquely identifies each post", cols)
text_col = st.selectbox("💬 Column that contains the caption / text", cols)

# ─────────────────────────────────────────────────────────────
# Step 2 • Sentence Tokenization
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Sentence Tokenizer (Step 2)"):
    records = []
    for _, row in raw_df.iterrows():
        post_id  = row[id_col]
        caption  = str(row[text_col])
        sentences = sent_tokenize(caption)
        for sid, sent in enumerate(sentences, start=1):
            records.append({
                "ID":          post_id,
                "Context":     caption,
                "Statement":   sent,
                "Sentence ID": sid,
            })

    token_df = pd.DataFrame(records)
    st.session_state.token_df = token_df  # store for step 3

    st.success(f"Tokenised {len(raw_df):,} posts into {len(token_df):,} sentences.")
    st.dataframe(token_df.head(10), use_container_width=True)

    st.download_button(
        "📥 Download ig_posts_tokenised.csv",
        token_df.to_csv(index=False).encode(),
        "ig_posts_tokenised.csv",
        "text/csv",
    )

# ─────────────────────────────────────────────────────────────
# Step 3 • Rolling Context Window (dyadic, 2‑sentence)
# ─────────────────────────────────────────────────────────────
if "token_df" in st.session_state:
    st.divider()
    st.header("Step 3 — Add Rolling Context Window (dyadic)")

    if st.button("➕ Build Rolling Context and Download"):
        token_df = st.session_state.token_df.copy()

        # Ensure proper intra‑post ordering
        sort_cols = ["ID"] + (["Sentence ID"] if "Sentence ID" in token_df.columns else [])
        token_df.sort_values(by=sort_cols, inplace=True)

        # previous sentence + space + current sentence
        token_df["Rolling_Context"] = token_df.groupby("ID")["Statement"].transform(
            lambda s: (s.shift(1).fillna("") + " " + s).str.strip()
        )

        st.success("Rolling context column added.")
        st.dataframe(token_df.head(10), use_container_width=True)

        st.download_button(
            "📥 Download ig_posts_tokenised_with_context.csv",
            token_df.to_csv(index=False).encode(),
            "ig_posts_tokenised_with_context.csv",
            "text/csv",
        )
