# pages/1_preprocessing_combined.py
# --------------------------------------------------------------------
# Pre‑processing Utility – Sentence Tokenizer **+** Rolling Context
# --------------------------------------------------------------------
# • Accepts ANY CSV
# • User maps Post‑ID and Caption/Text columns
# • Step 1  ➜ Tokenise captions into sentences (download token file)
# • Step 2  ➜ Append mandatory two‑sentence rolling context window
#             (download final context file)
# --------------------------------------------------------------------
import re
import pandas as pd
import streamlit as st

# ---------- sentence tokenizer (NLTK if available, regex fallback) ----------
try:
    from nltk.tokenize import sent_tokenize  # uses Punkt once downloaded
except ModuleNotFoundError:
    def sent_tokenize(text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pre‑processing (Tokenizer + Context)", page_icon="🧰")
st.title("🧰 Pre‑processing — Sentence Tokenizer + Rolling Context")

st.markdown(
    """
    **Workflow**  
    1️⃣ **Upload** any CSV and choose the *Post‑ID* and *Caption/Text* columns.  
    2️⃣ Click **Tokenise** to split each caption into sentences and preview / download
       **ig_posts_tokenised.csv**.  
    3️⃣ Click **Add Rolling Context** to append a two‑sentence window and download
       **ig_posts_tokenised_with_context.csv**.
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
# 2 • Column mapping
# ─────────────────────────────────────────────────────────────
cols = raw_df.columns.tolist()

id_col   = st.selectbox("🆔 Column that uniquely identifies each post", cols)
text_col = st.selectbox("💬 Column that contains the caption / text", cols)

# ─────────────────────────────────────────────────────────────
# 3 • Run tokeniser
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Tokenise"):
    rows = []
    for _, row in raw_df.iterrows():
        pid      = row[id_col]
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
    st.session_state.token_df = token_df  # store for next step

    st.success(f"Tokenised {len(raw_df):,} posts into {len(token_df):,} sentences.")
    st.dataframe(token_df.head(10), use_container_width=True)

    st.download_button(
        "📥 Download tokenised CSV",
        token_df.to_csv(index=False).encode(),
        "ig_posts_tokenised.csv",
        "text/csv",
    )

# ─────────────────────────────────────────────────────────────
# 4 • Rolling context (always available after tokenisation)
# ─────────────────────────────────────────────────────────────
if hasattr(st.session_state, "token_df"):
    st.divider()
    st.header("Add Two‑Sentence Rolling Context")

    if st.button("➕ Add rolling context"):
        token_df = st.session_state.token_df.copy()

        # Ensure correct ordering within each post if Sentence ID exists
        sort_cols = ["ID"] + (["Sentence ID"] if "Sentence ID" in token_df.columns else [])
        token_df.sort_values(by=sort_cols, inplace=True)

        # Build rolling context aligned with original index
        token_df["Rolling_Context"] = token_df.groupby("ID")["Statement"].transform(
            lambda s: (s.shift(1).fillna("") + " " + s).str.strip()
        )

        st.success("Rolling context column added.")
        st.dataframe(token_df.head(10), use_container_width=True)

        st.download_button(
            "📥 Download CSV with rolling context",
            token_df.to_csv(index=False).encode(),
            "ig_posts_tokenised_with_context.csv",
            "text/csv",
        )
