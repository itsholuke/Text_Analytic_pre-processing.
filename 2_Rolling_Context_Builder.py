# --------------------------------------------------------------------
import pandas as pd
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Rolling Context Builder", page_icon="📜")
st.title("📜 Rolling Context Builder – dyadic context (prev + current)")

st.markdown(
    """
    Upload the **sentence‑level CSV** you downloaded from the Sentence Tokenizer
    (e.g., `*_tokenised.csv`). I’ll append a two‑sentence rolling context
    window – the previous sentence plus the current one – and let you download
    the combined file.
    """
)

# ─────────────────────────────────────────────────────────────
# 1 • Upload tokenised CSV
# ─────────────────────────────────────────────────────────────
upload = st.file_uploader("📁 Upload sentence‑level CSV", type="csv")
if upload is None:
    st.stop()

sent_df = pd.read_csv(upload)
st.subheader("Preview of tokenised data")
st.dataframe(sent_df.head(), use_container_width=True)

# Determine output file name
base_name = Path(upload.name).stem  # strip .csv
out_name  = f"{base_name}_with_context.csv"

# ─────────────────────────────────────────────────────────────
# 2 • Column mapping
# ─────────────────────────────────────────────────────────────
cols = sent_df.columns.tolist()

id_col   = st.selectbox("🆔 Column that uniquely identifies each post", cols)
stmt_col = st.selectbox("💬 Column that contains individual sentences", cols)

has_sid  = "Sentence ID" in cols
if has_sid:
    st.info("✅ 'Sentence ID' column detected – rows will be ordered by it within each post.")
else:
    st.warning("⚠️ No 'Sentence ID' column found – rows will be kept in their current order.")

# ─────────────────────────────────────────────────────────────
# 3 • Build rolling context
# ─────────────────────────────────────────────────────────────
if st.button("➕ Add Rolling Context"):
    work = sent_df.copy()

    # sort for deterministic context window
    sort_cols = [id_col]
    if has_sid:
        sort_cols.append("Sentence ID")
    work = work.sort_values(sort_cols).reset_index(drop=True)

    # previous sentence + space + current sentence (strip leading/trailing spaces)
    work["Rolling_Context"] = (
        work.groupby(id_col)[stmt_col]
            .transform(lambda s: (s.shift(1).fillna("") + " " + s).str.strip())
    )

    st.success("Rolling context column added.")
    st.dataframe(work.head(), use_container_width=True)

    st.download_button(
        f"📥 Download {out_name}",
        work.to_csv(index=False).encode(),
        file_name=out_name,
        mime="text/csv",
    )
