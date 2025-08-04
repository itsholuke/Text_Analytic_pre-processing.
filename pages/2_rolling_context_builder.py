# pages/2_rolling_context_builder.py
# --------------------------------------------------------------------
# Rolling Context Builder  –  Professor Step 3
# --------------------------------------------------------------------
# • Accepts the *_tokenised.csv file
# • Appends Rolling_Context = previous + current sentence
# • Download name: <tokenised>_with_context.csv
# --------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st

# ── robust CSV reader (same helper) ──────────────────────────────────
def safe_read_csv(file_obj) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all tried encodings failed", b"", 0, 0, "")

# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rolling Context Builder", page_icon="📜")
st.title("📜 Rolling Context Builder – dyadic window (prev + current)")

upload = st.file_uploader("📁 Upload *_tokenised.csv", type="csv")
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
out_name  = f"{base_name}_with_context.csv"

cols = df.columns.tolist()
id_col   = st.selectbox("🆔 Post-ID column", cols)
stmt_col = st.selectbox("💬 Sentence column", cols)

sid_present = "Sentence ID" in cols
if sid_present:
    st.info("Ordering by 'Sentence ID' within each post.")

if st.button("➕ Add Rolling Context"):
    sort_cols = [id_col] + (["Sentence ID"] if sid_present else [])
    work = df.sort_values(sort_cols).reset_index(drop=True).copy()

    work["Rolling_Context"] = (
        work.groupby(id_col)[stmt_col]
            .transform(lambda s: (s.shift(1).fillna("") + " " + s).str.strip())
    )

    st.success("Rolling_Context column added.")
    st.dataframe(work.head(), use_container_width=True)

    st.download_button(
        label=f"📥 Download {out_name}",
        data=work.to_csv(index=False).encode(),
        file_name=out_name,
        mime="text/csv",
    )
