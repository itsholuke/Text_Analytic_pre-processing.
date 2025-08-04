import pandas as pd
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Rolling Context Builder", page_icon="📜")
st.title("📜 Rolling Context Builder")

upload = st.file_uploader("📁 Upload sentence-level CSV", type="csv")
if upload is None:
    st.stop()

df = pd.read_csv(upload)
st.dataframe(df.head(), use_container_width=True)

out_name = f"{Path(upload.name).stem}_with_context.csv"
cols = df.columns.tolist()
id_col   = st.selectbox("ID column", cols)
stmt_col = st.selectbox("Sentence column", cols)

sort_cols = [id_col] + (["Sentence ID"] if "Sentence ID" in cols else [])
df = df.sort_values(sort_cols).reset_index(drop=True)

if st.button("➕ Add Rolling Context"):
    df["Rolling_Context"] = (
        df.groupby(id_col)[stmt_col]
          .transform(lambda s: (s.shift(1).fillna("") + " " + s).str.strip())
    )
    st.success("Rolling_Context column added.")
    st.dataframe(df.head(), use_container_width=True)
    st.download_button(f"Download {out_name}",
                       df.to_csv(index=False).encode(),
                       out_name, "text/csv")
