import streamlit as st, pandas as pd
from instagram_preprocessing import sentence_tokenize_df

st.header("✂️ Sentence Tokenizer")
file = st.file_uploader("Upload ig_posts_raw.csv", type=["csv"])
if file:
    df_raw = pd.read_csv(file)
    st.info("Tokenizing…")
    try:
        df_tok = sentence_tokenize_df(df_raw)
    except ValueError as e:
        st.error(str(e))
    else:
        st.success(f"Extracted {len(df_tok):,} sentences.")
        st.dataframe(df_tok.head(20))
        st.download_button("⬇️ Download", df_tok.to_csv(index=False).encode(), "ig_posts_tokenized.csv", "text/csv")