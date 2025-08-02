import streamlit as st, pandas as pd
from instagram_preprocessing import add_rolling

st.header("🏗️ Rolling‑Context Builder")
file = st.file_uploader("Upload tokenized CSV", type=["csv"])
if file:
    df_tok = pd.read_csv(file)
    st.info("Building windows…")
    try:
        df_roll = add_rolling(df_tok)
    except ValueError as e:
        st.error(str(e))
    else:
        st.success("Done!")
        st.dataframe(df_roll.head(20))
        st.download_button("⬇️ Download", df_roll.to_csv(index=False).encode(), "ig_posts_rolling.csv", "text/csv")