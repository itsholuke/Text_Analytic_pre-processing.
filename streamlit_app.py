import streamlit as st
st.set_page_config(page_title="Instagram NLP", page_icon="👔", layout="centered")

st.title("👔 Classic Timeless Luxury – Instagram NLP Suite")
st.markdown("""
Welcome! Use the sidebar to launch:
1. **Sentence Tokenizer** – split captions into single sentences.
2. **Rolling Context Builder** – add two‑sentence windows.
""")

st.sidebar.header("🛠️ Tools")
st.sidebar.page_link("pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer")
st.sidebar.page_link("pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder")