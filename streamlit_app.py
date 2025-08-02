import streamlit as st
st.set_page_config(page_title="Instagram NLP", page_icon="👔", layout="centered")

st.title("👔 Classic Timeless Luxury – Instagram NLP Suite")
st.markdown(
    """
Welcome! Use the sidebar to launch:
1. **Sentence Tokenizer** – split captions into single sentences.
2. **Rolling Context Builder** – add two‑sentence windows.
"""
)

st.sidebar.header("🛠️ Tools")

# 👉 Some Streamlit builds (older than v1.22) don’t support st.page_link.
#    We try it first; if it fails, we fall back to simple instructions so the app
#    doesn’t crash. This prevents the KeyError you just saw.
try:
    st.sidebar.page_link("pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer")
    st.sidebar.page_link("pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder")
except Exception:
    st.sidebar.warning(
        "Quick‑links need Streamlit ≥ 1.22. Use the page selector (≡ icon) above or upgrade Streamlit.")
```python
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
