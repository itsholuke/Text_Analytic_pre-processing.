import streamlit as st

st.set_page_config(page_title="Instagram NLP", page_icon="👔", layout="centered")

st.title("👔 Classic Timeless Luxury – Instagram NLP Suite")
st.markdown(
    """
Welcome! Use the sidebar to launch:

1. **Sentence Tokenizer** – split captions into single sentences.  
2. **Rolling Context Builder** – add two-sentence windows.
"""
)

st.sidebar.header("🛠️ Tools")

# Primary navigation (Streamlit ≥ 1.22)
_nav_ok = True
try:
    st.sidebar.page_link("pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer")
    st.sidebar.page_link("pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder")
except Exception:
    _nav_ok = False

# Fallback for older Streamlit builds
if not _nav_ok:
    tool = st.sidebar.selectbox(
        "Select tool →",
        ["— choose —", "Sentence Tokenizer", "Rolling Context Builder"],
        index=0,
    )
    if tool == "Sentence Tokenizer":
        st.sidebar.info("🔄 Open the page selector (≡ icon) and choose → 1_Instagram_Sentence_Tokenizer.py")
    elif tool == "Rolling Context Builder":
        st.sidebar.info("🔄 Open the page selector (≡ icon) and choose → 2_Rolling_Context_Builder.py")
    else:
        st.sidebar.warning(
            "Quick-links need Streamlit ≥ 1.22. Either upgrade Streamlit or use the ≡ page selector."
        )
