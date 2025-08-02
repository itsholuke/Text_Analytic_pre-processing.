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

# ── Primary navigation: Streamlit ≥ 1.22 ────────────────────
_nav_ok = True
try:
    st.sidebar.page_link("pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer")
    st.sidebar.page_link("pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder")
except Exception:
    _nav_ok = False

# ── Fallback navigation for older Streamlit builds ──────────
if not _nav_ok:
    tool = st.sidebar.selectbox(
        "Select tool →",
        ["— choose —", "Sentence Tokenizer", "Rolling Context Builder"],
        index=0,
    )
    if tool == "Sentence Tokenizer":
        st.sidebar.info(
            "🔄 Please open the page selector (≡ icon) at the top‑left and choose “1_Instagram_Sentence_Tokenizer.py”."
        )
    elif tool == "Rolling Context Builder":
        st.sidebar.info(
            "🔄 Please open the page selector (≡ icon) at the top‑left and choose “2_Rolling_Context_Builder.py”."
        )
    else:
        st.sidebar.warning(
            "Quick‑links need Streamlit ≥ 1.22. Either upgrade Streamlit *or* open the page selector (≡ icon) at the top‑left to navigate between pages."
        )
```python
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

# Some Streamlit builds below v1.22 lack st.page_link.
try:
    st.sidebar.page_link("pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer")
    st.sidebar.page_link("pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder")
except Exception:
    st.sidebar.warning(
        "Quick‑links need Streamlit ≥ 1.22. Use the page selector (≡ icon) at the top or upgrade Streamlit.")
