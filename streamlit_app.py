import streamlit as st
import pandas as pd
from instagram_preprocessing import sentence_tokenize_df, add_rolling

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

# ─────────────────────────────────────────────────────────────
# Primary navigation – Streamlit ≥ 1.22 supports st.page_link
# ─────────────────────────────────────────────────────────────
_nav_ok = True
try:
    st.sidebar.page_link(
        "pages/1_Instagram_Sentence_Tokenizer.py", label="Sentence Tokenizer"
    )
    st.sidebar.page_link(
        "pages/2_Rolling_Context_Builder.py", label="Rolling Context Builder"
    )
except Exception:
    _nav_ok = False

# ─────────────────────────────────────────────────────────────
# Inline fallback for older Streamlit builds (<1.22)
# ─────────────────────────────────────────────────────────────
if not _nav_ok:
    tool = st.sidebar.selectbox(
        "Select tool →",
        ["— choose —", "Sentence Tokenizer", "Rolling Context Builder"],
        index=0,
    )

    if tool == "Sentence Tokenizer":
        st.header("✂️ Sentence Tokenizer (inline)")
        file = st.file_uploader("Upload ig_posts_raw.csv", type=["csv"], key="raw_inline")
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
                st.download_button(
                    "⬇️ Download tokenized CSV",
                    df_tok.to_csv(index=False).encode(),
                    "ig_posts_tokenized.csv",
                    "text/csv",
                )

    elif tool == "Rolling Context Builder":
        st.header("🏗️ Rolling‑Context Builder (inline)")
        file = st.file_uploader(
            "Upload the tokenized CSV", type=["csv"], key="tok_inline"
        )
        if file:
            df_tok = pd.read_csv(file)
            st.info("Building two‑sentence windows…")
            try:
                df_roll = add_rolling(df_tok)
            except ValueError as e:
                st.error(str(e))
            else:
                st.success("Added Rolling_Context!")
                st.dataframe(df_roll.head(20))
                st.download_button(
                    "⬇️ Download with rolling context",
                    df_roll.to_csv(index=False).encode(),
                    "ig_posts_rolling.csv",
                    "text/csv",
                )
    else:
        st.sidebar.warning(
            "Quick‑links need Streamlit ≥ 1.22. Either upgrade Streamlit or pick a tool from the dropdown above."
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
