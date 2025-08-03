# Classifier Word Metrics – Classic Timeless Luxury Style
# Dependencies: streamlit, pandas, nltk  •  ≈125 LOC

import streamlit as st, pandas as pd, re
from nltk.tokenize import wordpunct_tokenize   # works without Punkt model


def main() -> None:
    # ── Page & style ────────────────────────────────────────────────────────
    st.set_page_config(page_title="Classic Timeless Luxury Metrics",
                       layout="centered")
    st.markdown(
        "<style>h1{font-family:serif;} "
        ".stButton>button{background:#DC143C;color:white;font-weight:600;}</style>",
        unsafe_allow_html=True)
    st.title("Classifier Word Metrics – Classic Timeless Luxury Style")

    # ── STEP 1 • Upload CSV ─────────────────────────────────────────────────
    files = st.file_uploader("Upload a CSV (1 row = 1 sentence)",
                             type="csv", accept_multiple_files=False)

    if not files:
        st.info("↖ Upload a CSV to begin.")
        return

    df = pd.read_csv(files)
    st.subheader("Preview"); st.dataframe(df.head())

    # ── STEP 2 • Select columns ─────────────────────────────────────────────
    id_col   = st.selectbox("ID column (post identifier)",   df.columns, index=0)
    text_col = st.selectbox("Sentence / text column",        df.columns, index=1)

    # ── STEP 3 • Keyword dictionary ────────────────────────────────────────
    default_kw = ("timeless, heritage, vintage, couture, "
                  "iconic, elegant, refined, bespoke, classic")
    kw_input = st.text_input("Classic-timeless-luxury keywords (comma-separated)",
                             value=default_kw)
    keywords = {k.strip().lower() for k in kw_input.split(",") if k.strip()}
    if not keywords:
        st.warning("⚠ Please provide at least one keyword.")
        return

    # ── STEP 4 • Generate metrics ───────────────────────────────────────────
    if not st.button("Generate Metrics", type="primary"):
        return

    def tok(text: str) -> list[str]:
        words = wordpunct_tokenize(str(text).lower())
        return [re.sub(r"\W+", "", w) for w in words if re.sub(r"\W+", "", w)]

    # sentence-level flags & counts
    df["_is_classic"]    = df[text_col].apply(lambda t: any(w in keywords for w in tok(t)))
    df["_classic_words"] = df[text_col].apply(lambda t: sum(w in keywords for w in tok(t)))
    df["_total_words"]   = df[text_col].apply(lambda t: len(tok(t)))

    # post-level aggregation
    agg = (df.groupby(id_col)
             .agg(total_stmts   = ("_is_classic",    "size"),
                  classic_stmts = ("_is_classic",    "sum"),
                  total_words   = ("_total_words",   "sum"),
                  classic_words = ("_classic_words", "sum"))
             .reset_index())

    agg["pct_classic_stmts"] = agg["classic_stmts"] / agg["total_stmts"] * 100
    agg["pct_classic_words"] = agg["classic_words"] / agg["total_words"] * 100

    # ── STEP 5 • Show & download ────────────────────────────────────────────
    st.subheader("Post-level metrics")
    st.dataframe(agg.head())

    st.download_button("Download full CSV",
                       agg.to_csv(index=False).encode(),
                       file_name="classic_timeless_metrics.csv",
                       mime="text/csv")

    # tidy requirements reference
    with st.expander("requirements.txt"):
        st.code("streamlit\npandas\nnltk", language="text")


# so other scripts can import main()
if _name_ == "_main_":
    main()
