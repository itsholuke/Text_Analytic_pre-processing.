# pages/4_classifier_metrics.py
# ---------------------------------------------------------
# Timeless Luxury Style – post-level keyword-share metrics
# ---------------------------------------------------------
import re
import pandas as pd
import streamlit as st

# ---------- tokenizer (works with or without NLTK) ----------
try:
    from nltk.tokenize import wordpunct_tokenize        # no extra downloads
except ModuleNotFoundError:
    def wordpunct_tokenize(text: str):
        # Same split behaviour as NLTK's wordpunct_tokenize
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
# ------------------------------------------------------------

DEFAULT_KEYWORDS = (
    "timeless, heritage, vintage, couture, iconic, "
    "elegant, refined, bespoke, classic"
)

st.set_page_config(page_title="Timeless Luxury Style", page_icon="💎")
st.title("Timeless Luxury Style")

# ─────────────────────────────────────────────────────────────
# STEP 1 • upload CSV  (1 row = 1 sentence)
# ─────────────────────────────────────────────────────────────
csv = st.file_uploader("Upload a CSV (1 row = 1 sentence)", type="csv")
if not csv:
    st.stop()

df = pd.read_csv(csv)
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# STEP 2 • choose columns
# ─────────────────────────────────────────────────────────────
id_col     = st.selectbox("ID column (post identifier)", df.columns, index=0)
text_col   = st.selectbox("Sentence / text column", df.columns, index=3)

kw_input = st.text_input(
    "Classic·timeless·luxury keywords (comma-separated)",
    value=DEFAULT_KEYWORDS,
)

if not kw_input.strip():
    st.warning("Enter at least one keyword.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# STEP 3 • generate metrics
# ─────────────────────────────────────────────────────────────
if st.button("Generate Metrics"):

    keywords = {k.strip().lower() for k in kw_input.split(",") if k.strip()}
    st.write(f"Using **{len(keywords)}** keywords.")

    # sentence-level flags & counts
    def analyse_sentence(text: str):
        toks = [t.lower() for t in wordpunct_tokenize(str(text))]
        hits = [t for t in toks if t in keywords]
        return len(toks), len(hits), 1 if hits else 0

    sent_metrics = df[text_col].apply(analyse_sentence)
    df[["word_cnt", "classic_word_cnt", "classic_stmt"]] = pd.DataFrame(
        sent_metrics.tolist(), index=df.index
    )

    # post-level aggregation
    agg = (
        df.groupby(id_col)
          .agg(total_stmts      = ("classic_stmt", "count"),
               classic_stmts    = ("classic_stmt", "sum"),
               total_words      = ("word_cnt", "sum"),
               classic_words    = ("classic_word_cnt", "sum"))
          .reset_index()
    )
    agg["pct_classic_stmts"] = (agg["classic_stmts"] / agg["total_stmts"] * 100).round(2)
    agg["pct_classic_words"] = (agg["classic_words"] / agg["total_words"] * 100).round(2)

    st.subheader("Post-level metrics")
    st.dataframe(agg, use_container_width=True)

    # ─── download buttons ───
    st.download_button(
        "Download post-level metrics CSV",
        agg.to_csv(index=False).encode(),
        "post_level_metrics.csv",
        "text/csv",
    )
    st.download_button(
        "Download sentence-level detail CSV",
        df.to_csv(index=False).encode(),
        "sentence_level_metrics.csv",
        "text/csv",
    )
