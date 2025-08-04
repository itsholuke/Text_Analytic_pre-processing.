# streamlit_app.py  –  Dictionary-Refiner-style all-in-one tool
import re, ast, pandas as pd, streamlit as st

# ── optional plotting (skip if matplotlib missing) ───────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Text Content Analysis Tool", page_icon="🧮", layout="wide")
st.title("🧮 Text Content Analysis Tool")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
ENCODINGS = ("utf-8", "utf-8-sig", "latin1", "cp1252")

def safe_read_csv(f):
    for enc in ENCODINGS:
        try:
            f.seek(0)
            return pd.read_csv(f, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("common encodings failed", b"", 0, 0, "")

def tokenize(text: str):
    return re.findall(r"\w+", str(text).lower())

# default dictionaries
DEFAULT_DICTS = {
    "urgency_marketing": {
        "now","today","hurry","limited","deadline","instant","ending","soon","rush"
    },
    "exclusive_marketing": {
        "exclusive","vip","members","invite","private","insider","elite","only"
    },
    "classic_timeless_luxury": {
        "timeless","heritage","vintage","classic","iconic","elegant","refined",
        "bespoke","luxury","craftsmanship"
    },
}

# ─────────────────────────────────────────────────────────────
# 1 • Upload
# ─────────────────────────────────────────────────────────────
st.header("1. Upload Your Data")
up = st.file_uploader("📁 Upload CSV", type="csv")
if up is None:
    st.stop()

try:
    df = safe_read_csv(up)
except UnicodeDecodeError as e:
    st.error(f"Unable to decode CSV: {e}")
    st.stop()

st.success(f"Loaded file with {len(df):,} rows.")
with st.expander("Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2 • Select text column
# ─────────────────────────────────────────────────────────────
st.header("2. Select Text Column")
text_col = st.selectbox("Column containing text to analyse:", df.columns)

# ─────────────────────────────────────────────────────────────
# 3 • Configure dictionaries
# ─────────────────────────────────────────────────────────────
st.header("3. Configure Analysis Dictionaries")
dict_source = st.radio("Dictionary source",
                       ("Use default dictionaries", "Enter custom dictionaries"),
                       horizontal=True)

if dict_source == "Use default dictionaries":
    dictionaries = {k: set(v) for k, v in DEFAULT_DICTS.items()}
else:
    template = '{"my_tactic": ["word1", "word2", "word3"]}'
    dict_text = st.text_area("Paste Python dict here:", template, height=140)
    try:
        parsed = ast.literal_eval(dict_text)
        dictionaries = {k: set(map(str.lower, v)) for k, v in parsed.items()}
        st.success("Custom dictionaries parsed.")
    except Exception as e:
        st.error(f"Dict parse error → {e}")
        st.stop()

# summary
dict_df = pd.DataFrame(
    {"Category": list(dictionaries),
     "Terms":   [len(v) for v in dictionaries.values()]}
)
with st.expander("Dictionary summary"):
    st.dataframe(dict_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 4 • Run analysis
# ─────────────────────────────────────────────────────────────
st.header("4. Run Analysis")
if st.button("🚀 Start Analysis"):
    with st.spinner("Tokenising & classifying…"):
        tokens = df[text_col].fillna("").apply(tokenize)
        df["categories"] = tokens.apply(
            lambda toks: [k for k,v in dictionaries.items() if set(toks) & v] or ["uncategorized"]
        )

    # category frequency
    cat_counts = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()
    st.subheader("Category frequency")
    st.dataframe(cat_counts.rename("Posts"), use_container_width=True)

    # keyword frequency
    all_tokens = [w for toklist in tokens for w in toklist]
    kw_freq = pd.Series(all_tokens).value_counts()

    st.subheader("Top keywords")
    st.dataframe(kw_freq.head(25).rename("Freq"), use_container_width=True)

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(6,3))
        kw_freq.head(20).plot.bar(ax=ax)
        ax.set_title("Top 20 keywords")
        st.pyplot(fig)
    else:
        st.info("Matplotlib not installed – skipping chart.")

    # sample tagged posts
    st.subheader("Sample tagged posts")
    cat_choice = st.selectbox("Choose a category to view examples:", cat_counts.index)
    examples = df[df["categories"].apply(lambda c: cat_choice in c)][text_col].head(3)
    for idx, txt in examples.items():
        st.text_area(f"Post {idx}", txt, height=80, disabled=True)

    # precision / recall / F1 when exactly one binary GT column present
    bin_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
    if len(bin_cols) == 1:
        gt = bin_cols[0]
        st.subheader("Ground-truth metrics")
        metric_choice = st.selectbox("Select tactic for metrics:", list(dictionaries))
        y_true = df[gt].astype(int)
        y_pred = df["categories"].apply(lambda cats: int(metric_choice in cats))
        tp = int(((y_true==1)&(y_pred==1)).sum()); fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall    = tp/(tp+fn) if tp+fn else 0.0
        f1        = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall",    f"{recall:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")
    else:
        st.info("No single 0/1 ground-truth column found – metrics skipped.")

    # downloads
    st.header("5. Download Results")
    st.download_button("📥 Download full results CSV",
                       df.to_csv(index=False).encode(),
                       "analysis_results.csv","text/csv")
    st.download_button("📥 Download category frequency CSV",
                       cat_counts.to_csv().encode(),
                       "category_frequency.csv","text/csv")
    st.download_button("📥 Download keyword frequency CSV",
                       kw_freq.to_csv().encode(),
                       "keyword_frequency.csv","text/csv")
