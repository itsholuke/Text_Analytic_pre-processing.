# streamlit_app.py
# -------------------------------------------------------------
# Text Content Analysis Tool
# End‑to‑end workflow: upload → pick text column → configure
#   dictionaries → run analysis → view metrics + precision/recall/F1
# -------------------------------------------------------------
import re
import ast
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# OPTIONAL PLOTTING (auto‑skip if matplotlib missing)
# ─────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clean(txt: str) -> str:
    """Lower‑case & keep only letters/digits/spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(txt)).lower().strip()


def classify_text(tokens: list[str], dictionaries: dict[str, set[str]]):
    """Return a list of categories whose keyword sets intersect tokens."""
    hit_cats = [n for n, kws in dictionaries.items() if kws & set(tokens)]
    return hit_cats or ["uncategorized"]


# ─────────────────────────────────────────────────────────────
# DEFAULT DICTIONARIES (can be overwritten / extended)
# ─────────────────────────────────────────────────────────────
DEFAULT_DICTIONARIES = {
    "urgency_marketing": {
        "now", "today", "hurry", "limited", "last", "final",
        "act", "instant", "immediately", "deadline", "ending",
        "soon", "rush", "while", "stock", "running", "gone",
    },
    "exclusive_marketing": {
        "exclusive", "members", "vip", "private", "invite", "selected",
        "access", "insider", "limited access", "by invitation",
        "privileged", "special", "only", "elite", "premier",
    },
    "classic_timeless_luxury": {
        "timeless", "heritage", "vintage", "classic", "iconic",
        "elegant", "refined", "bespoke", "couture",
    },
}


# ─────────────────────────────────────────────────────────────
# SESSION STATE KEYS INITIALISATION
# ─────────────────────────────────────────────────────────────
for key in (
    "df", "clean_tokens", "dictionaries", "analysis_done",
    "category_freq", "keyword_freq", "y_true", "y_pred",
):
    st.session_state.setdefault(key, None)

# ─────────────────────────────────────────────────────────────
# 1.  UPLOAD YOUR DATA
# ─────────────────────────────────────────────────────────────
st.title("📊 Text Content Analysis Tool")
st.caption("Analyze text content using customizable dictionaries to identify patterns and categories.")

st.header("1. Upload Your Data")
file = st.file_uploader(
    "Upload CSV file containing text data", type="csv",
    help="CSV must include at least one text column",
)
if file:
    df = pd.read_csv(file)
    st.success(f"File uploaded successfully! Found {len(df):,} rows.")
    st.session_state.df = df
else:
    st.stop()

# ─────────────────────────────────────────────────────────────
# 2.  SELECT TEXT COLUMN
# ─────────────────────────────────────────────────────────────
st.header("Select Text Column")
text_col = st.selectbox("Choose the column containing text to analyze:", df.columns)

with st.expander("🔍 Data Preview"):
    st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 3.  CONFIGURE ANALYSIS DICTIONARIES
# ─────────────────────────────────────────────────────────────
st.header("2. Configure Analysis Dictionaries")

dict_source = st.radio(
    "Dictionary Input",
    ("Use Default Dictionaries", "Enter Custom Dictionaries"),
    horizontal=True,
)

if dict_source == "Use Default Dictionaries":
    dictionaries = DEFAULT_DICTIONARIES.copy()
else:
    st.markdown("Enter your custom dictionaries in **valid Python dict** format:")
    ex = '{"urgency_marketing": ["now", "hurry"],\n "exclusive_marketing": ["exclusive", "vip"]}'
    user_dict_text = st.text_area("Paste your dictionaries here:", value=ex, height=150)
    try:
        parsed = ast.literal_eval(user_dict_text)
        dictionaries = {k: set(map(str.lower, v)) for k, v in parsed.items()}
        st.success("Custom dictionaries parsed successfully!")
    except Exception as e:
        st.error(f"❌ Error parsing dictionaries: {e}")
        st.stop()

# Show dictionary summary (terms per category)
with st.container():
    st.subheader("Dictionary Summary")
    summary_df = pd.DataFrame(
        {"Category": list(dictionaries.keys()),
         "Terms": [len(v) for v in dictionaries.values()]}
    )
    st.table(summary_df.set_index("Category"))

# ─────────────────────────────────────────────────────────────
# 4.  RUN ANALYSIS
# ─────────────────────────────────────────────────────────────
st.header("3. Run Analysis")
if st.button("🚀 Start Analysis"):

    # Tokenise text column + store tokens for reuse
    clean_tokens = df[text_col].fillna("").apply(lambda x: re.findall(r"\w+", x.lower()))
    st.session_state.clean_tokens = clean_tokens

    # Classification per sentence / row
    categories = clean_tokens.apply(lambda toks: classify_text(toks, dictionaries))
    df["categories"] = categories

    # Flatten to get overall keyword frequency
    all_tokens = [tok for toks in clean_tokens for tok in toks]
    keyword_freq = pd.Series(all_tokens, name="keyword").value_counts()

    # Category frequency (# of posts containing category)
    cat_flat = [c for cats in categories for c in cats]
    category_freq = pd.Series(cat_flat, name="category").value_counts()

    st.session_state.keyword_freq  = keyword_freq
    st.session_state.category_freq = category_freq

    # Ground‑truth handling (optional)
    gt_cols = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
    y_true = y_pred = None
    if gt_cols:
        gt_col = st.selectbox("Optional ground‑truth 0/1 column for metrics:", gt_cols)
        y_true = df[gt_col].astype(int)

        # For a multi‑label scenario we consider positive if the selected
        # ground‑truth tactic name appears somewhere in the dictionaries keys.
        # Match by column name prefix if possible.
        target_cat = None
        for cat in dictionaries:
            if cat in gt_col or gt_col in cat:
                target_cat = cat
                break
        if target_cat is None:
            target_cat = list(dictionaries)[0]  # fallback first category

        y_pred = df["categories"].apply(lambda cats: int(target_cat in cats))
        st.session_state.y_true = y_true
        st.session_state.y_pred = y_pred
    else:
        st.info("No 0/1 ground‑truth columns detected – precision/recall/F1 will be skipped.")

    st.session_state.analysis_done = True
    st.success("Analysis completed successfully!")

# ─────────────────────────────────────────────────────────────
# 5.  ANALYSIS RESULTS
# ─────────────────────────────────────────────────────────────
if st.session_state.analysis_done:

    st.header("4. Analysis Results")

    # Category Analysis
    st.subheader("📊 Category Analysis")
    cat_df = st.session_state.category_freq.rename_axis("Category").reset_index(name="Posts")
    cat_df["Percentage"] = (cat_df["Posts"] / cat_df["Posts"].sum() * 100).round(1)
    st.dataframe(cat_df, use_container_width=True)

    # Top Keywords
    st.subheader("🔑 Top Keywords Overall")
    kw_df = st.session_state.keyword_freq.reset_index()
    kw_df.columns = ["Keyword", "Frequency"]
    st.dataframe(kw_df.head(20), use_container_width=True)

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(6, 3))
        kw_df.head(20).set_index("Keyword")["Frequency"].plot.bar(ax=ax)
        ax.set_ylabel("Freq")
        ax.set_title("Top 20 keywords")
        st.pyplot(fig)

    # Sample Tagged Posts
    st.subheader("📄 Sample Tagged Posts")
    chosen_cat = st.selectbox("Select category to view sample posts:", cat_df["Category"])
    sample_posts = df[df["categories"].apply(lambda cats: chosen_cat in cats)][text_col].head(3)
    for idx, post in sample_posts.iteritems():
        st.text_area(f"Post {idx}", value=post, height=80, disabled=True)

    # Precision / Recall / F1 (if ground‑truth available)
    if st.session_state.y_true is not None:
        y_true = st.session_state.y_true
        y_pred = st.session_state.y_pred
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        st.subheader("🎯 Classification Metrics (binary)")
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1‑score", f"{f1:.3f}")

    # Full Results table download
    st.header("5. Download Results")
    st.download_button(
        "📥 Download Full Results (CSV)",
        df.to_csv(index=False).encode(),
        "full_results.csv",
        "text/csv",
    )
