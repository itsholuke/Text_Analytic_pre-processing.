# streamlit_app.py
# --------------------------------------------------------------------
# Text Content Analysis Tool  •  tactic first → upload → analyse
# --------------------------------------------------------------------
import re, ast, pandas as pd, streamlit as st

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False

# ---------- helpers --------------------------------------------------
def sent_tokens(text: str):
    return re.findall(r"\w+", str(text).lower())

def classify(tok_set, dicts):
    return [k for k, v in dicts.items() if tok_set & v] or ["uncategorized"]

def is_binary(col: pd.Series) -> bool:
    """Return True if column contains only 0/1 values; ignore unhashable."""
    try:
        vals = set(col.dropna().unique())
        return vals <= {0,1} or vals <= {0.0,1.0}
    except TypeError:
        return False
# ---------------------------------------------------------------------

DEFAULT_DICTS = {
    "urgency_marketing":  {"now","hurry","limited","today","final"},
    "exclusive_marketing":{"exclusive","vip","members","invite"},
    "classic_timeless_luxury":{"timeless","heritage","classic","iconic"},
}

st.set_page_config("Text Content Analysis Tool", "🔍")
st.title("🔍 Text Content Analysis Tool")

# 0 ── select tactic ---------------------------------------------------
mode = st.radio("Dictionary source", ("Built-in tactic", "Custom tactic"),
                horizontal=True)

if mode == "Built-in tactic":
    tactic = st.selectbox("Choose built-in tactic:", list(DEFAULT_DICTS))
    dictionaries = {tactic: DEFAULT_DICTS[tactic]}
else:
    example = '{"my_tactic": ["word1","word2","word3"]}'
    user_text = st.text_area("Paste one-tactic dict:", value=example, height=120)
    try:
        parsed = ast.literal_eval(user_text)
        if len(parsed) != 1:
            raise ValueError("Provide exactly one tactic")
        tactic, words = next(iter(parsed.items()))
        dictionaries = {tactic: set(map(str.lower, words))}
        st.success(f"Custom tactic **{tactic}** loaded.")
    except Exception as e:
        st.error(f"Dict parse error → {e}")
        st.stop()

# 1 ── upload CSV ------------------------------------------------------
csv = st.file_uploader("📁 Upload CSV to analyse", type="csv")
if csv is None: st.stop()

df = pd.read_csv(csv)
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("Text column:", df.columns)

# 2 ── run analysis ----------------------------------------------------
if st.button("🚀 Run Analysis"):
    tokens = df[text_col].fillna("").apply(sent_tokens)
    df["categories"] = tokens.apply(lambda t: classify(set(t), dictionaries))

    # frequencies
    kw_freq  = pd.Series([w for t in tokens for w in t]).value_counts()
    cat_freq = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()

    st.header("Results")
    st.subheader("Category frequency")
    st.dataframe(cat_freq.rename_axis("Category").to_frame("Posts"))

    st.subheader("Top keywords")
    st.dataframe(kw_freq.head(20).rename_axis("Keyword").to_frame("Freq"))

    if HAS_PLOT and not kw_freq.empty:
        fig, ax = plt.subplots(figsize=(6,3))
        kw_freq.head(20).plot.bar(ax=ax); ax.set_title("Top 20 keywords")
        st.pyplot(fig)

    # automatic ground-truth detection
    bin_cols = [c for c in df.columns if is_binary(df[c])]
    if len(bin_cols) == 1:
        gt_col = bin_cols[0]
        y_true = df[gt_col].astype(int)
        y_pred = df["categories"].apply(lambda cats: int(tactic in cats))
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0

        st.subheader(f"Precision / Recall / F1  •  tactic = **{tactic}** (ground-truth: {gt_col})")
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall",    f"{rec:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")
    else:
        st.info("No single binary ground-truth column detected – metrics skipped.")

    # download
    st.download_button("📥 Download results CSV",
                       df.to_csv(index=False).encode(),
                       "content_analysis_results.csv",
                       "text/csv")
