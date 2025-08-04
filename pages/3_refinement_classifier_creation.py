# streamlit_app.py
# -------------------------------------------------------------
# Text Content Analysis Tool  – professor-spec version
# -------------------------------------------------------------
import re, ast, pandas as pd, streamlit as st

# ─── optional plotting ───────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ─────────────────────────────────────────────────────────────

def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\\s]", " ", str(txt)).lower().strip()

def classify(tokens: set[str], dicts: dict[str, set[str]]):
    return [k for k, v in dicts.items() if tokens & v] or ["uncategorized"]

# built-in dictionaries
DEFAULT_DICTS = {
    "urgency_marketing":  {"now","today","hurry","limited","final","deadline","ending","soon"},
    "exclusive_marketing":{"exclusive","vip","members","invite","private","insider","only"},
    "classic_timeless_luxury":{"timeless","heritage","vintage","classic","iconic","elegant"},
}

st.set_page_config("Text Content Analysis Tool", "🔍")
st.title("🔍 Text Content Analysis Tool")

# ------------------------------------------------------------------
# STEP 0 • choose / build tactic dictionary
# ------------------------------------------------------------------
st.header("0. Select or Create Tactic Dictionary")

mode = st.radio("Dictionary source",
                ("Use built-in tactic", "Provide custom dictionary"),
                horizontal=True)

if mode == "Use built-in tactic":
    tactic = st.selectbox("Built-in tactic to evaluate:", list(DEFAULT_DICTS))
    dictionaries = {tactic: DEFAULT_DICTS[tactic]}
    st.success(f"Selected built-in tactic **{tactic}**")
else:
    example = '{"my_tactic": ["word1", "word2", "word3"]}'
    user_text = st.text_area("Paste a Python dict with ONE tactic:", value=example, height=120)
    try:
        parsed = ast.literal_eval(user_text)
        if len(parsed) != 1:
            raise ValueError("Provide exactly ONE tactic in the dict.")
        tactic, words = next(iter(parsed.items()))
        dictionaries = {tactic: set(map(str.lower, words))}
        st.success(f"Custom tactic **{tactic}** loaded with {len(words)} keywords.")
    except Exception as e:
        st.error(f"❌ Problem parsing dict: {e}")
        st.stop()

# ------------------------------------------------------------------
# STEP 1 • upload CSV
# ------------------------------------------------------------------
st.header("1. Upload Data")
csv = st.file_uploader("Upload CSV file", type="csv")
if csv is None:
    st.stop()

df = pd.read_csv(csv)
st.dataframe(df.head(), use_container_width=True)

# choose text column
text_col = st.selectbox("Text column to analyse:", df.columns)

# optional ground-truth 0/1 column
bin_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
gt_col = st.selectbox("Ground-truth 0/1 column (optional):",
                      ["<none>"]+bin_cols, index=0)

# ------------------------------------------------------------------
# STEP 2 • run analysis
# ------------------------------------------------------------------
if st.button("🚀 Run Analysis"):
    # clean + tokenise
    tokens = df[text_col].fillna("").apply(lambda t: set(re.findall(r"\\w+", t.lower())))
    df["categories"] = tokens.apply(lambda s: classify(s, dictionaries))

    # overall keyword & category frequency
    all_toks = [w for s in tokens for w in s]
    kw_freq  = pd.Series(all_toks).value_counts()
    cat_freq = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()

    # -------------------------------- metrics ------------------------
    st.header("Results")

    st.subheader("Category frequency")
    st.dataframe(cat_freq.rename_axis("Category").to_frame("Posts"))

    st.subheader("Top keywords")
    st.dataframe(kw_freq.head(20).rename_axis("Keyword").to_frame("Freq"))

    if HAS_PLOT and not kw_freq.empty:
        fig, ax = plt.subplots(figsize=(6,3))
        kw_freq.head(20).plot.bar(ax=ax)
        ax.set_title("Top 20 keywords")
        st.pyplot(fig)

    # precision / recall / F1 if ground-truth selected
    if gt_col != "<none>":
        y_true = df[gt_col].astype(int)
        y_pred = df["categories"].apply(lambda cats: int(tactic in cats))
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0

        st.subheader(f"Classification metrics for **{tactic}**")
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall",    f"{rec:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")

    # download results
    st.download_button("📥 Download results csv",
                       df.to_csv(index=False).encode(),
                       "content_analysis_results.csv",
                       "text/csv")
