# streamlit_app.py  —  trimmed: no ground-truth dropdown
import re, ast, pandas as pd, streamlit as st
try:
    import matplotlib.pyplot as plt; HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False

# ------------ helpers -------------------------------------------------
def clean(txt): return re.sub(r"[^a-zA-Z0-9\\s]", " ", str(txt)).lower()
def classify(toks, d): return [k for k,v in d.items() if set(toks)&v] or ["uncategorized"]
# ---------------------------------------------------------------------

DEFAULT_DICTS = {     # shortened for brevity
    "urgency_marketing": {"now","today","hurry","limited","final"},
    "exclusive_marketing":{"exclusive","vip","members","invite"},
}

st.set_page_config("Text Content Analysis Tool", "🔍")
st.title("🔍 Text Content Analysis Tool")

# 0 ── choose tactic ----------------------------------------------------
mode = st.radio("Dictionary source", ("Built-in", "Custom"), horizontal=True)
if mode == "Built-in":
    tactic = st.selectbox("Choose tactic:", list(DEFAULT_DICTS))
    dictionaries = {tactic: DEFAULT_DICTS[tactic]}
else:
    user = st.text_area("Paste a ONE-tactic dict:", '{"my_tactic":["word1","word2"]}')
    try:
        parsed = ast.literal_eval(user); tactic, words = next(iter(parsed.items()))
        dictionaries = {tactic: set(map(str.lower, words))}
        st.success(f"Loaded custom tactic **{tactic}** with {len(words)} keywords.")
    except Exception as e:
        st.error(f"Dict parse error → {e}"); st.stop()

# 1 ── upload data ------------------------------------------------------
csv = st.file_uploader("Upload CSV", type="csv")
if csv is None: st.stop()
df = pd.read_csv(csv)
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("Text column:", df.columns)

# 2 ── run analysis -----------------------------------------------------
if st.button("🚀 Run Analysis"):
    tokens  = df[text_col].fillna("").apply(lambda t: re.findall(r"\\w+", t.lower()))
    df["categories"] = tokens.apply(lambda t: classify(t, dictionaries))

    # keyword / category frequency
    kw_freq  = pd.Series([w for toks in tokens for w in toks]).value_counts()
    cat_freq = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()

    st.subheader("Category frequency")
    st.dataframe(cat_freq.rename_axis("Category").to_frame("Posts"))

    st.subheader("Top keywords")
    st.dataframe(kw_freq.head(20).rename_axis("Keyword").to_frame("Freq"))

    if HAS_PLOT and not kw_freq.empty:
        fig, ax = plt.subplots(figsize=(6,3))
        kw_freq.head(20).plot.bar(ax=ax); ax.set_title("Top 20 keywords"); st.pyplot(fig)

    # --- automatic ground-truth detection -----------------------------
    bin_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
    if len(bin_cols) == 1:
        gt_col = bin_cols[0]
        y_true = df[gt_col].astype(int)
        y_pred = df["categories"].apply(lambda cats: int(tactic in cats))
        tp = int(((y_true==1)&(y_pred==1)).sum()); fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0

        st.subheader(f"Precision / Recall / F1 for **{tactic}** (ground-truth: {gt_col})")
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall",    f"{rec:.3f}")
        st.metric("F1-score",  f"{f1:.3f}")
    else:
        st.info("No single 0/1 ground-truth column detected → metrics skipped.")

    # download
    st.download_button("📥 Download results CSV",
                       df.to_csv(index=False).encode(),
                       "content_analysis_results.csv",
                       "text/csv")
