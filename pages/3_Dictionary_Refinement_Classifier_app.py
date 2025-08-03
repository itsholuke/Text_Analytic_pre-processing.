# streamlit_app.py
import re, ast
import pandas as pd
import streamlit as st

# ── matplotlib is optional ──────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ────────────────────────────────────────────────────────────

# ---------- helpers ----------
def clean(txt: str) -> str:
    "Lower-case & keep only letters / digits / spaces."
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(txt)).lower().strip()

def classify(txt: str, dictionaries: dict) -> list[str]:
    words = set(txt.split())
    hits  = [name for name, terms in dictionaries.items() if words & terms]
    return hits or ["uncategorized"]

def init_state():
    default = dict(df=None, top_words=None,
                   dictionary=None, dict_ready=False)
    for k, v in default.items():
        st.session_state.setdefault(k, v)
# -----------------------------

st.set_page_config(page_title="Marketing-Tactic Text Classifier",
                   page_icon="📊", layout="wide")
st.title("📊 Marketing-Tactic Text Classifier")

# STEP 1 ─── tactic selector
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(DEFAULT_TACTICS))
st.write(f"Chosen tactic: **{tactic}**")

# STEP 2 ─── data upload
uploaded = st.file_uploader("📁 Step 2 — upload CSV", type="csv")
init_state()

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("📋 Step 3 — select text column", df.columns)
df["cleaned"] = df[text_col].apply(clean)

# STEP 4 ─── generate / edit dictionary
if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
    word_freq = (
        pd.Series(" ".join(df["cleaned"]).split())
        .value_counts()
        .loc[lambda s: s.gt(1)]
        .head(20)
    )
    st.subheader("Top keywords in the file")
    st.dataframe(word_freq, use_container_width=True)

    auto_dict = {tactic: word_freq.index.tolist()}
    st.code(auto_dict, language="python")

    dict_text = st.text_area("✏️ Edit dictionary (Python literal)",
                             value=str(auto_dict), height=150)

    try:
        final_dict = ast.literal_eval(dict_text)
        if not isinstance(final_dict, dict):
            raise ValueError("Not a dict.")
        final_dict = {k: set(map(str.lower, v)) for k, v in final_dict.items()}
        st.success("Dictionary saved.")
        st.session_state.update(df=df,
                                top_words=word_freq,
                                dictionary=final_dict,
                                dict_ready=True)
    except Exception as err:
        st.error(f"❌ Dictionary parse error: {err}")

# STEP 5 ─── classification
if st.session_state.dict_ready and st.button("🧪 Step 5 — Run Classification"):
    df         = st.session_state.df.copy()
    dictionary = st.session_state.dictionary
    top_words  = st.session_state.top_words

    df["categories"]  = df["cleaned"].apply(lambda t: classify(t, dictionary))
    df["tactic_flag"] = df["categories"].apply(lambda c: int(tactic in c))

    counts = (
        pd.Series([c for cats in df["categories"] for c in cats])
        .value_counts()
        .rename_axis("category")
        .to_frame("count")
    )

    st.subheader("📊 Category frequencies")
    st.dataframe(counts, use_container_width=True)

    st.subheader("🔑 Top keywords")
    st.dataframe(top_words, use_container_width=True)

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(8, 3))
        top_words.sort_values(ascending=False).plot.bar(ax=ax)
        ax.set_title("Top keyword frequencies")
        st.pyplot(fig)
    else:
        st.info("Matplotlib not found – skipping chart.")

    # downloads
    st.download_button("📥 classified_results.csv",
                       df.to_csv(index=False).encode(),
                       "classified_results.csv",
                       "text/csv")
    st.download_button("📥 category_frequencies.csv",
                       counts.to_csv().encode(),
                       "category_frequencies.csv",
                       "text/csv")
    st.download_button("📥 top_keywords.csv",
                       top_words.to_csv().encode(),
                       "top_keywords.csv",
                       "text/csv")
