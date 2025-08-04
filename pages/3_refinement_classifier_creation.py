# streamlit_app.py  –  FULL VERSION with tactic-aware Step 4  (robust)
import streamlit as st
import pandas as pd
import re, ast

# ── optional plotting (auto-skip if matplotlib missing) ──────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Marketing-Tactic Text Classifier", page_icon="📊")
st.title("📊 Marketing-Tactic Text Classifier")

# ───────── STEP 1 – choose tactic ─────────
default_tactics = {
    "urgency_marketing":  ["now","today","limited","hurry","exclusive"],
    "social_proof":       ["bestseller","popular","trending","recommended"],
    "discount_marketing": ["sale","discount","deal","free","offer"],
    "Classic_Timeless_Luxury_style": [
        "elegance","heritage","sophistication","refined","timeless","grace",
        "legacy","opulence","bespoke","tailored","understated","prestige",
        "quality","craftsmanship","heirloom","classic","tradition","iconic",
        "enduring","rich","authentic","luxury","fine","pure","exclusive",
        "elite","mastery","immaculate","flawless","distinction","noble",
        "chic","serene","clean","minimal","poised","balanced","eternal",
        "neutral","subtle","grand","timelessness","tasteful","quiet","sublime",
    ]
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(default_tactics))
st.write(f"Chosen tactic: *{tactic}*")

# ---------- helpers --------------------------------------------------
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\\s]", "", str(txt).lower())

def classify(txt: str, d):
    toks = txt.split()
    return [c for c, terms in d.items() if any(w in toks for w in terms)] \
           or ["uncategorized"]

def safe_read_csv(f):
    """Try utf-8, utf-8-sig, latin1, cp1252 encodings until one works."""
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            f.seek(0)
            return pd.read_csv(f, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("common encodings failed", b"", 0, 0, "")
# ---------------------------------------------------------------------

# session defaults
for k in ("dict_ready","dictionary","top_words","df"):
    st.session_state.setdefault(k, None)

# ───────── STEP 2 – upload CSV ───────────
upload = st.file_uploader("📁 Step 2 — upload CSV", type="csv")

if upload:
    try:
        df = safe_read_csv(upload)
    except UnicodeDecodeError as e:
        st.error(f"Cannot decode CSV: {e}")
        st.stop()

    st.dataframe(df.head(), use_container_width=True)
    text_col = st.selectbox("📋 Step 3 — text column", df.columns)

    # ----- Step 4 — generate / refine dictionary -----
    if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
        with st.spinner("Cleaning & mining keywords…"):
            df["cleaned"] = df[text_col].apply(clean)

            base_terms = set(default_tactics[tactic])
            df["row_matches"] = df["cleaned"].apply(
                lambda s: any(w in s.split() for w in base_terms)
            )
            pos_df = df[df["row_matches"]]

            stop = {"the","is","in","on","and","a","for","you","i","are","of",
                    "your","to","my","with","it","me","this","that","or"}

            if pos_df.empty:
                contextual_terms, contextual_freq = [], pd.Series(dtype=int)
                st.warning("No rows contained the base terms for this tactic.")
            else:
                words = " ".join(pos_df["cleaned"]).split()
                freq  = pd.Series(words).value_counts()
                contextual_terms = [
                    w for w in freq.index if w not in stop and w not in base_terms
                ][:30]
                contextual_freq = freq.loc[contextual_terms]

            auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

        st.subheader("Top contextual keywords (filtered)")
        st.dataframe(contextual_freq.rename("Freq")) \
            if not contextual_freq.empty else st.write("-- none found --")

        dict_txt = st.text_area("✏ Edit dictionary (Python dict)", value=str(auto_dict), height=160)
        try:
            st.session_state.dictionary = ast.literal_eval(dict_txt)
            st.success("Dictionary saved.")
        except Exception:
            st.error("Bad format → using auto dict.")
            st.session_state.dictionary = auto_dict

        st.session_state.df        = df
        st.session_state.top_words = contextual_freq
        st.session_state.dict_ready = True

    # ----- Step 5 — run classifier -----
    if st.session_state.dict_ready and st.button("🧪 Step 5 — Run Classification"):
        df         = st.session_state.df.copy()
        top_words  = st.session_state.top_words
        dictionary = st.session_state.dictionary

        with st.spinner("Classifying…"):
            df["categories"] = df["cleaned"].apply(lambda x: classify(x, dictionary))
            df["tactic_flag"] = df["categories"].apply(lambda cats: int(tactic in cats))

        counts = pd.Series(
            [c for cats in df["categories"] for c in cats]
        ).value_counts()

        st.subheader("📊 Category frequencies")
        st.dataframe(counts.rename("Posts"))

        st.subheader("🔑 Top contextual keywords")
        st.dataframe(top_words.rename("Freq")) if not top_words.empty \
            else st.write("-- none to display --")

        if HAS_PLOT and not top_words.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            top_words.sort_values(ascending=False).plot.bar(ax=ax)
            ax.set_title("Top contextual keyword frequencies")
            st.pyplot(fig)
        elif not HAS_PLOT:
            st.info("Matplotlib not installed – skipping bar chart.")

        # downloads
        st.download_button("📥 classified_results.csv",
                           df.to_csv(index=False).encode(),
                           "classified_results.csv", "text/csv")
        st.download_button("📥 category_frequencies.csv",
                           counts.to_csv().encode(),
                           "category_frequencies.csv", "text/csv")
        if not top_words.empty:
            st.download_button("📥 top_keywords.csv",
                               top_words.to_csv().encode(),
                               "top_keywords.csv", "text/csv")
else:
    st.info("Upload a CSV to begin.")
