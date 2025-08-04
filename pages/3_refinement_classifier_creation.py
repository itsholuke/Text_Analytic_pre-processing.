# ────────────────────────────────────────────────────────────
#  streamlit_app.py  –  FULL VERSION with tactic‑aware Step 4
# ────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re, ast

st.title("📊 Marketing‑Tactic Text Classifier")

# ───────────────── STEP 1 – choose tactic ───────────────────
default_tactics = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        'elegance', 'heritage', 'sophistication', 'refined', 'timeless', 'grace',
        'legacy', 'opulence', 'bespoke', 'tailored', 'understated', 'prestige',
        'quality', 'craftsmanship', 'heirloom', 'classic', 'tradition', 'iconic',
        'enduring', 'rich', 'authentic', 'luxury', 'fine', 'pure', 'exclusive',
        'elite', 'mastery', 'immaculate', 'flawless', 'distinction', 'noble',
        'chic', 'serene', 'clean', 'minimal', 'poised', 'balanced', 'eternal',
        'neutral', 'subtle', 'grand', 'timelessness', 'tasteful', 'quiet', 'sublime'
    ]
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(default_tactics.keys()))
st.write(f"Chosen tactic: *{tactic}*")

# ───────────────── STEP 2 – upload CSV ──────────────────────
file = st.file_uploader("📁 Step 2 — upload CSV", type="csv")

# ---------- helper functions --------------------------------
def clean(txt: str) -> str:
    """Lower‑case & remove punctuation/digits for simple tokenisation."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, dct):
    """Return list of categories whose term list appears at least once."""
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]
# ------------------------------------------------------------

# ────────── Streamlit session default flags/objects ─────────
if "dict_ready"  not in st.session_state: st.session_state.dict_ready  = False
if "dictionary"  not in st.session_state: st.session_state.dictionary  = {}
if "top_words"   not in st.session_state: st.session_state.top_words   = pd.Series(dtype=int)
if "df"          not in st.session_state: st.session_state.df          = pd.DataFrame()

# ────────────────────────────────────────────────────────────
#                   MAIN APP LOGIC
# ────────────────────────────────────────────────────────────
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    text_col = st.selectbox("📋 Step 3 — select text column", df.columns)

    # ─────────── STEP 4 – generate / refine dictionary ──────
    if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
        df["cleaned"] = df[text_col].apply(clean)

        # 1. tactic seed terms
        base_terms = set(default_tactics[tactic])

        # 2. rows that mention ≥1 base term
        df["row_matches_tactic"] = df["cleaned"].apply(
            lambda x: any(tok in x.split() for tok in base_terms)
        )
        pos_df = df[df["row_matches_tactic"]]

        # 3. contextual term mining inside matching rows
        stop_words = {
            'the', 'is', 'in', 'on', 'and', 'a', 'for', 'you', 'i', 'are', 'of',
            'your', 'to', 'my', 'with', 'it', 'me', 'this', 'that', 'or'
        }

        if pos_df.empty:
            st.warning(
                "None of the rows contained the base terms for this tactic. "
                "Auto dictionary will consist of the default terms only."
            )
            contextual_terms = []
            contextual_freq  = pd.Series(dtype=int)
        else:
            all_pos_words = " ".join(pos_df["cleaned"]).split()
            word_freq = pd.Series(all_pos_words).value_counts()

            contextual_terms = [
                w for w in word_freq.index
                if w not in stop_words and w not in base_terms
            ][:30]                              # top‑30 contextual words
            contextual_freq = word_freq.loc[contextual_terms]

        # 4. merge seed + contextual → dictionary
        auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

        # 5. show info and allow editing
        st.subheader("Top contextual keywords (filtered)")
        if not contextual_freq.empty:
            st.dataframe(contextual_freq.rename("Frequency"))
        else:
            st.write("‑‑ none found ‑‑")

        st.write("Auto‑generated dictionary:", auto_dict)

        dict_text = st.text_area(
            "✏ Edit dictionary (Python dict syntax)",
            value=str(auto_dict),
            height=150
        )
        try:
            final_dict = ast.literal_eval(dict_text)
            st.session_state.dictionary = final_dict
            st.success("Dictionary saved.")
        except Exception:
            st.error("Bad format → using auto dict.")
            st.session_state.dictionary = auto_dict

        # 6. persist for Step 5
        st.session_state.df         = df
        st.session_state.top_words  = contextual_freq           # Series
        st.session_state.dict_ready = True

    # ─────────── STEP 5 – run classifier (only if ready) ────
    if st.session_state.dict_ready:
        if st.button("🧪 Step 5 — Run Classification"):
            df         = st.session_state.df.copy()
            top_words  = st.session_state.top_words
            dictionary = st.session_state.dictionary

            df["categories"] = df["cleaned"].apply(lambda x: classify(x, dictionary))
            df["tactic_flag"] = df["categories"].apply(
                lambda cats: 1 if tactic in cats else 0
            )

            counts = pd.Series(
                [c for cats in df["categories"] for c in cats]
            ).value_counts()

            st.subheader("📊 Category frequencies")
            st.table(counts)

            st.subheader("🔑 Top contextual keywords")
            if not top_words.empty:
                st.table(top_words)
            else:
                st.write("‑‑ none to display ‑‑")

            fig, ax = plt.subplots(figsize=(10, 4))
            if not top_words.empty:
                top_words.sort_values(ascending=False).plot.bar(ax=ax)
                ax.set_title("Top contextual keyword frequencies")
            else:
                ax.text(0.5, 0.5, "No contextual keywords", ha="center", va="center")
                ax.set_axis_off()
            st.pyplot(fig)

            # downloads
            st.download_button(
                "📥 classified_results.csv",
                df.to_csv(index=False).encode(),
                "classified_results.csv",
                "text/csv",
            )
            st.download_button(
                "📥 category_frequencies.csv",
                counts.to_csv().encode(),
                "category_frequencies.csv",
                "text/csv",
            )
            if not top_words.empty:
                st.download_button(
                    "📥 top_keywords.csv",
                    top_words.to_csv().encode(),
                    "top_keywords.csv",
                    "text/csv",
                )
else:
    st.info("Upload a CSV to begin.")
