# streamlit_app.py  (root level)
# -------------------------------------------------------------
# Entry point for the project.  The real functionality lives in
# pages/1_sentence_tokenizer.py  →  Sentence Tokenizer
# pages/2_rolling_context_builder.py  →  Rolling Context Builder
# -------------------------------------------------------------
import streamlit as st

st.set_page_config(page_title="Text-Analytic Project by Group A (Joseph, Jennifer, Basant, & Ismail)", page_icon="🔧")

st.title("🔧 Text-Analytic Project by Group A (Joseph, Jennifer, Basant, & Ismail)")

st.markdown(
    """
    Welcome! Use the sidebar to run the five-step pipeline:

    1. **Pre-processing_Sentence Tokenizer** – split each caption into individual sentences  
       *(downloads **_tokenised.csv**).*  
    2. **Pre-processing_Rolling Context Builder** – add a two-sentence (previous + current)
       rolling context window  
       *(downloads **_with_context.csv**).*
    3. **Refinement_Classifier Creationr** 
    4. **Classifier_Word_Metrics** 
    5. **Join_Table**
    ---
    **Tip:** pages appear automatically because they live in the `pages/`
    folder.  No extra routing code needed.
    """
)

# You could add a logo or additional instructions here if desired.
# Otherwise this page is just an informational landing screen.



