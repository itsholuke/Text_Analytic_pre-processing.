# pages/3_Refinement_Classifier_Creation.py
# --------------------------------------------------------------------
# Refinement Classifier Creation – tactic dictionary + metrics
# --------------------------------------------------------------------
# • Upload a CSV that already contains your ground‑truth 0/1 column(s)
# • Paste / edit a Python dict that maps **one tactic** → list of keywords
# • Classify every row, show frequency table, bar‑chart (if matplotlib is
#   installed), and precision / recall / F1 when a single binary column
#   is chosen.
# • Safe: skips chart gracefully if matplotlib is missing.
# --------------------------------------------------------------------
import re, ast, pandas as pd, streamlit as st

# ── optional plotting (auto‑skip if matplotlib missing) ───────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    HAS_PLOT = False
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Refinement Classifier Creation", page_icon="🛠️")
st.title("🛠️ Refinement Classifier Creation")

# ---------- upload ---------------------------------------------------
upload = st.file_uploader("📁 Upload CSV", type="csv")
if upload is None:
    st.stop()

df = pd.read_csv(upload)
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

text_col = st.selectbox("Text column:", df.columns)

# auto‑detect 0/1 columns
bin_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0, 1}]
gt_col = st.selectbox("Ground‑truth 0/1 column (optional):", ["<none>"] + bin_cols)

example = '{"urgency_marketing": ["now", "today", "hurry"]}'
user_dict = st.text_area("Paste tactic dictionary (ONE key)", example, height=140)
try:
    tactic_dict = {k: set(map(str.lower, v)) for k, v in ast.literal_eval(user_dict).items()}
    if len(tactic_dict) != 1:
        raise ValueError("Provide exactly ONE tactic in the dict.")
except Exception as e:
    st.error(f"Dict parse error → {e}")
    st.stop()

tactic_name, keywords = next(iter(tactic_dict.items()))

# ---------- helper functions ----------------------------------------
clean = lambda t: re.sub(r"[^a-zA-Z0-9\s]", "", str(t).lower())
classify_flag = lambda toks: int(any(w in toks for w in keywords))
# --------------------------------------------------------------------

if st.button("🚀 Run"):
    with st.spinner("Classifying…"):
        df["_clean"]   = df[text_col].apply(clean)
        df["pred_flag"] = df["_clean"].apply(lambda x: classify_flag(x.split()))

    freq = df["pred_flag"].value_counts().rename(index={0: "No", 1: "Yes"})
    st.subheader("Predicted tactic flag frequency")
    st.dataframe(freq.rename("Rows"))

    # bar‑chart if matplotlib available
    if HAS_PLOT and not freq.empty:
        fig, ax = plt.subplots()
        freq.plot.bar(ax=ax)
        ax.set_ylabel("Rows"); ax.set_title("Predicted tactic flag")
        st.pyplot(fig)
    elif not HAS_PLOT:
        st.info("Matplotlib not installed – skipping bar chart.")

    # metrics if ground‑truth chosen
    if gt_col != "<none>":
        y_true = df[gt_col].astype(int)
        y_pred = df["pred_flag"]
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        st.subheader("Classification metrics")
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall",    f"{rec:.3f}")
        st.metric("F1‑score",  f"{f1:.3f}")

    st.download_button("💾 Download classified CSV",
                       df.drop(columns="_clean").to_csv(index=False).encode(),
                       "refined_results.csv", "text/csv")
