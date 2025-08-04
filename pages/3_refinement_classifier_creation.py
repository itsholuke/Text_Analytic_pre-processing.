# ───────────────────────────────────────────────────────────
#  refinement_classifier_creation – numeric-flag (patched)
#  ----------------------------------------------------------
#  • Build / edit tactic-aware dictionary
#  • Classify text → 0/1 tactic_flag
#  • Optional ground-truth (upload or manual)
#  • Precision / recall / F1
# ───────────────────────────────────────────────────────────
import ast, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing-Tactic Text Classifier + Metrics")

# ---------- dictionaries ----------
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now","today","limited","hurry","exclusive"],
    "social_proof":       ["bestseller","popular","trending","recommended"],
    "discount_marketing": ["sale","discount","deal","free","offer"],
    "Classic_Timeless_Luxury_style":[
        "elegance","heritage","sophistication","refined","timeless","grace","legacy",
        "opulence","bespoke","tailored","understated","prestige","quality","craftsmanship",
        "heirloom","classic","tradition","iconic","enduring","rich","authentic","luxury",
        "fine","pure","exclusive","elite","mastery","immaculate","flawless","distinction",
        "noble","chic","serene","clean","minimal","poised","balanced","eternal","neutral",
        "subtle","grand","timelessness","tasteful","quiet","sublime",
    ],
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: *{tactic}*")

# ---------- helpers ----------
def clean(txt:str)->str:
    return re.sub(r"[^a-zA-Z0-9\\s]","",str(txt).lower())

def classify(txt:str,dct):
    toks = txt.split()
    return [cat for cat,terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

def to_list(x):
    if isinstance(x,list):             return x
    if isinstance(x,str) and x.startswith("["):
        try: return ast.literal_eval(x)
        except Exception: return []
    return []

def safe_bool(x):
    if isinstance(x,(int,float)): return bool(x)
    if isinstance(x,str): return x.strip().lower() in {"1","true","yes"}
    return False

# ---------- session flags ----------
for flag in ["dict_ready","pred_ready","gt_ready"]:
    st.session_state.setdefault(flag, False)

# ---------- STEP 2 upload ----------
raw = st.file_uploader("📁 Step 2 — upload raw CSV",type="csv")
if raw:
    st.session_state.raw_df = pd.read_csv(raw)
    st.session_state.pred_ready = st.session_state.gt_ready = False
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0,"ID",st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

if "raw_df" not in st.session_state or st.session_state.raw_df.empty:
    st.stop()

text_col = st.selectbox("📋 Step 3 — select text column", st.session_state.raw_df.columns)

# ---------- STEP 4 dict ----------
if st.button("🧠 Step 4 — Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    seeds = set(DEFAULT_TACTICS[tactic])
    pos = df[df["cleaned"].apply(lambda x:any(tok in x.split() for tok in seeds))]
    stop = {"the","is","in","on","and","a","for","you","i","are","of",
            "your","to","my","with","it","me","this","that","or"}
    if pos.empty:
        ctx_terms, ctx_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; default list only.")
    else:
        freq = pos["cleaned"].str.split(expand=True).stack().value_counts()
        ctx_terms = [w for w in freq.index if w not in stop and w not in seeds][:30]
        ctx_freq  = freq.loc[ctx_terms]

    auto_dict = {tactic: sorted(seeds.union(ctx_terms))}
    st.dataframe(ctx_freq.rename("Freq")) if not ctx_freq.empty else st.write("— none found —")
    dict_txt = st.text_area("✏ Edit dictionary (Python dict)", value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_txt)
        st.session_state.dict_ready = True
        st.success("Dictionary saved.")
    except Exception:
        st.error("Bad format — dict not saved.")
        st.session_state.dict_ready = False

# ---------- STEP 5-A classify ----------
st.subheader("Step 5-A — Classification")
if st.button("🔹 1. Run Classification", disabled=not st.session_state.dict_ready):
    df = st.session_state.raw_df.copy()
    df["cleaned"]    = df[text_col].apply(clean)
    df["categories"] = df["cleaned"].apply(lambda x: classify(x, st.session_state.dictionary))
    df["tactic_flag"]= df["categories"].apply(lambda cats:int(tactic in cats))
    st.session_state.pred_df = df
    st.session_state.pred_ready = True
    st.session_state.gt_ready   = False
    st.dataframe(df.head())

if st.session_state.pred_ready:
    counts = pd.Series([c for cats in st.session_state.pred_df["categories"] for c in cats]).value_counts()
    st.table(counts)

# ---------- STEP 5-B GT ----------
st.subheader("Step 5-B — Ground-truth & Metrics")
choice = st.radio("Ground-truth source",["None","Upload CSV","Manual entry"],horizontal=True)

if choice!="Upload CSV":
    st.session_state.gt_df = pd.DataFrame()

if choice=="Upload CSV":
    up = st.file_uploader("Upload CSV with ID + true_label / flag",type="csv",key="gt_up")
    if up:
        st.session_state.gt_df = pd.read_csv(up)
        st.session_state.gt_ready=True
        st.success("Ground-truth loaded.")

elif choice=="Manual entry" and st.session_state.pred_ready:
    flag = f"{tactic}_flag_gt"
    prev = "_snippet_"
    df_e = st.session_state.pred_df.copy()
    if flag not in df_e.columns: df_e[flag]=0
    df_e[flag]=pd.to_numeric(df_e[flag],errors="coerce").fillna(0).astype(int)
    if prev not in df_e.columns:
        df_e[prev]=df_e[text_col].astype(str).str.slice(0,120)
    edited = st.data_editor(df_e[["ID",prev,flag]],
                            column_config={flag:st.column_config.NumberColumn(min_value=0,max_value=1)},
                            use_container_width=True,height=600,key="edit_gt")
    st.session_state.pred_df[flag]=pd.to_numeric(edited[flag],errors="coerce").fillna(0).astype(int)
    st.session_state.pred_df["true_label"]=st.session_state.pred_df[flag].apply(lambda x:[tactic] if x else [])
    st.session_state.gt_ready=True

# ---------- metrics ----------
if st.button("🔹 2. Compute Metrics", disabled=not (st.session_state.pred_ready and st.session_state.gt_ready)):
    dfp = st.session_state.pred_df.copy()
    if not st.session_state.gt_df.empty:
        gt = st.session_state.gt_df.copy()
        flag_col = f"{tactic}_flag"
        if flag_col in gt.columns:
            gt["true_label"]=gt[flag_col].apply(lambda x:[tactic] if safe_bool(x) else [])
        elif "true_label" not in gt.columns:
            st.error("Ground-truth CSV must have true_label or flag column.")
            st.stop()
        dfp = dfp.merge(gt[["ID","true_label"]],on="ID",how="left",suffixes=("","_y"))
        if "true_label_y" in dfp.columns:
            dfp["true_label"]=dfp["true_label_y"].combine_first(dfp["true_label"])
            dfp.drop(columns="true_label_y",inplace=True)

    if dfp["true_label"].isna().all():
        st.warning("No ground-truth → can't compute.")
        st.stop()

    dfp["gt_list"]   = dfp["true_label"].apply(to_list)
    dfp["pred_list"] = dfp["categories"]

    rows=[]
    for tac,terms in st.session_state.dictionary.items():
        dfp["pred_flag"]=dfp["pred_list"].apply(lambda lst:tac in lst)
        dfp["gt_flag"]  =dfp["gt_list"].apply(lambda lst:tac in lst)
        TP=int((dfp.pred_flag & dfp.gt_flag).sum())
        FP=int((dfp.pred_flag & ~dfp.gt_flag).sum())
        FN=int((~dfp.pred_flag & dfp.gt_flag).sum())
        prec=TP/(TP+FP) if TP+FP else 0
        rec =TP/(TP+FN) if TP+FN else 0
        f1 =2*prec*rec/(prec+rec) if prec+rec else 0
        rows.append(dict(tactic=tac,TP=TP,FP=FP,FN=FN,
                         precision=prec,recall=rec,f1=f1))

    st.dataframe(pd.DataFrame(rows).set_index("tactic").style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}))
    st.session_state.pred_df = dfp  # store with GT merged

# ---------- download ----------
if st.session_state.pred_ready:
    st.download_button("classified_results.csv",
                       st.session_state.pred_df.to_csv(index=False).encode(),
                       "classified_results.csv", mime="text/csv")
