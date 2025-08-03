import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title="CSV File Joiner", layout="wide")
st.title("🔗 CSV File Joiner")

st.write("Upload two CSV files and choose how to join them using inner, left, right, or outer joins. "
         "You can also remove duplicate rows before merging.")

uploaded_file1 = st.file_uploader("📁 Upload File 1", type=["csv"])
uploaded_file2 = st.file_uploader("📁 Upload File 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    st.subheader("⚙️ Join Configuration")

    col1 = st.selectbox("Select join column from File 1", df1.columns)
    col2 = st.selectbox("Select join column from File 2", df2.columns)
    join_type = st.selectbox("Select Join Type", ["inner", "left", "right", "outer"])
    remove_duplicates = st.checkbox("✅ Remove duplicate rows before joining", value=True)

    if st.button("🔗 Perform Join"):
        try:
            if remove_duplicates:
                df1 = df1.drop_duplicates()
                df2 = df2.drop_duplicates()

            result = pd.merge(
                df1, df2,
                how=join_type,
                left_on=col1,
                right_on=col2,
                suffixes=("", "_file2")
            )

            st.success(f"✅ Join successful! {len(result)} rows returned.")
            st.dataframe(result.head(50), use_container_width=True)

            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Joined CSV",
                data=csv,
                file_name='joined_result.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"❌ Error during join: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit. Developed by Joseph Reyes.")
