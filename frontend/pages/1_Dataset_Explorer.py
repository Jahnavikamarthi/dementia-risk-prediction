
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📂",
    layout="wide"
)

st.title("📂 ADNI Dataset Explorer")

st.write(
    "Upload and inspect clinical, cognitive and MRI numerical "
    "datasets from the Alzheimer's Disease Neuroimaging Initiative."
)

st.info(
    "Upload CSV files only. Do not upload restricted participant "
    "data to public services or share screenshots containing records."
)

uploaded_files = st.file_uploader(
    "Upload ADNI CSV datasets",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    for file in uploaded_files:

        st.divider()

        st.subheader(file.name)

        try:
            df = pd.read_csv(file, low_memory=False)

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", f"{len(df):,}")
            col2.metric("Total Columns", len(df.columns))
            col3.metric(
                "Missing Values",
                f"{df.isna().sum().sum():,}"
            )

            st.markdown("### Column Names")
            st.write(df.columns.tolist())

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("### Missing Value Analysis")

            missing = df.isnull().sum()

            missing = missing[missing > 0].sort_values(
                ascending=False
            )

            if len(missing) > 0:
                st.dataframe(
                    missing.rename("Missing Count").to_frame(),
                    use_container_width=True
                )
            else:
                st.success("No empty cells detected.")

        except Exception as e:
            st.error(f"Unable to read this file: {e}")

else:
    st.warning("Please upload your ADNI CSV files to begin.")
