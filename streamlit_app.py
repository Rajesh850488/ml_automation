# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from auto import run_automation
import time

st.set_page_config(page_title="ML Automation Enterprise", layout="wide")

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div style="background-color:#0B5394;padding:15px;border-radius:10px">
<h1 style="color:white;text-align:center;">🚀 ML Automation Enterprise Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
model_option = st.sidebar.selectbox("Select ML Model", ["RandomForest", "DecisionTree", "LogisticRegression"])
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

# Initialize result dictionary
result = {}

# ---------------------------
# Main Workflow
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    st.success("CSV Loaded Successfully!")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Preview", "🧹 Cleaning", "📈 EDA", "🤖 ML Automation", "🖥️ Single Prediction"])

    # ---------------------------
    # Tab 1: Preview
    # ---------------------------
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.markdown(f"**Dataset Shape:** {df.shape[0]} rows x {df.shape[1]} columns")

    # ---------------------------
    # Tab 2: Cleaning
    # ---------------------------
    with tab2:
        st.subheader("Null Values")
        nulls = df.isnull().sum()
        for col, val in nulls.items():
            color = "#0F9D58" if val == 0 else "#F4B400"
            st.markdown(f"<span style='color:{color}'>{col}: {val}</span>", unsafe_allow_html=True)

        st.subheader("Duplicate Rows")
        dup = df.duplicated().sum()
        color = "#0F9D58" if dup == 0 else "#F4B400"
        st.markdown(f"<span style='color:{color}'>Duplicate Rows: {dup}</span>", unsafe_allow_html=True)

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed!")

    # ---------------------------
    # Tab 3: EDA
    # ---------------------------
    with tab3:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include='all'))

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.subheader("Column Distributions")
            for col in numeric_df.columns:
                fig, ax = plt.subplots()
                sns.histplot(numeric_df[col], kde=True, color="#0B5394", ax=ax)
                st.pyplot(fig)

    # ---------------------------
    # Tab 4: ML Automation
    # ---------------------------
    with tab4:
        if st.button("Run ML Automation"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, step in enumerate(["Cleaning Data", "EDA", "Training Model", "Generating Results"]):
                status_text.text(f"Step: {step}")
                time.sleep(0.5)
                progress_bar.progress((i + 1) / 4)

            result = run_automation(df, target_column, model_option, test_size / 100)
            st.success("ML Automation Completed ✅")
            progress_bar.empty()
            status_text.empty()

            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rows", df.shape[0])
            k2.metric("Columns", df.shape[1])
            k3.metric("Duplicates", result['duplicates'])
            if result['ml_accuracy'] is not None:
                k4.metric("ML Accuracy", f"{result['ml_accuracy'] * 100:.2f}%")

            # Feature Importance
            if result['feature_importance'] is not None:
                st.subheader("Feature Importance")
                fi = result['feature_importance']
                st.bar_chart(fi.set_index('Feature')['Importance'])

            # Cleaned CSV Download
            st.subheader("Download Cleaned Data")
            cleaned_csv = result['cleaned_df'].to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=cleaned_csv, file_name="cleaned_data.csv", mime="text/csv")

    # ---------------------------
    # Tab 5: Single Prediction
    # ---------------------------
    with tab5:
        st.subheader("Predict Single Row")
        if result.get('predict_single') is not None:
            input_dict = {}
            for col in df.drop(columns=[target_column]).columns:
                val = st.text_input(f"{col}")
                input_dict[col] = val
            if st.button("Predict"):
                try:
                    pred = result['predict_single'](input_dict)
                    st.success(f"Predicted value: {pred}")
                except:
                    st.error("Invalid input. Make sure numeric columns are numbers.")
        else:
            st.info("Run ML Automation first to enable single row prediction")

else:
    st.info("Upload CSV to start ML Automation")
