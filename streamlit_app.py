import streamlit as st
import pandas as pd
import joblib
from auto import automate_training

st.title("⚡ No-Code ML Automation App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

    target = st.selectbox("Select Target Column", df.columns)

    # Train model
    if st.button("Train Model Automatically"):
        model_type, score, features = automate_training(df, target)
        st.success("Model training completed!")
        st.write("**Model Type:**", model_type)
        st.write("**Score:**", score)
        st.write("**Features Used:**", features)

    st.divider()

    st.write("### Predict on New Row")
    try:
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")

        inputs = {}

        for col in df.columns:
            if col != target:
                inputs[col] = st.text_input(f"Enter {col}")

        if st.button("Predict"):
            new_df = pd.DataFrame([inputs])

            # Apply encoders
            for col in new_df.columns:
                if col in encoders:
                    new_df[col] = encoders[col].transform(new_df[col].astype(str))

            pred = model.predict(new_df)[0]

            st.success(f"Prediction: {pred}")

    except:
        st.info("Train the model first before predicting.")
