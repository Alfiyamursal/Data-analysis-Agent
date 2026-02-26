import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Data Analysis Agent", layout="wide")

st.title("📊 Intelligent Data Analysis Agent")

# File Upload
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Dataset Info
    st.subheader("Dataset Information")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        plt.figure(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    else:
        st.write("No numeric columns available.")

    # Outlier Detection
    st.subheader("Outlier Detection (Boxplots)")
    numeric_cols = numeric_df.columns

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=numeric_df[col], ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    # =====================
    # Machine Learning Agent
    # =====================
    st.subheader("Model Training")

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Keep only numeric features
            X = X.select_dtypes(include=np.number)
            X = X.fillna(0)

            if X.shape[1] == 0:
                st.error("No numeric features available for training.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Auto detect problem type
                if y.nunique() <= 10:
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    accuracy = model.score(X_test, y_test)
                    st.success(f"Classification Accuracy: {accuracy:.2f}")
                else:
                    model = RandomForestRegressor()
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    st.success(f"Regression R² Score: {score:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

    # Download processed data
    st.subheader("Download Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")