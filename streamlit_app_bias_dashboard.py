import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="centered")
st.title("🧠 AI Ethics & Bias Detection")
st.markdown("This app evaluates gender bias in income prediction using the UCI Adult Dataset.")

@st.cache_data
def load_data():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    df = X.copy()
    df["income"] = y
    df = df.replace("?", np.nan).dropna()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df

try:
    df = load_data()
    st.success(f"Dataset successfully loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Preview only 5 rows (no huge rendering)
    st.subheader("📊 Sample Data")
    st.dataframe(df.sample(5), use_container_width=True)

    # Sensitive attribute & features
    sensitive = df["sex"]
    X = df.drop(columns=["sex", "income"])
    y = df["income"]
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )

    # Train basic logistic model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.3f}")

    # Fairness metrics
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

    col1, col2 = st.columns(2)
    col1.metric("Demographic Parity Difference", f"{dp_diff:.3f}")
    col2.metric("Demographic Parity Ratio", f"{dp_ratio:.3f}")

    # Visualization: Income by gender
    st.subheader("📈 Gender-Based Income Prediction")
    gender_income = pd.concat([sensitive, df["income"]], axis=1)
    gender_income.columns = ["sex", "income"]
    fig, ax = plt.subplots()
    sns.barplot(data=gender_income, x="sex", y="income", ax=ax)
    ax.set_title("Proportion of >50K Income by Gender")
    st.pyplot(fig)

    # Summary
    st.subheader("📝 Insights")
    st.markdown("""
    - The model shows clear **gender-based prediction gaps**.
    - Demographic Parity metrics suggest possible **bias**.
    - Next steps: Apply mitigation techniques (e.g., `ExponentiatedGradient`).
    """)

except Exception as e:
    st.error("⚠️ An error occurred while running the app.")
    st.exception(e)

