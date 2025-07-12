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

# Load data
@st.cache_data
def load_data():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    df = X.copy()
    df["income"] = y
    df = df.replace("?", np.nan).dropna()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df

df = load_data()
st.success(f"Dataset loaded: {df.shape[0]} samples")

# Overview
st.subheader("📊 Dataset Overview")
st.write(df.sample(5))

# Sensitive feature
sensitive = df["sex"]
X = df.drop(columns=["sex", "income"])
y = df["income"]

# Simple encoding
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Train model
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
col1.metric("DP Difference", f"{dp_diff:.3f}")
col2.metric("DP Ratio", f"{dp_ratio:.3f}")

# Bar chart: Income by gender
st.subheader("📈 Income Distribution by Gender")
gender_income = pd.concat([sensitive, df["income"]], axis=1)
gender_income.columns = ["sex", "income"]
fig, ax = plt.subplots()
sns.barplot(data=gender_income, x="sex", y="income", ax=ax)
ax.set_title("Proportion of >50K Income by Gender")
st.pyplot(fig)

# Final notes
st.subheader("📝 Summary")
st.markdown("""
- We analyzed the **UCI Adult** dataset for gender bias.
- The model achieved decent accuracy but shows **potential demographic disparity**.
- This suggests gender may **indirectly affect** income predictions.
- Next steps: Apply **bias mitigation techniques** for fairness.
""")
