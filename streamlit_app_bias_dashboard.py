import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Title
st.set_page_config(layout="wide")
st.title("🤖 AI Ethics & Bias Evaluation Dashboard")
st.markdown("Analyze model fairness on the UCI Adult dataset using Fairlearn. Evaluate bias before and after mitigation.")

# Load Data
st.header("📂 Load UCI Adult Dataset")
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)

st.success(f"✅ Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

# Extract sensitive feature
sensitive_feature = df['sex'].copy()
df.drop(columns=['sex'], inplace=True)
y = df['income'].values
X = df.drop(columns=['income'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()

# Split data
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
st.header("🧠 Train Model & Evaluate")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Accuracy (Before Mitigation)", f"{acc:.3f}")

# Fairness evaluation
st.subheader("⚖️ Fairness Metrics (Before)")
metric_frame = MetricFrame(
    metrics={"Accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

st.write("Overall:")
st.write(metric_frame.overall)

st.write("By Group:")
st.dataframe(metric_frame.by_group)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

col1, col2 = st.columns(2)
col1.metric("Demographic Parity Difference", f"{dp_diff:.3f}")
col2.metric("Demographic Parity Ratio", f"{dp_ratio:.3f}")

# Bias mitigation
st.header("🔧 Bias Mitigation: Exponentiated Gradient")

X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

mitigator = ExponentiatedGradient(
    LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01
)

mitigator.fit(X_train_transformed, y_train, sensitive_features=sens_train)
y_pred_mitigated = mitigator.predict(X_test_transformed)

acc_mit = accuracy_score(y_test, y_pred_mitigated)
dp_diff_mit = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sens_test)
dp_ratio_mit = demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=sens_test)

col3, col4, col5 = st.columns(3)
col3.metric("Accuracy (After Mitigation)", f"{acc_mit:.3f}")
col4.metric("DP Difference (After)", f"{dp_diff_mit:.3f}")
col5.metric("DP Ratio (After)", f"{dp_ratio_mit:.3f}")

# Summary
st.header("📘 Ethical Summary & Takeaways")
st.markdown(f"""
- Sensitive feature: `sex`
- Accuracy dropped slightly after mitigation: **{acc:.3f} → {acc_mit:.3f}**
- Demographic Parity improved:
    - Difference: **{dp_diff:.3f} → {dp_diff_mit:.3f}**
    - Ratio: **{dp_ratio:.3f} → {dp_ratio_mit:.3f}**

✅ Fairness improved with Exponentiated Gradient.
💡 Balance accuracy and ethics depending on your deployment case.
""")

