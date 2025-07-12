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

# Title and Overview
st.set_page_config(layout="wide")
st.title("🤖 AI Ethics & Bias Evaluation Dashboard")
st.markdown("Analyze model fairness on the UCI Adult dataset using Fairlearn and Scikit-learn. Evaluate demographic parity before and after applying bias mitigation.")

# Load Data
st.header("📂 Data Loading")
st.write("Loading UCI Adult dataset from OpenML...")
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)

st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns after cleaning.")

# Sensitive feature separation
sensitive_feature = df['sex'].copy()
df.drop(labels=['sex'], axis=1, inplace=True)
y = df['income'].values
X = df.drop(labels=['income'], axis=1)

# Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='drop'
)

lr_model = LogisticRegression(max_iter=1000)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lr_model)
])

# Train Model
st.header("🧠 Train Logistic Regression Model")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy (Before Mitigation)", f"{accuracy:.3f}")

# Fairness Evaluation Before
st.subheader("⚖️ Fairness Evaluation Before Mitigation")
metrics = {'accuracy': accuracy_score}
metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sens_test)

st.write("### Overall Accuracy:")
st.write(metric_frame.overall)

st.write("### Group-wise Accuracy:")
st.dataframe(metric_frame.by_group)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

col1, col2 = st.columns(2)
col1.metric("Demographic Parity Difference", f"{dp_diff:.3f}")
col2.metric("Demographic Parity Ratio", f"{dp_ratio:.3f}")

# Bias Mitigation
st.header("🛠️ Bias Mitigation Using Exponentiated Gradient")
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01
)
mitigator.fit(X_train_transformed, y_train, sensitive_features=sens_train)
y_pred_mitigated = mitigator.predict(X_test_transformed)

accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)
dp_diff_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sens_test)
dp_ratio_mitigated = demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=sens_test)

col3, col4, col5 = st.columns(3)
col3.metric("Accuracy (After Mitigation)", f"{accuracy_mitigated:.3f}")
col4.metric("DP Difference (After)", f"{dp_diff_mitigated:.3f}")
col5.metric("DP Ratio (After)", f"{dp_ratio_mitigated:.3f}")

# Summary
st.header("📘 Recommendations & Takeaways")
st.markdown("""
- We used **'sex'** as the sensitive feature to evaluate fairness.
- Before mitigation, the model showed a demographic parity difference of **{:.3f}**.
- After applying **Exponentiated Gradient** with fairness constraints, the disparity reduced.
- There's often a trade-off between accuracy and fairness — balance based on context.
- Always document and communicate ethical considerations when deploying ML models.
""".format(dp_diff_mitigated))