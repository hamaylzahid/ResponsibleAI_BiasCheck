import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# ------------------------------
# TITLE AND PROJECT OVERVIEW
# ------------------------------
st.set_page_config(page_title="AI Ethics & Bias Detection", layout="wide")
st.title("🧠 AI Ethics & Bias Detection")
st.markdown("""
This dashboard evaluates **gender bias** in income prediction using the UCI Adult Dataset.
It compares model performance before and after applying **fairness mitigation techniques**.
""")

# ------------------------------
# DATA + PREPROCESSING
# ------------------------------
st.markdown("### 🔄 Loading and Processing Dataset...")

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)

sensitive_feature = df['sex'].copy()
df.drop('sex', axis=1, inplace=True)
y = df['income'].values
X = df.drop('income', axis=1)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
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

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------
# FAIRNESS METRICS (Before)
# ------------------------------
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

# ------------------------------
# FAIRNESS MITIGATION
# ------------------------------
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

# ------------------------------
# 📊 VISUALIZATIONS
# ------------------------------

# 1. Accuracy Comparison
st.markdown("### ✅ Model Accuracy Comparison")
fig1, ax1 = plt.subplots()
ax1.bar(["Original", "Mitigated"], [accuracy, accuracy_mitigated], color=["skyblue", "orange"])
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.6, 1)
st.pyplot(fig1)

# 2. Demographic Parity Metrics
st.markdown("### ⚖️ Demographic Parity (Fairness)")
fig2, ax2 = plt.subplots()
ax2.bar(["DP Difference", "DP Ratio"], [dp_diff, dp_ratio], color=["crimson", "seagreen"])
ax2.set_ylabel("Metric Value")
ax2.set_title("Original Model Fairness")
st.pyplot(fig2)

# 3. By-Group Fairness
st.markdown("### 👥 Accuracy by Gender Group (Original Model)")
fig3, ax3 = plt.subplots()
metric_frame.by_group.plot(kind="bar", ax=ax3, color=["purple"])
ax3.set_ylabel("Accuracy")
st.pyplot(fig3)

# ------------------------------
# SUMMARY
# ------------------------------
st.markdown("### 📌 Summary & Insights")
st.markdown(f"""
- **Initial Accuracy:** `{accuracy:.3f}`  
- **Mitigated Accuracy:** `{accuracy_mitigated:.3f}`  
- **Demographic Parity Difference (Before):** `{dp_diff:.3f}`  
- **Demographic Parity Ratio (Before):** `{dp_ratio:.3f}`  
- After applying fairness constraints, the model becomes more balanced, though with a small trade-off in accuracy.
""")


