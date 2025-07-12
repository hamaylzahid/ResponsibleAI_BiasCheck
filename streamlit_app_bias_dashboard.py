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

# -------------------------------------------
# PAGE CONFIG & TITLE
# -------------------------------------------
st.set_page_config(page_title="AI Bias Evaluation", layout="centered")
st.title("🤖 AI Ethics & Bias Evaluation")
st.markdown("""
This interactive app performs **gender bias detection and fairness evaluation** in income prediction using the **UCI Adult Dataset**.

💡 **Key Highlights**:
- Uses **Logistic Regression** to predict whether an individual's income exceeds \$50K.
- Evaluates fairness across **sensitive attributes** (gender in this case).
- Applies **bias mitigation** using `Fairlearn`’s **Exponentiated Gradient algorithm** with a **Demographic Parity constraint**.
- Shows how **accuracy and fairness trade-offs** occur when deploying AI models in sensitive applications.

📊 Visual and tabular metrics help you understand:
- How well your model performs overall.
- Whether it behaves **differently for different genders**.
- What changes after **fairness constraints are applied**.

This project demonstrates how **responsible AI practices** can help reduce unfair bias in real-world machine learning pipelines.
""")


# -------------------------------------------
# LOAD DATASET
# -------------------------------------------
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)

sensitive_feature = df['sex']
df = df.drop(columns=['sex'])  # exclude from training
y = df['income'].values
X = df.drop(columns=['income'])

# -------------------------------------------
# SPLIT & PIPELINE
# -------------------------------------------
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# -------------------------------------------
# METRICS (Before Mitigation)
# -------------------------------------------
acc_before = accuracy_score(y_test, y_pred)
dp_diff_before = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio_before = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

mf = MetricFrame(metrics={"accuracy": accuracy_score}, y_true=y_test, y_pred=y_pred, sensitive_features=sens_test)

# -------------------------------------------
# APPLY MITIGATION
# -------------------------------------------
X_train_tf = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_tf = pipeline.named_steps['preprocessor'].transform(X_test)

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01
)

mitigator.fit(X_train_tf, y_train, sensitive_features=sens_train)
y_pred_mit = mitigator.predict(X_test_tf)

acc_after = accuracy_score(y_test, y_pred_mit)
dp_diff_after = demographic_parity_difference(y_test, y_pred_mit, sensitive_features=sens_test)
dp_ratio_after = demographic_parity_ratio(y_test, y_pred_mit, sensitive_features=sens_test)

# -------------------------------------------
# VISUALIZATIONS
# -------------------------------------------

st.markdown("### 🎯 Accuracy Comparison")
fig1, ax1 = plt.subplots()
ax1.bar(["Before", "After"], [acc_before, acc_after], color=["#007acc", "#f39c12"])
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.6, 1.0)
st.pyplot(fig1)

st.markdown("### ⚖️ Demographic Parity Metrics")
fig2, ax2 = plt.subplots()
bars = ax2.bar(
    ["DP Diff (Before)", "DP Diff (After)", "DP Ratio (Before)", "DP Ratio (After)"],
    [dp_diff_before, dp_diff_after, dp_ratio_before, dp_ratio_after],
    color=["#e74c3c", "#27ae60", "#e67e22", "#2ecc71"]
)
ax2.axhline(0 if max([dp_diff_before, dp_diff_after]) < 1 else 1, linestyle='--', color='gray')
ax2.set_ylabel("Metric Value")
ax2.set_ylim(0, max(dp_ratio_before, dp_ratio_after) + 0.5)
st.pyplot(fig2)

st.markdown("### 👥 Accuracy by Gender (Before Mitigation)")
fig3, ax3 = plt.subplots()
mf.by_group.plot(kind='bar', ax=ax3, color='#8e44ad')
ax3.set_ylabel("Accuracy")
ax3.set_title("Fairness by Group")
st.pyplot(fig3)

# -------------------------------------------
# SUMMARY
# -------------------------------------------
st.markdown("### 📝 Summary")
st.markdown(f"""
- **Initial Accuracy:** `{acc_before:.3f}`  
- **Fair Accuracy (After Mitigation):** `{acc_after:.3f}`  
- **DP Difference (Before → After):** `{dp_diff_before:.3f} → {dp_diff_after:.3f}`  
- **DP Ratio (Before → After):** `{dp_ratio_before:.3f} → {dp_ratio_after:.3f}`  
- **Insight:** After applying bias mitigation, fairness improves with a slight trade-off in accuracy.
""")
