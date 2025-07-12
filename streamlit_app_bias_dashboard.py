import streamlit as st
import numpy as np
import pandas as pd
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

# Streamlit UI Title
st.set_page_config(page_title="AI Ethics & Bias Dashboard", layout="centered")
st.title("🧠 AI Ethics & Bias Detection")
st.markdown("""
This interactive dashboard evaluates **gender bias** in income prediction using the UCI Adult Dataset. It demonstrates bias mitigation using Fairlearn.
""")

# Load dataset
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
df = X.copy()
df['income'] = y

# Clean dataset
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = (df['income'] == '>50K').astype(int)
sensitive_feature = df['sex'].copy()
df.drop(labels=['sex'], axis=1, inplace=True)

y = df['income'].values
X = df.drop(labels=['income'], axis=1)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# Train base model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Fairness metrics
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

# Bias Mitigation
X_train_enc = pipeline.named_steps['preprocessor'].transform(X_train)
X_test_enc = pipeline.named_steps['preprocessor'].transform(X_test)

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01
)
mitigator.fit(X_train_enc, y_train, sensitive_features=sens_train)
y_pred_mitigated = mitigator.predict(X_test_enc)

accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)
dp_diff_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sens_test)
dp_ratio_mitigated = demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=sens_test)

# Section: Summary Metrics
st.subheader("📊 Fairness Metrics Summary")
st.markdown("""
**Before and After Mitigation Comparison:**
""")
summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Demographic Parity Diff", "Demographic Parity Ratio"],
    "Before Mitigation": [accuracy, dp_diff, dp_ratio],
    "After Mitigation": [accuracy_mitigated, dp_diff_mitigated, dp_ratio_mitigated]
})
summary_df = summary_df.round(3)
st.dataframe(summary_df, use_container_width=True)

# Section: Visualization
st.subheader("📉 Visual Comparison")
fig, ax = plt.subplots(figsize=(6, 4))
labels = summary_df["Metric"]
before = summary_df["Before Mitigation"]
after = summary_df["After Mitigation"]
width = 0.35
x = np.arange(len(labels))
ax.bar(x - width / 2, before, width, label='Before', color='salmon')
ax.bar(x + width / 2, after, width, label='After', color='seagreen')
ax.set_ylabel('Metric Value')
ax.set_title('Fairness Metrics Before vs After Mitigation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
st.pyplot(fig)

# Section: Accuracy by Gender
st.subheader("👩‍💼 Accuracy by Gender (Pre-Mitigation)")
by_group_df = metric_frame.by_group.round(3).reset_index()
st.dataframe(by_group_df.rename(columns={"index": "Group"}))

fig2, ax2 = plt.subplots(figsize=(4, 3))
metric_frame.by_group.plot(kind='bar', ax=ax2, color='skyblue', legend=False)
ax2.set_title("Accuracy by Gender")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0.5, 1.0)
plt.xticks(rotation=0)
st.pyplot(fig2)

# End
st.success("✅ Evaluation Complete. Mitigation successfully applied.")
