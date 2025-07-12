import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# App title
st.set_page_config(page_title="AI Ethics & Bias Dashboard", layout="wide")
st.title("🧠 AI Ethics & Bias Detection")
st.markdown("This app evaluates **gender bias** in income prediction using the UCI Adult Dataset and shows how fairness can be improved using AI mitigation strategies.")


# Split data
y = df['income'].values
X = df.drop('income', axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Build pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Initial fairness
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)

# Display metrics
st.subheader("📊 Initial Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc:.2f}")
col2.metric("DP Difference", f"{dp_diff:.3f}")
col3.metric("DP Ratio", f"{dp_ratio:.3f}")

# Plot income by gender
st.subheader("🔍 Bias Visualization: Income by Gender")
gender_income = pd.DataFrame({"sex": sens_test, "income": y_test})
fig, ax = plt.subplots()
sns.barplot(data=gender_income, x="sex", y="income", ax=ax)
ax.set_title("Proportion of >50K Income by Gender")
st.pyplot(fig)

# Mitigation
st.subheader("🧪 Bias Mitigation Using ExponentiatedGradient")
X_train_enc = model.named_steps['preprocessor'].transform(X_train)
X_test_enc = model.named_steps['preprocessor'].transform(X_test)

mitigator = ExponentiatedGradient(
    LogisticRegression(max_iter=1000),
    constraints=DemographicParity(),
    eps=0.01
)
mitigator.fit(X_train_enc, y_train, sensitive_features=sens_train)
y_pred_mitigated = mitigator.predict(X_test_enc)

acc_mit = accuracy_score(y_test, y_pred_mitigated)
dp_diff_mit = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sens_test)
dp_ratio_mit = demographic_parity_ratio(y_test, y_pred_mitigated, sensitive_features=sens_test)

col1, col2, col3 = st.columns(3)
col1.metric("🎯 Accuracy (Mitigated)", f"{acc_mit:.2f}")
col2.metric("DP Diff (Mitigated)", f"{dp_diff_mit:.3f}")
col3.metric("DP Ratio (Mitigated)", f"{dp_ratio_mit:.3f}")

# Visualization of mitigation impact
st.subheader("📈 Fairness Shift (Before vs After Mitigation)")
fig2, ax2 = plt.subplots()
ax2.scatter(dp_diff, dp_ratio, color='red', s=100, label='Before')
ax2.scatter(dp_diff_mit, dp_ratio_mit, color='green', s=100, label='After')
ax2.axvline(0, color='gray', linestyle='--')
ax2.axhline(1, color='gray', linestyle='--')
ax2.set_xlabel("DP Difference (Ideal: 0)")
ax2.set_ylabel("DP Ratio (Ideal: 1)")
ax2.set_title("Fairness Metrics Shift")
ax2.legend()
st.pyplot(fig2)

# Footer
st.markdown("---")
st.info("Developed with ❤️ using Streamlit + Fairlearn + Scikit-Learn")
