<!-- Banner Image -->
<p align="center">
  <img src="https://raw.githubusercontent.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation/refs/heads/master/AI%20ETHICS%20BANNER.png" alt="AI Ethics and Bias Evaluation Banner" style="max-width: 100%; border-radius: 12px;">
</p>

<br><h1 align="center">🚀 AI Ethics and Bias Evaluation – End-to-End Pipeline</h1><br>

This project uses Fairlearn to detect and mitigate bias in a logistic regression classifier trained on the UCI Adult dataset. It showcases demographic parity, ethical trade-offs, and deploys an interactive Streamlit dashboard to explore results in real time.


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python 3.8+"></a>
  <a href="https://fairlearn.org/"><img src="https://img.shields.io/badge/Fairlearn-Enabled-success?logo=scikit-learn" alt="Fairlearn Enabled"></a>
  <a href="https://github.com/hamaylzahid/ResponsibleAI_BiasCheck">
  <img src="https://img.shields.io/github/repo-size/hamaylzahid/ResponsibleAI_BiasCheck?color=lightgrey" alt="Repo Size"></a>
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation/commits">
    <img src="https://img.shields.io/github/last-commit/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation?color=blue" alt="Last Commit"> </a>
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation/stargazers">
    <img src="https://img.shields.io/github/stars/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation?style=social" alt="GitHub Stars">
  </a>
  <img src="https://img.shields.io/badge/Project%20Status-Completed-brightgreen" alt="Project Status">
</p>


> **A complete, product**

  ---
<p align="center">
  <h3>🎯 Live Demo – Try the App</h3>
  <em>Experience the full project in action through an interactive Streamlit dashboard.</em><br>
  This app lets users <b>analyze gender bias</b> in income prediction and <b>visualize fairness metrics</b> interactively.
  <br><br>
  <a href="https://responsibleaibiascheck-h6p9vyyjmsmyemycpsidaz.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="View in Streamlit">
  </a>
  <br><br>
  🔗 <b>Live App:</b><br>
  <a href="https://responsibleaibiascheck-h6p9vyyjmsmyemycpsidaz.streamlit.app/" target="_blank">
    responsibleaibiascheck.streamlit.app
  </a>
  <br><br>
  ⚙️ <i>Deployment was handled using <code>streamlit run</code> for local development and a <code>requirements.txt</code> file for dependency management. The app is hosted on Streamlit Community Cloud.</i>
</p>



---

<br><h2 align="center">📌 Features</h2><br>

- ✅ Bias evaluation with **Demographic Parity**
- ✅ Mitigation using **Exponentiated Gradient** from `Fairlearn`
- ✅ Preprocessing pipeline with `scikit-learn`
- ✅ Visual metrics for fairness before and after mitigation
- ✅ Clean, modular code with real-world deployability

---

<br><h2 align="center">📖 Table of Contents</h2><br>

- [🧠 Project Overview](#-project-overview)  
- [🎯 Objectives](#-objectives)  
- [📊 Dataset Info](#-dataset-info)  
- [🛠️ Workflow Breakdown](#️-workflow-breakdown)  
- [📈 Metrics & Results](#-metrics--results)  
- [🔥 Bias Mitigation Strategy](#-bias-mitigation-strategy)  
- [⚙️ Setup & Installation](#️-setup--installation)  
- [📚 Concepts Covered](#-key-concepts-covered)  
- [🚀 Deployment Notes](#-recommendations-for-ethical-deployment)  
- [🙏 Acknowledgments](#-acknowledgments)  
- [📚 Core Libraries Used](#-core-libraries-used)  
- [🤝 Contact & Contribution](#-contact--contribution)  
- [📜 License](#-license)
 

---

<br><h2 align="center">🧠 Project Overview</h2><br>

Fairness in AI is not optional — it’s essential. This repo showcases a principled approach to identifying and mitigating bias in a classification pipeline. The goal is to build trustable and transparent systems that uphold ethical standards while maintaining performance.

> 🧭 **Why This Matters**  
<br>Bias in machine learning systems isn’t just a theoretical problem — it has real-world consequences, affecting fairness in hiring, lending, healthcare, and justice. This project is a step toward building **responsible AI** that doesn't just optimize metrics but respects the **human values** behind the data. By combining technical precision with ethical intention, we aim to make AI systems **trustworthy, transparent, and fair for everyone**.<br>


---

<br><h2 align="center">🎯 Objectives</h2><br>

- Clean and preprocess the UCI Adult dataset  
- Train a **Logistic Regression** classifier  
- Evaluate group-wise fairness using **Fairlearn metrics**  
- Apply **Exponentiated Gradient** mitigation  
- Compare performance before and after debiasing  
- Provide actionable recommendations for deployment

---

<br><h2 align="center">📊 Dataset Info</h2><br>

- **Source**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
- **Task**: Predict if income > $50K  
- **Sensitive Attribute**: `sex`  
- **Target**: `income` (binary: `>50K` → `1`, else `0`)

---

<br><h2 align="center">🛠️ Workflow Breakdown</h2><br>

1. **Load & Clean Data**  
2. **Split into Train/Test Sets**  
3. **Preprocessing Pipeline** (OneHotEncoder + StandardScaler)  
4. **Train Model**  
5. **Evaluate Bias**  
6. **Apply Exponentiated Gradient Mitigation**  
7. **Compare Fairness Before vs After**

---

<br><h2 align="center">📈 Metrics & Results</h2><br>

| Metric                           | Before Mitigation | After Mitigation |
|----------------------------------|-------------------|------------------|
| Accuracy                         | High              | Slightly Lower   |
| Demographic Parity Difference    | High              | ↓ Closer to 0    |
| Demographic Parity Ratio         | ≠ 1 (biased)       | ✅ Approaching 1 |

> ⚠️ *Ethical Trade-Off*: Mitigation slightly reduces accuracy but enhances fairness significantly.


<br><h2 align="center">📌 Key Learnings</h2><br>

- Fairness can **improve drastically** using constraint-based debiasing like Exponentiated Gradient.  
- Mitigation always involves a **trade-off between accuracy and equity**.  
- Visual tools like confusion matrices and demographic parity plots help **communicate ethical trade-offs** clearly.  
- Continuous monitoring and **stakeholder communication** are essential when deploying fair AI models in the real world.<br>


### 🧮 MetricFrame Output (Fairlearn)

<br>This section demonstrates how we used Fairlearn's `MetricFrame` to evaluate group-wise fairness (accuracy per gender group):<br>

```python
from fairlearn.metrics import MetricFrame, accuracy_score

metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

print(metric_frame.by_group)
```

<details>
<summary>🖥️ Sample Output</summary>

<pre>
        accuracy
sex             
Female  0.930088
Male    0.812737
</pre>
</details>


<br><b> 🎯 Visual Comparison: Fairness & Accuracy<b><br>

<p align="center">
  <img src="visuals/accuracy&fairness%20before%20&after.png" width="600" alt="Fairness Accuracy Comparison">
</p>

---

<br><b> 🔳 Confusion Matrices (Before vs After)<b><br>

<p align="center">
  <img src="visuals/confustion%20matrix%20before&after.png" width="600" alt="Confusion Matrix">
</p>

> 💡 **Interpretation**:  
> The confusion matrices above show the distribution of predictions before and after mitigation.  
> After bias mitigation, the model becomes **more fair** by reducing false positives for the underrepresented group,  
> even if accuracy slightly drops.

---

<br><b> ⚖️ Fairness vs Bias Trade-off (ε)<b><br>

<p align="center">
  <img src="visuals/fairness&bias%20trade%20off.png" width="600" alt="Fairness vs Bias Tradeoff">
</p>

---

<br><h2 align="center">🔥 Bias Mitigation Strategy</h2><br>

- Used `Fairlearn.reductions.ExponentiatedGradient`  
- Applied **DemographicParity** constraint  
- Fine-tuned `eps` parameter for fairness-performance tradeoff  
- Compared results using `MetricFrame`, `demographic_parity_difference`, and `ratio`

---

<br><h2 align="center">⚙️ Setup & Installation</h2><br>


# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install numpy pandas scikit-learn fairlearn


---


<br><h2 align="center">Expected Output</h2><br>

✅Initial Accuracy & Fairness Metrics

✅Mitigated Model Performance

✅Summary of Recommendations

---



<br><h2 align="center">🧠 Key Concepts Covered</h2><br>

> These foundational ideas drive the project's ethical core and technical strength:

- **🔄 Demographic Parity**  
  Ensures that predicted outcomes are distributed equally across sensitive demographic groups, helping to identify hidden biases.

- **📉 Exponentiated Gradient Mitigation**  
  A reduction-based debiasing strategy that trains multiple classifiers under fairness constraints to balance accuracy and ethics.

- **📊 Fairlearn Metrics & MetricFrame**  
  Provides robust tools to compute and analyze fairness metrics like **Demographic Parity Difference** and **Ratio** at group levels.

---

<br><h2 align="center">📢 Recommendations for Ethical Deployment</h2><br>

> Building fair AI isn't a one-time fix — it's a continuous responsibility. Here’s what to keep in mind:

- 🎯 **Fine-Tune Hyperparameters (`eps`)**  
  Balance the trade-off between model performance and fairness by adjusting mitigation tolerance levels.

- 🧬 **Explore Alternative Techniques**  
  Consider post-processing (equalized odds) or pre-processing (reweighing) strategies for different fairness objectives.

- 💬 **Stakeholder Communication**  
  Clearly document fairness trade-offs, rationale, and mitigation outcomes when presenting to non-technical stakeholders.

- 🛡️ **Real-Time Monitoring**  
  Implement fairness dashboards or logging pipelines to detect drift and bias in production environments.

---

<br><h2 align="center">🙏 Acknowledgments</h2><br>

This project utilizes the [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) — a classic benchmark for fairness in income prediction tasks.  
Big thanks to the **Fairlearn** community for making fairness metrics and mitigation tools accessible and open-source.

---

<br><h2 align="center">📚 Core Libraries Used</h2><br>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Fairlearn-005571?style=for-the-badge&logo=python&logoColor=white" alt="Fairlearn"/>
</p>

---

<br><h2 align="center">🤝 Contact & Contribution</h2><br>

Have feedback, ideas, or want to collaborate?

- 📧 **Email**: [maylzahid588@gmail.com](mailto:maylzahid588@gmail.com)  
- 🌟 Star this repo to support the work  
- 🤝 Fork and PRs welcome!

---

<br><h2 align="center">📜 License</h2><br>


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation/commits/master"><img src="https://img.shields.io/github/last-commit/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation?color=blue" alt="Last Commit"></a>
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation"><img src="https://img.shields.io/github/repo-size/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation?color=lightgrey" alt="Repo Size"></a>
</p>


This project is licensed under the **MIT License** – free to use, modify, and distribute.

**✅ Project Status:** Completed and ready for portfolio showcase  
**🧾 License:** MIT – [View License »](LICENSE)

---
<br><br>

<p align="center" style="font-family:Segoe UI, sans-serif;">
  <img src="https://img.shields.io/badge/Built%20with-Python-blue?style=flat-square&logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/Fairlearn-Ethical%20AI-005571?style=flat-square&logo=justice&logoColor=white" alt="Fairlearn Badge" />
</p>

<p align="center">
  <b>Crafted with purpose & precision</b> ⚖️  
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •  
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email Badge" />
  </a>
  •  
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/InternIntelligence_AIEthicsandBiasEvaluation/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Start%20Building-2ea44f?style=flat-square&logo=github" alt="Fork Project Badge" />
  </a>
</p>


<p align="center">
  <sub><i>Inspired by fairness. Driven by responsibility. Designed to make a difference.</i></sub>
</p>
<p align="center">
  🔍 <b>Use this project to showcase your commitment to Responsible AI</b>  
  <br>
  📦 Clone it, run it, improve it — and advocate for fairness in ML systems.
</p>

