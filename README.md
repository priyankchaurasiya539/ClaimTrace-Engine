# 🛡️ ClaimTrace Engine: Hybrid Warranty Fraud & Anomaly Audit System

ClaimTrace Engine is an enterprise-grade automotive warranty audit platform powered by **FastAPI**, **Streamlit**, and **Scikit-Learn**. It integrates supervised probability calibration with unsupervised structural outlier detection to evaluate repair claims in real time.

---

### 🇮🇳 Daily Life Analogy: Highway Toll Plaza Multi-Check Barrier
Is system ko **National Highway FASTag & Automatic Weighbridge** ki tarah samjho:
* **FASTag Camera (`CalibratedClassifierCV`):** Claim history, dealer type, aur mechanic logs scan karke smooth probability score ($0\%$ to $100\%$) assign karta hai.
* **Overload Weighbridge (`IsolationForest`):** Physical numeric dimensions (Claim Amount, Mileage, Vehicle Age, Parts Count) ko evaluate karke baseline outlier audit karta hai.
* **Harmonized Boom Barrier:** Dono pipelines ko synchronize karke ek unambiguous decision (`GENUINE`, `SUSPICIOUS`, ya `FRAUD`) deliver karta hai.

---

## 📌 Problem Solved

Traditional warranty verification systems suffer from two core failure modes:
1. **Model Probability Saturation:** Unregularized classifiers collapse into extreme $0.00\%$ or $100.00\%$ predictions, eliminating nuanced risk assessment for borderline cases.
2. **Curse of Dimensionality in Anomaly Detection:** Feeding sparse TF-IDF text features into Isolation Forests dilutes distance metrics, causing severe outliers to slip through as normal patterns.

ClaimTrace eliminates these issues through **Sigmoid-calibrated soft-margins**, **isolated numeric anomaly trees**, and a **centralized 3-tier ensemble decision matrix**.

---

## 🏗️ Architecture & Data Flow
