# 🛡️ ClaimTrace Engine: Smart Warranty Fraud Detection

ClaimTrace Engine is a tool built with **FastAPI**, **Streamlit**, and **Scikit-Learn** to check vehicle warranty repair claims and find fraudulent or abnormal requests quickly and accurately.

---

### 🇮🇳 Simple Analogy: Toll Plaza Checkpost
Think of this project like a **Highway Toll Plaza**:
* **FASTag Scanner (Classifier):** Checks the dealer, repair history, and mechanic notes to give an exact fraud risk percentage from 0% to 100%.
* **Weighbridge (Isolation Forest):** Checks the physical numbers (claim amount, mileage, vehicle age, and parts count) to see if something looks unusual or out of place.
* **Toll Gate Barrier:** Combines both checks together to give a clear, single result: **Genuine**, **Suspicious**, or **Fraud**.

---

## 📌 What Problem Does It Solve?

Checking warranty claims manually takes too much time, and basic computer models often make two big mistakes:
1. **Extreme Scores:** Giving only 0% or 100% scores with nothing in between, missing tricky borderline cases.
2. **Text Confusion:** Getting confused when long mechanic notes hide suspicious numbers.

ClaimTrace fixes this by:
* Using smooth probability calibration so scores reflect real risk (like 25%, 55%, or 85%).
* Separating numbers and text properly so strange numbers are always caught.
* Combining rules into one clear decision banner to avoid confusing messages.

---

## 🏗️ How It Works

1. **User Input:** Enter claim details and mechanic logs in the Streamlit web dashboard.
2. **FastAPI Backend:** Checks the inputs and runs the data through the machine learning pipeline.
3. **Preprocessing:** Numbers are scaled, categories are encoded, and mechanic notes are turned into TF-IDF text features.
4. **Dual Model Check:**
   * **Supervised Model:** Estimates the exact fraud probability score.
   * **Isolation Forest:** Scans numbers for weird or rare patterns.
5. **Final Result:** Displays a clean verdict, risk tier, and score progress bar on the screen.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (clean, interactive dashboard)
* **Backend:** FastAPI + Uvicorn (fast REST API gateway)
* **Machine Learning:** Scikit-Learn (Calibrated Logistic Regression & Isolation Forest)
* **Data Processing:** Pandas, NumPy, and Joblib

---

## 📂 Project Structure

```text
ClaimTrace-Engine/
├── models/
│   ├── anomaly_model.joblib        # Outlier detection model
│   ├── anomaly_threshold.joblib    # Cutoff threshold score
│   ├── logistic_model.joblib       # Calibrated fraud classifier
│   ├── preprocessor.joblib         # Combined data transformer
│   └── scaler_num.joblib           # Numeric feature scaler
├── app.py                          # Streamlit UI dashboard
├── main.py                         # FastAPI backend server
├── train.py                        # Model training script
└── requirements.txt                # Required Python libraries
