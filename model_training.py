import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("models", exist_ok=True)
np.random.seed(42)
n_samples = 4000

# 1. Dataset Generation
categories = ["Engine", "Transmission", "Electrical", "Body", "EV_Battery"]
part_choices = np.random.choice(categories, size=n_samples)

base_cost_map = {
    "Electrical": 12000,
    "Body": 18000,
    "Engine": 40000,
    "Transmission": 50000,
    "EV_Battery": 85000,
}
base_costs = np.array([base_cost_map[p] for p in part_choices])
amounts = np.random.exponential(scale=base_costs, size=n_samples) + 2500

df = pd.DataFrame({
    "claim_amount": amounts,
    "vehicle_age_months": np.random.randint(1, 60, n_samples),
    "mileage_at_claim": np.random.uniform(1000, 120000, n_samples),
    "past_claims_count": np.random.poisson(lam=1.1, size=n_samples),
    "parts_replaced_count": np.random.randint(1, 6, n_samples),
    "mechanic_tenure_months": np.random.randint(1, 48, n_samples),
    "region": np.random.choice(["North", "South", "East", "West"], size=n_samples),
    "dealer_type": np.random.choice(["Authorized", "Third-Party"], size=n_samples, p=[0.70, 0.30]),
    "part_category": part_choices,
    "mechanic_notes": np.random.choice(
        [
            "Routine scheduled maintenance. Replaced oil and filter. No abnormal wear.",
            "Normal wear and tear observed during periodic inspection.",
            "Minor gear slippage reported. Inspected and adjusted.",
            "Component burnt due to severe electrical surge. Demanded full replacement.",
            "Suspicious breakdown without submitting failed parts. Highly inflated cost.",
        ],
        size=n_samples,
        p=[0.35, 0.25, 0.15, 0.15, 0.10],
    ),
})

# 2. Risk Score Engine
cost_ratio = df["claim_amount"] / np.array([base_cost_map[p] for p in df["part_category"]])
risk_score = (
    (cost_ratio - 1.0) * 0.75
    + (df["past_claims_count"] * 0.4)
    + (df["dealer_type"] == "Third-Party") * 0.5
    + (df["parts_replaced_count"] * 0.2)
    - (df["mechanic_tenure_months"] / 36.0) * 0.3
    + (df["mechanic_notes"].str.contains("Suspicious|inflated", regex=True)) * 1.1
    + (df["mechanic_notes"].str.contains("surge|burnt", regex=True)) * 0.4
    + np.random.normal(0, 0.6, n_samples)
)

prob_target = 1 / (1 + np.exp(-(risk_score - 1.2)))
df["is_fraud"] = (prob_target > 0.50).astype(int)

# 3. Column Transformer Pipeline
num_cols = [
    "claim_amount",
    "vehicle_age_months",
    "mileage_at_claim",
    "past_claims_count",
    "parts_replaced_count",
    "mechanic_tenure_months",
]
cat_cols = ["region", "dealer_type", "part_category"]
text_col = "mechanic_notes"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("text", TfidfVectorizer(max_features=20, sublinear_tf=True), text_col),
    ]
)

X_all = preprocessor.fit_transform(df)
y = df["is_fraud"].values

# 4. Fit Calibrated Classifier
base_clf = LogisticRegression(C=0.15, penalty="l2", random_state=42, max_iter=1000)
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=5)
calibrated_clf.fit(X_all, y)

# 5. Fit Isolation Forest ONLY on Numeric Dimensions (Eliminates high-dimension noise)
scaler_num = StandardScaler()
X_numeric = scaler_num.fit_transform(df[num_cols])

iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=150)
iso_forest.fit(X_numeric)

# Compute dynamic 20th-percentile anomaly cutoff threshold
scores = iso_forest.decision_function(X_numeric)
anomaly_threshold = float(np.percentile(scores, 20))

# 6. Save Artifacts
joblib.dump(preprocessor, "models/preprocessor.joblib")
joblib.dump(calibrated_clf, "models/logistic_model.joblib")
joblib.dump(iso_forest, "models/anomaly_model.joblib")
joblib.dump(scaler_num, "models/scaler_num.joblib")
joblib.dump(anomaly_threshold, "models/anomaly_threshold.joblib")

print(f"Models Saved. Calibrated Anomaly Threshold: {anomaly_threshold:.4f}")