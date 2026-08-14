import requests
import streamlit as st
import threading
import uvicorn
from FastAPI import app as fastapi_app  # your FastAPI app object

def run_fastapi():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

@st.cache_resource
def start_fastapi_server():
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    return thread

start_fastapi_server()

st.set_page_config(
    page_title="ClaimTrace Engine | Warranty Audit",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ClaimTrace Engine")
st.caption("Enterprise Industrial Warranty Fraud & Outlier Detection Pipeline")
st.markdown("---")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Operational Parameters")
    claim_amount = st.number_input(
        "Claim Amount (₹)", min_value=0.0, value=4500.0, step=500.0
    )
    vehicle_age_months = st.number_input(
        "Vehicle Age (Months)", min_value=0, value=14, step=1
    )
    mileage_at_claim = st.number_input(
        "Mileage at Claim (Km)", min_value=0.0, value=18000.0, step=1000.0
    )
    past_claims_count = st.number_input(
        "Past Claims Count", min_value=0, value=0, step=1
    )

with col_right:
    st.subheader("⚙️ Maintenance & Dealer Metadata")
    parts_replaced_count = st.number_input(
        "Parts Replaced Count", min_value=1, value=1, step=1
    )
    mechanic_tenure_months = st.number_input(
        "Mechanic Tenure (Months)", min_value=0, value=36, step=1
    )
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    dealer_type = st.selectbox("Dealer Type", ["Authorized", "Third-Party"])
    part_category = st.selectbox(
        "Part Category",
        ["Engine", "Transmission", "Electrical", "Body", "EV_Battery"],
    )

st.subheader("📝 Mechanic Diagnostic Notes")
mechanic_notes = st.text_area(
    "Enter shop-floor observation logs:",
    value="Routine scheduled maintenance. Replaced oil and filter. No abnormal wear.",
    height=90,
)

st.markdown("---")

col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
with col_b2:
    analyze_clicked = st.button(
        "🚀 Analyze Claim", type="primary", use_container_width=True
    )

if analyze_clicked:
    payload = {
        "claim_amount": claim_amount,
        "vehicle_age_months": vehicle_age_months,
        "mileage_at_claim": mileage_at_claim,
        "past_claims_count": past_claims_count,
        "parts_replaced_count": parts_replaced_count,
        "mechanic_tenure_months": mechanic_tenure_months,
        "region": region,
        "dealer_type": dealer_type,
        "part_category": part_category,
        "mechanic_notes": mechanic_notes,
    }

    with st.spinner("Executing Harmonized Inference..."):
        try:
            response = requests.post(
                FASTAPI_URL,
                json=payload,
                timeout=10,
                proxies={"http": None, "https": None},
            )

            if response.status_code == 200:
                result = response.json()
                st.subheader("📊 Audit Verdict & Probability Matrix")
                c1, c2, c3 = st.columns(3)

                # Card 1: Harmonized Verdict
                with c1:
                    with st.container(border=True):
                        st.caption("CLASSIFICATION VERDICT")
                        if result["verdict"] == "FRAUD DETECTED":
                            st.markdown("### 🚨 FRAUD DETECTED")
                        elif result["verdict"] == "SUSPICIOUS / AUDIT":
                            st.markdown("### ⚠️ SUSPICIOUS CLAIM")
                        else:
                            st.markdown("### ✅ GENUINE CLAIM")
                        st.caption(result["action_notes"])

                # Card 2: Probability Score
                with c2:
                    with st.container(border=True):
                        st.caption("FRAUD PROBABILITY SCORE")
                        prob_pct = result["fraud_probability"] * 100
                        st.metric(
                            label="Score",
                            value=f"{prob_pct:.2f}%",
                            delta=(
                                "High"
                                if prob_pct >= 70
                                else ("Medium" if prob_pct >= 40 else "Normal")
                            ),
                            delta_color=(
                                "inverse" if prob_pct >= 40 else "normal"
                            ),
                        )
                        st.progress(max(0, min(100, int(prob_pct))))

                # Card 3: Isolation Engine
                with c3:
                    with st.container(border=True):
                        st.caption("ISOLATION ENGINE AUDIT")
                        if result["is_anomaly"]:
                            st.error("🔴 OUTLIER PATTERN")
                            st.caption(
                                f"Score: {result['anomaly_score']} (Outlier)"
                            )
                        else:
                            st.success("🟢 NORMAL PATTERN")
                            st.caption(
                                f"Score: {result['anomaly_score']} (Normal)"
                            )

                # Bottom Harmonized Banner
                risk_tier = result["risk_tier"]
                if "High" in risk_tier:
                    st.error(f"⚠️ **Action Tier:** {risk_tier}")
                elif "Medium" in risk_tier:
                    st.warning(f"⚡ **Action Tier:** {risk_tier}")
                else:
                    st.info(f"ℹ️ **Action Tier:** {risk_tier}")

            else:
                st.error(
                    f"Backend Error [{response.status_code}]: {response.text}"
                )

        except Exception as e:
            st.error(f"❌ Execution Error: {str(e)}")