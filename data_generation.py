import numpy as np
import pandas as pd

# 1. Basic Setup
n_rows = 50000
np.random.seed(42)

# 2. Equipment & Dealer IDs
equipment_ids = ["EQ-" + str(i) for i in range(1000, 1000 + n_rows)]
dealer_ids = ["DLR-" + str(np.random.randint(101, 150)) for _ in range(n_rows)]

# 3. Categorical Text Columns (To be One-Hot Encoded later in Pandas)
# Vehicle & Machinery Categories
equipment_types = [
    "Truck",
    "Bus",
    "Drilling Vehicle",
    "Bike",
    "Car",
    "Tractor",
    "Harvester",
    "Bulldozers",
    "Double Drum Rollers",
    "Cranes",
    
]
equipment_type = np.random.choice(equipment_types, size=n_rows)

# Region Categories
regions = ["North", "South", "East", "West", "Central"]
region = np.random.choice(regions, size=n_rows)

# Fuel Categories
fuel_types = ["Diesel", "Petrol", "EV", "CNG"]
fuel_type = np.random.choice(fuel_types, size=n_rows)

# Severity Categories
claim_severities = ["Low", "Medium", "High", "Critical"]
claim_severity = np.random.choice(claim_severities, size=n_rows)

# 4. Streamlit User Input Features & Telemetry
runtime_hrs = np.random.randint(1, 48, size=n_rows)
past_claims = np.random.randint(0, 10, size=n_rows)
claim_amount = np.random.randint(5000, 300000, size=n_rows)
part_replacement_cost = np.random.randint(2000, 180000, size=n_rows)

# 5. Features with Missing Data (NaN) for Pandas Practice
operating_hours = np.random.randint(100, 100000, size=n_rows).astype(float)
last_service_days_ago = np.random.randint(5, 365, size=n_rows).astype(float)

# Inject 8% random missing values (NaN)
mask_op = np.random.rand(n_rows) < 0.08
operating_hours[mask_op] = np.nan

mask_service = np.random.rand(n_rows) < 0.08
last_service_days_ago[mask_service] = np.nan

# 6. Basic Mechanic Text Notes (For NLP Processing)
sample_notes = [
    # Genuine / Routine Service Notes
    "Regular scheduled maintenance and oil filter change completed.",
    "Minor wear and tear on brake pads, replaced with original parts.",
    "Engine oil topped up during routine service checkup.",
    "Normal dust accumulation in air filter cleaned thoroughly.",
    "Routine hydraulic fluid replacement done during annual service.",
    
    # Genuine Mechanical Failures
    "Hydraulic pressure dropped suddenly during normal operations.",
    "Engine overheated due to radiator fan belt failure.",
    "Minor fuel line leak detected and repaired under normal warranty.",
    "Gearbox shifting hard during low gear acceleration.",
    "Brake caliper sticking during heavy duty operation.",
    
    # Suspicious / Fraud Risk Notes (High failure indicators)
    "Engine seized completely after running non-stop without cooling pause.",
    "Piston rod broken due to severe overloading beyond capacity limits.",
    "Clutch plate burnt out due to continuous high speed driving.",
    "Transmission failure caused by using non-recommended cheap oil.",
    "Hydraulic cylinder cracked under illegal load weight.",
]
mechanic_notes = np.random.choice(sample_notes, size=n_rows)

# 7. Simple Rule for Fraud Label (1 or 0)
is_fraud = np.where(
    (runtime_hrs > 15) & (claim_amount > 150000) | (past_claims > 4), 1, 0
)

# 8. Create DataFrame
df = pd.DataFrame(
    {
        "equipment_id": equipment_ids,
        "dealer_id": dealer_ids,
        "equipment_type": equipment_type,  # String -> Categorical
        "region": region,  # String -> Categorical
        "fuel_type": fuel_type,  # String -> Categorical
        "claim_severity": claim_severity,  # String -> Categorical
        "operating_hours": operating_hours,  # Has NaNs
        "runtime_hrs": runtime_hrs,
        "past_claims": past_claims,
        "last_service_days_ago": last_service_days_ago,  # Has NaNs
        "claim_amount": claim_amount,
        "part_replacement_cost": part_replacement_cost,
        "mechanic_notes": mechanic_notes,  # Text for NLP
        "is_fraud": is_fraud,  # Target variable
    }
)

# 9. Save to CSV
df.to_csv("warranty.csv", index=False)
print("Dataset created successfully with new vehicle types and text categories!")