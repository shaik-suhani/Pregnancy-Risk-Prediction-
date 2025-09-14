import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("data", exist_ok=True)
CSV_PATH = os.path.join("data", "pregnancy_risk.csv")

# Create a synthetic dataset if no CSV found
if not os.path.exists(CSV_PATH):
    np.random.seed(42)
    n = 500
    age = np.random.randint(18, 46, size=n)
    systolic = np.random.normal(115, 12, size=n).clip(80, 180).round().astype(int)
    diastolic = np.random.normal(75, 8, size=n).clip(50, 120).round().astype(int)
    bs = np.random.normal(5.5, 1.2, size=n).clip(2.5, 15).round(2)
    hr = np.random.normal(78, 8, size=n).clip(45, 120).round().astype(int)

    risk_score = (
        (age - 20) * 0.02 +
        (systolic - 120) * 0.03 +
        (diastolic - 80) * 0.02 +
        (bs - 5.5) * 0.4 +
        (hr - 80) * 0.02
    )

    def to_label(r):
        if r < -1: return "Low"
        elif r < 1.5: return "Medium"
        else: return "High"

    risk_level = [to_label(r) for r in risk_score]

    df = pd.DataFrame({
        "Age": age,
        "SystolicBP": systolic,
        "DiastolicBP": diastolic,
        "BS": bs,
        "HeartRate": hr,
        "RiskLevel": risk_level
    })
    df.to_csv(CSV_PATH, index=False)
    print(f"Sample dataset created at {CSV_PATH}")

# Load dataset and run EDA
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

print("\nRiskLevel counts:")
print(df['RiskLevel'].value_counts())

# Save visualization
os.makedirs("data/plots", exist_ok=True)
plt.figure(figsize=(6,4))
sns.countplot(x='RiskLevel', data=df)
plt.title("RiskLevel distribution")
plt.savefig("data/plots/risk_dist.png")
plt.close()
print("Saved plot to data/plots/risk_dist.png")