import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join("models", "pipeline.joblib")
DATA_PATH = os.path.join("data", "pregnancy_risk.csv")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
le = saved["label_encoder"]
feature_order = saved["feature_order"]

df = pd.read_csv(DATA_PATH)
X = df[feature_order].astype(float)
y_true_raw = df["Risk"].astype(str)
y_true = le.transform(y_true_raw)

y_pred = model.predict(X)
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("data/plots", exist_ok=True)
plt.savefig("data/plots/confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved plot to data/plots/confusion_matrix.png")