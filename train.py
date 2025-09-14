import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "pregnancy_risk.csv")
MODEL_PATH = os.path.join("models", "pipeline.joblib")
os.makedirs("models", exist_ok=True)

# ---- Load & validate ----
df = pd.read_csv(DATA_PATH)
needed_cols = ["Age", "BloodPressure", "BloodSugar", "Hemoglobin", "BMI", "Parity", "Risk"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df.dropna(subset=needed_cols).reset_index(drop=True)

# ---- Features & target ----
feature_order = ["Age", "BloodPressure", "BloodSugar", "Hemoglobin", "BMI", "Parity"]
X = df[feature_order].astype(float)
y_raw = df["Risk"].astype(str)

# ---- Encode target ----
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ---- Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Pipeline & search ----
pipe = Pipeline([
    ("scaler", StandardScaler()),            # harmless before trees; keeps interface consistent
    ("clf", RandomForestClassifier(
        random_state=42, n_estimators=200, n_jobs=-1
    )),
])

param_grid = {
    "clf__max_depth": [6, 12, None],
    "clf__min_samples_split": [2, 5],
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("Best params:", grid.best_params_)

# ---- Test metrics ----
y_pred = best.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---- Save everything we need for inference ----
joblib.dump(
    {
        "model": best,
        "label_encoder": le,
        "feature_order": feature_order,
        "target_name": "Risk",
        "version": "1.0.0"
    },
    MODEL_PATH
)
print("✅ Saved model to", MODEL_PATH)