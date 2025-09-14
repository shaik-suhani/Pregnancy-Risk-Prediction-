import os
import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF
from datetime import datetime
import io

# ------- Page config -------
st.set_page_config(page_title="MaternAI - Pregnancy Risk Predictor",
                   page_icon="🤰", layout="centered")

st.title("🤰  MaternAI: Pregnancy Risk Predictor")
st.write("Fill in the details below to predict pregnancy risk level.")

# ------- Load model -------
MODEL_PATH = os.path.join("models", "pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Please run python src/train.py first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]
feature_order = bundle["feature_order"]

# ------- Input form like screenshot -------
with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=15, max_value=50, value=25, step=1)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1)
        sugar = st.number_input("Blood Sugar (mg/dL)", min_value=60, max_value=300, value=100, step=1)

    with col2:
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1, format="%.2f")
        bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=24.0, step=0.1, format="%.2f")
        parity = st.number_input("No. of Previous Pregnancies", min_value=0, max_value=10, value=0, step=1)

    submitted = st.form_submit_button("🔍 Predict Risk")

def make_pdf_report(summary_df: pd.DataFrame, pred_label: str, confidence: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MaternAI - Pregnancy Risk Report", ln=1, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Patient Summary", ln=1)

    pdf.set_font("Arial", "", 11)
    for k, v in summary_df.iloc[0].items():
        pdf.cell(0, 7, f"{k}: {v}", ln=1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Prediction", ln=1)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Risk: {pred_label}", ln=1)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Confidence Scores:", ln=1)

    pdf.set_font("Arial", "", 11)
    for k in le.classes_:
        v = confidence.get(k, 0.0)
        pdf.cell(0, 7, f"{k}: {v:.3f}", ln=1)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ------- Prediction -------
if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "BloodPressure": bp,
        "BloodSugar": sugar,
        "Hemoglobin": hemoglobin,
        "BMI": bmi,
        "Parity": parity
    }])[feature_order]

    st.subheader("📋 Patient Summary")
    st.table(input_df)

    try:
        pred_idx = int(model.predict(input_df)[0])
        pred_label = le.inverse_transform([pred_idx])[0]
        probas = model.predict_proba(input_df)[0]
        confidence = {cls: float(probas[i]) for i, cls in enumerate(le.classes_)}

        st.subheader("📊 Prediction Result")
        if pred_label.lower().startswith("high"):
            st.error("⚠ High Pregnancy Risk Detected!")
        elif pred_label.lower().startswith("medium"):
            st.warning("⚠ Medium Pregnancy Risk Detected!")
        else:
            st.success("✅ Low Pregnancy Risk Detected!")

        # --- Generate PDF and provide download button ---
        pdf_bytes = make_pdf_report(input_df, pred_label, confidence)
        st.download_button(
            label="📄 Download Final Report (PDF)",
            data=pdf_bytes,
            file_name="Pregnancy_Risk_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")