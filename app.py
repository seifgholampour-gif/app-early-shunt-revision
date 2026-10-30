import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_FILE = "ShuntFailure_FullCatBoost_3Cause_BinaryGrades_artifact.joblib"


@st.cache_resource
def load_artifact():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARTIFACT_FILE)
    return joblib.load(path)


def calibrated_class_probs(artifact, raw):
    cols = [
        np.asarray(cal.predict(raw[:, j]), dtype=float)
        for j, cal in enumerate(artifact["class_calibrators"])
    ]
    p = np.column_stack(cols)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def yes_no(label, default=False):
    return int(st.toggle(label, value=default))


def grade_name(risk, cutoff):
    return "High" if float(risk) >= float(cutoff) else "Low"


def grade_card(title, risk, cutoff):
    grade = grade_name(risk, cutoff)
    st.metric(title, f"{100 * risk:.2f}%", delta=f"{grade} risk")
    st.caption(f"Low/High cutoff: {100 * cutoff:.2f}%")


def main():
    st.set_page_config(
        page_title="30-Day Shunt Failure Cause-Specific Risk Grades",
        page_icon="🧠",
        layout="wide",
    )
    artifact = load_artifact()
    metadata = artifact["input_metadata"]

    st.title("30-Day Shunt Failure Cause-Specific Risk Grading System")

    patient = {}
    with st.sidebar:
        st.header("Patient inputs")
        patient["corrected_age_months"] = st.number_input(
            "Corrected age (months)",
            min_value=float(metadata["corrected_age_months"]["min"]),
            max_value=float(metadata["corrected_age_months"]["max"]),
            value=float(metadata["corrected_age_months"]["median"]),
            step=0.1,
        )
        patient["weight_kg"] = st.number_input(
            "Weight (kg)",
            min_value=float(metadata["weight_kg"]["min"]),
            max_value=float(metadata["weight_kg"]["max"]),
            value=float(metadata["weight_kg"]["median"]),
            step=0.1,
        )
        categorical_inputs = [
            ("sex", "Sex"),
            ("race", "Race"),
            ("hispanic_ethnicity", "Ethnicity"),
            ("prematurity_status", "Prematurity status"),
            ("hydrocephalus_etiology", "Hydrocephalus etiology"),
            ("index_procedure_urgency", "Index procedure urgency"),
        ]
        for column, label in categorical_inputs:
            patient[column] = st.selectbox(label, metadata[column]["levels"], index=0)

        st.subheader("Prior CSF procedures / conditions")
        patient["prior_evd"] = yes_no("Prior EVD")
        patient["prior_temporization_device"] = yes_no("Prior temporization device")
        patient["prior_etv_or_etv_cpc"] = yes_no("Prior ETV or ETV/CPC")
        patient["cerebral_palsy_or_neuromuscular"] = yes_no(
            "Cerebral palsy / neuromuscular condition"
        )
        patient["respiratory_status"] = yes_no("Respiratory condition/risk")
        patient["cardiac_condition_or_risk"] = yes_no("Cardiac condition/risk")

        st.subheader("Adjuncts")
        patient["adjunct_aic"] = yes_no("AIC")
        patient["adjunct_programmable"] = yes_no("Programmable valve")
        patient["adjunct_image_guidance"] = yes_no("Image guidance")

    X = pd.DataFrame([patient])[artifact["features"]]
    X_encoded = artifact["preprocessor"].transform(X)
    raw = np.asarray(artifact["model"].predict_proba(X_encoded), dtype=float)
    raw = raw / raw.sum(axis=1, keepdims=True)
    probabilities = calibrated_class_probs(artifact, raw)[0]
    cutoffs = artifact["class_grade_cutoffs"]

    st.subheader("Cause-specific 30-day risk grades")
    col1, col2, col3 = st.columns(3)
    with col1:
        grade_card(
            "Mechanical / obstruction",
            float(probabilities[1]),
            float(cutoffs["1"]["cutoff"]),
        )
    with col2:
        grade_card(
            "Infection",
            float(probabilities[2]),
            float(cutoffs["2"]["cutoff"]),
        )
    with col3:
        grade_card(
            "Other / loculation",
            float(probabilities[3]),
            float(cutoffs["3"]["cutoff"]),
        )

    st.info("Risk grades are independent across failure mechanisms.")
    st.warning("For research use only. Not a stand-alone treatment decision rule.")


if __name__ == "__main__":
    main()
