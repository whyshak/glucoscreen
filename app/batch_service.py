"""
batch_service.py — Batch prediction, validation, template generation, and export service.

Provides robust processing for bulk CSV and Excel datasets using the GlucoScreen
RBF SVM model and standard scaler.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Feature Definitions & Order ──────────────────────────────────────────────
# Exact feature order expected by the trained SVM model
FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

# Human-readable labels for features
FEATURE_LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check (5 yrs)",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoking History",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Intake",
    "Veggies": "Vegetable Intake",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor Due to Cost",
    "GenHlth": "General Health (1-5)",
    "MentHlth": "Mental Health Days (0-30)",
    "PhysHlth": "Physical Health Days (0-30)",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex (0=F, 1=M)",
    "Age": "Age Group (1-13)",
    "Education": "Education Level (1-6)",
    "Income": "Income Level (1-8)",
}

# Features that need standard scaling before model inference
SCALED_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

# Column aliases for flexible user input parsing
COLUMN_ALIASES: Dict[str, str] = {
    "highbp": "HighBP",
    "high_bp": "HighBP",
    "highbloodpressure": "HighBP",
    "bp": "HighBP",
    "highchol": "HighChol",
    "high_chol": "HighChol",
    "highcholesterol": "HighChol",
    "cholcheck": "CholCheck",
    "chol_check": "CholCheck",
    "cholesterolcheck": "CholCheck",
    "bmi": "BMI",
    "bodymassindex": "BMI",
    "smoker": "Smoker",
    "smoking": "Smoker",
    "stroke": "Stroke",
    "heartdiseaseorattack": "HeartDiseaseorAttack",
    "heartdisease": "HeartDiseaseorAttack",
    "heart_disease": "HeartDiseaseorAttack",
    "heart_disease_or_attack": "HeartDiseaseorAttack",
    "physactivity": "PhysActivity",
    "phys_activity": "PhysActivity",
    "physicalactivity": "PhysActivity",
    "physical_activity": "PhysActivity",
    "fruits": "Fruits",
    "fruit": "Fruits",
    "veggies": "Veggies",
    "vegetables": "Veggies",
    "veg": "Veggies",
    "hvyalcoholconsump": "HvyAlcoholConsump",
    "heavyalcohol": "HvyAlcoholConsump",
    "hvy_alcohol": "HvyAlcoholConsump",
    "hvy_alcohol_consump": "HvyAlcoholConsump",
    "anyhealthcare": "AnyHealthcare",
    "any_healthcare": "AnyHealthcare",
    "healthcare": "AnyHealthcare",
    "nodocbccost": "NoDocbcCost",
    "no_doc_bc_cost": "NoDocbcCost",
    "nodoccost": "NoDocbcCost",
    "genhlth": "GenHlth",
    "gen_hlth": "GenHlth",
    "generalhealth": "GenHlth",
    "general_health": "GenHlth",
    "menthlth": "MentHlth",
    "ment_hlth": "MentHlth",
    "mentalhealth": "MentHlth",
    "mental_health": "MentHlth",
    "physhlth": "PhysHlth",
    "phys_hlth": "PhysHlth",
    "physicalhealth": "PhysHlth",
    "physical_health": "PhysHlth",
    "diffwalk": "DiffWalk",
    "diff_walk": "DiffWalk",
    "difficultywalking": "DiffWalk",
    "sex": "Sex",
    "gender": "Sex",
    "age": "Age",
    "agegroup": "Age",
    "age_group": "Age",
    "education": "Education",
    "educationlevel": "Education",
    "education_level": "Education",
    "income": "Income",
    "incomelevel": "Income",
    "income_level": "Income",
}


def get_age_group(age: float) -> Optional[int]:
    """Map continuous age (years) to CDC BRFSS age group (1 to 13)."""
    if 18 <= age <= 24:
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    elif age >= 80:
        return 13
    return None


def _clean_str(val: Any) -> str:
    """Normalize string representation for comparison."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip().lower()


def _coerce_binary(val: Any) -> Optional[int]:
    """Coerce boolean/binary values to 0 or 1."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return 1 if val else 0
    if isinstance(val, (int, np.integer)):
        return 1 if val != 0 else 0
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return None
        return 1 if round(val) != 0 else 0

    s = _clean_str(val)
    if s in {"1", "1.0", "true", "t", "yes", "y", "positive"}:
        return 1
    if s in {"0", "0.0", "false", "f", "no", "n", "negative"}:
        return 0
    return None


def _coerce_sex(val: Any) -> Optional[int]:
    """Coerce sex to 0 (Female) or 1 (Male)."""
    b = _coerce_binary(val)
    if b is not None:
        return b
    s = _clean_str(val)
    if s in {"m", "male", "man", "boy"}:
        return 1
    if s in {"f", "female", "woman", "girl"}:
        return 0
    return None


def _coerce_gen_hlth(val: Any) -> Optional[int]:
    """Coerce general health to 1-5 scale."""
    if val is None:
        return None
    try:
        f = float(val)
        if not math.isnan(f):
            i = int(round(f))
            if 1 <= i <= 5:
                return i
    except (ValueError, TypeError):
        pass

    s = _clean_str(val)
    text_map = {
        "excellent": 1,
        "very good": 2,
        "verygood": 2,
        "good": 3,
        "fair": 4,
        "poor": 5,
    }
    return text_map.get(s)


def _coerce_age(val: Any) -> Optional[int]:
    """Coerce age group (1-13) or continuous age (18-120)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        # If passed as age group 1-13
        if 1 <= f <= 13 and f == int(f):
            return int(f)
        # If passed as continuous age in years
        if 18 <= f <= 120:
            return get_age_group(f)
        if 10 <= f < 18:
            return 1
    except (ValueError, TypeError):
        pass
    return None


def _coerce_float(val: Any, min_val: float, max_val: float) -> Optional[float]:
    """Coerce numeric float within specified bounds."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        if min_val <= f <= max_val:
            return round(f, 2)
    except (ValueError, TypeError):
        pass
    return None


def _coerce_int_range(val: Any, min_val: int, max_val: int) -> Optional[int]:
    """Coerce integer within range [min_val, max_val]."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        i = int(round(f))
        if min_val <= i <= max_val:
            return i
    except (ValueError, TypeError):
        pass
    return None


# ── Sample Dataset for Templates ─────────────────────────────────────────────
SAMPLE_DATA = [
    {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 32.4,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 4,
        "MentHlth": 10,
        "PhysHlth": 15,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 9,
        "Education": 4,
        "Income": 3,
    },
    {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 22.8,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 1,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 3,
        "Education": 6,
        "Income": 8,
    },
    {
        "HighBP": 1,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 28.1,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 3,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 5,
        "Income": 6,
    },
    {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 36.7,
        "Smoker": 1,
        "Stroke": 1,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 0,
        "HvyAlcoholConsump": 1,
        "AnyHealthcare": 1,
        "NoDocbcCost": 1,
        "GenHlth": 5,
        "MentHlth": 25,
        "PhysHlth": 28,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 11,
        "Education": 2,
        "Income": 1,
    },
    {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 24.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 2,
        "MentHlth": 1,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 5,
        "Education": 5,
        "Income": 7,
    },
]


# ── Template Generator ───────────────────────────────────────────────────────
def generate_template(file_format: str = "csv", sample: bool = False) -> Tuple[bytes, str, str]:
    """
    Generate blank or sample CSV/Excel template files.

    Returns:
        (file_bytes, mimetype, filename)
    """
    if sample:
        df = pd.DataFrame(SAMPLE_DATA)[FEATURE_COLUMNS]
        filename_prefix = "glucoscreen_batch_sample"
    else:
        # Create empty DataFrame with required column headers
        df = pd.DataFrame(columns=FEATURE_COLUMNS)
        filename_prefix = "glucoscreen_batch_template"

    if file_format.lower() in {"xlsx", "excel"}:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Batch Screening")
            # Auto-adjust column widths
            worksheet = writer.sheets["Batch Screening"]
            for col_idx, col in enumerate(df.columns, start=1):
                col_letter = worksheet.cell(row=1, column=col_idx).column_letter
                max_len = max(len(str(col)), 12)
                worksheet.column_dimensions[col_letter].width = max_len + 3

        output.seek(0)
        return (
            output.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"{filename_prefix}.xlsx",
        )
    else:
        # CSV format
        csv_str = df.to_csv(index=False)
        return (
            csv_str.encode("utf-8"),
            "text/csv",
            f"{filename_prefix}.csv",
        )


# ── File Parser & Validator ──────────────────────────────────────────────────
def parse_and_validate_file(
    file_bytes: bytes, filename: str, max_rows: int = 10000
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Parse uploaded CSV or Excel file, map column names, validate types,
    and return cleaned DataFrame or detailed error list.

    Returns:
        (cleaned_df, validation_errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not file_bytes:
        return None, ["The uploaded file is empty."], []

    # Detect format from filename
    lower_name = filename.lower()
    try:
        if lower_name.endswith(".csv"):
            # Try reading with utf-8, fallback to latin-1
            try:
                df_raw = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        elif lower_name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return (
                None,
                ["Unsupported file format. Please upload a .csv or .xlsx file."],
                [],
            )
    except Exception as exc:
        return None, [f"Failed to parse file: {str(exc)}"], []

    if df_raw.empty or len(df_raw) == 0:
        return None, ["The uploaded file contains no data rows."], []

    if len(df_raw) > max_rows:
        return (
            None,
            [f"File contains {len(df_raw)} rows, which exceeds the maximum limit of {max_rows} rows."],
            [],
        )

    # ── Normalize Column Names ──────────────────────────────────────────────
    col_mapping = {}
    for col in df_raw.columns:
        norm_key = str(col).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        mapped_target = COLUMN_ALIASES.get(norm_key)
        if mapped_target:
            col_mapping[col] = mapped_target

    df_renamed = df_raw.rename(columns=col_mapping)

    # Check for missing required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_renamed.columns]
    if missing_cols:
        missing_formatted = ", ".join(f"'{c}'" for c in missing_cols)
        errors.append(
            f"Missing {len(missing_cols)} required column(s): {missing_formatted}. "
            f"Please download the template to ensure your dataset contains all 21 health indicator columns."
        )
        return None, errors, warnings

    # ── Row-Level Validation and Type Coercion ──────────────────────────────
    cleaned_rows = []
    binary_cols = {
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
    }

    for row_idx, (_, row) in enumerate(df_renamed.iterrows(), start=1):
        row_errors = []
        clean_row = {}

        # 1. Binary columns (0 or 1)
        for col in binary_cols:
            val = _coerce_binary(row[col])
            if val is None:
                row_errors.append(f"{col} must be 0/1, True/False, or Yes/No (got '{row[col]}')")
            else:
                clean_row[col] = val

        # 2. Sex (0=Female, 1=Male)
        sex_val = _coerce_sex(row["Sex"])
        if sex_val is None:
            row_errors.append(f"Sex must be 0 (Female) or 1 (Male) (got '{row['Sex']}')")
        else:
            clean_row["Sex"] = sex_val

        # 3. BMI (continuous 10.0 to 100.0)
        bmi_val = _coerce_float(row["BMI"], 10.0, 100.0)
        if bmi_val is None:
            row_errors.append(f"BMI must be a number between 10.0 and 100.0 (got '{row['BMI']}')")
        else:
            clean_row["BMI"] = bmi_val

        # 4. GenHlth (1 to 5)
        gen_val = _coerce_gen_hlth(row["GenHlth"])
        if gen_val is None:
            row_errors.append(f"GenHlth must be between 1 (Excellent) and 5 (Poor) (got '{row['GenHlth']}')")
        else:
            clean_row["GenHlth"] = gen_val

        # 5. MentHlth & PhysHlth (0 to 30)
        ment_val = _coerce_float(row["MentHlth"], 0.0, 30.0)
        if ment_val is None:
            row_errors.append(f"MentHlth must be between 0 and 30 days (got '{row['MentHlth']}')")
        else:
            clean_row["MentHlth"] = ment_val

        phys_val = _coerce_float(row["PhysHlth"], 0.0, 30.0)
        if phys_val is None:
            row_errors.append(f"PhysHlth must be between 0 and 30 days (got '{row['PhysHlth']}')")
        else:
            clean_row["PhysHlth"] = phys_val

        # 6. Age (1 to 13 or age in years 18-120)
        age_val = _coerce_age(row["Age"])
        if age_val is None:
            row_errors.append(f"Age must be age group (1-13) or years (18-120) (got '{row['Age']}')")
        else:
            clean_row["Age"] = age_val

        # 7. Education (1 to 6)
        edu_val = _coerce_int_range(row["Education"], 1, 6)
        if edu_val is None:
            row_errors.append(f"Education must be between 1 and 6 (got '{row['Education']}')")
        else:
            clean_row["Education"] = edu_val

        # 8. Income (1 to 8)
        inc_val = _coerce_int_range(row["Income"], 1, 8)
        if inc_val is None:
            row_errors.append(f"Income must be between 1 and 8 (got '{row['Income']}')")
        else:
            clean_row["Income"] = inc_val

        if row_errors:
            if len(errors) < 15:  # Keep list manageable
                errors.append(f"Row {row_idx}: {'; '.join(row_errors)}")
            elif len(errors) == 15:
                errors.append("... Additional row validation errors truncated.")
        else:
            cleaned_rows.append(clean_row)

    if errors:
        return None, errors, warnings

    cleaned_df = pd.DataFrame(cleaned_rows)[FEATURE_COLUMNS]
    return cleaned_df, [], warnings


# ── Batch Prediction Engine ──────────────────────────────────────────────────
def run_batch_prediction(
    df: pd.DataFrame, model: Any, scaler: Any
) -> Dict[str, Any]:
    """
    Perform efficient vectorized scaling and prediction on the dataset.

    Returns structured summary metrics and enriched per-row prediction details.
    """
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Cannot run predictions on an empty dataset.")

    # 1. Prepare feature matrix matching model expectations
    X = df[FEATURE_COLUMNS].copy()

    # 2. Scale continuous features (BMI, MentHlth, PhysHlth)
    X_scaled_cols = scaler.transform(X[SCALED_FEATURES])
    X_matrix = X.values.astype(float)
    # Indices in FEATURE_COLUMNS: BMI=3, MentHlth=14, PhysHlth=15
    X_matrix[:, 3] = X_scaled_cols[:, 0]
    X_matrix[:, 14] = X_scaled_cols[:, 1]
    X_matrix[:, 15] = X_scaled_cols[:, 2]

    # Create DataFrame with exact column names for model
    X_transformed_df = pd.DataFrame(X_matrix, columns=FEATURE_COLUMNS)

    # 3. Model Inference
    probabilities = model.predict_proba(X_transformed_df)[:, 1]
    predictions = model.predict(X_transformed_df).astype(int)

    # 4. Classify risk levels and build row objects
    rows_data = []
    low_count = 0
    mod_count = 0
    high_count = 0

    for idx in range(n_rows):
        prob = float(probabilities[idx])
        pred_class = int(predictions[idx])
        score_pct = round(prob * 100, 1)

        if prob < 0.30:
            risk_level = "Low Risk"
            risk_tier = "low"
            low_count += 1
        elif prob <= 0.60:
            risk_level = "Moderate Risk"
            risk_tier = "mod"
            mod_count += 1
        else:
            risk_level = "High Risk"
            risk_tier = "high"
            high_count += 1

        original_features = {col: df.iloc[idx][col] for col in FEATURE_COLUMNS}

        rows_data.append(
            {
                "row_id": idx + 1,
                "risk_score": prob,
                "risk_score_pct": score_pct,
                "risk_level": risk_level,
                "risk_tier": risk_tier,
                "predicted_class": pred_class,
                "features": original_features,
            }
        )

    # 5. Aggregate Summary Statistics
    avg_score = float(np.mean(probabilities)) * 100
    min_score = float(np.min(probabilities)) * 100
    max_score = float(np.max(probabilities)) * 100

    summary = {
        "total_records": n_rows,
        "low_risk_count": low_count,
        "low_risk_pct": round((low_count / n_rows) * 100, 1) if n_rows > 0 else 0.0,
        "moderate_risk_count": mod_count,
        "moderate_risk_pct": round((mod_count / n_rows) * 100, 1) if n_rows > 0 else 0.0,
        "high_risk_count": high_count,
        "high_risk_pct": round((high_count / n_rows) * 100, 1) if n_rows > 0 else 0.0,
        "avg_risk_score_pct": round(avg_score, 1),
        "min_risk_score_pct": round(min_score, 1),
        "max_risk_score_pct": round(max_score, 1),
    }

    return {
        "summary": summary,
        "rows": rows_data,
    }


# ── Results Export Generator ─────────────────────────────────────────────────
def export_results(
    results: Dict[str, Any], file_format: str = "csv"
) -> Tuple[bytes, str, str]:
    """
    Generate downloadable CSV or Excel file containing inputs + predictions.

    Returns:
        (file_bytes, mimetype, filename)
    """
    rows = results.get("rows", [])
    export_records = []

    for r in rows:
        record = {"Record ID": r["row_id"]}
        record.update(r["features"])
        record["Risk Score (%)"] = r["risk_score_pct"]
        record["Risk Category"] = r["risk_level"]
        record["Predicted Class"] = (
            "Diabetes / High Risk (1)"
            if r["predicted_class"] == 1
            else "No Diabetes (0)"
        )
        export_records.append(record)

    df_out = pd.DataFrame(export_records)

    if file_format.lower() in {"xlsx", "excel"}:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Batch Results")
            worksheet = writer.sheets["Batch Results"]
            for col_idx, col in enumerate(df_out.columns, start=1):
                col_letter = worksheet.cell(row=1, column=col_idx).column_letter
                max_len = max(len(str(col)), 10)
                worksheet.column_dimensions[col_letter].width = max_len + 3

        output.seek(0)
        return (
            output.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "glucoscreen_batch_results.xlsx",
        )
    else:
        csv_str = df_out.to_csv(index=False)
        return (
            csv_str.encode("utf-8"),
            "text/csv",
            "glucoscreen_batch_results.csv",
        )
