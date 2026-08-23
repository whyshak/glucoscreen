"""
evaluation_service.py — Developer evaluation and benchmarking service for labeled datasets.

Enables local developers to upload a labeled test dataset containing feature columns (X)
and ground-truth labels (y) to benchmark model accuracy, precision, recall, F1, confusion
matrix, and row-by-row actual vs. predicted comparisons.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.batch_service import (
    COLUMN_ALIASES,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    SCALED_FEATURES,
    _clean_str,
    _coerce_age,
    _coerce_binary,
    _coerce_float,
    _coerce_gen_hlth,
    _coerce_int_range,
    _coerce_sex,
)

# Standard aliases for ground-truth target column
TARGET_ALIASES: Dict[str, str] = {
    "diabetes_binary": "Diabetes_binary",
    "diabetesbinary": "Diabetes_binary",
    "diabetes": "Diabetes_binary",
    "target": "Diabetes_binary",
    "label": "Diabetes_binary",
    "outcome": "Diabetes_binary",
    "y": "Diabetes_binary",
    "actual": "Diabetes_binary",
    "actual_class": "Diabetes_binary",
    "actualclass": "Diabetes_binary",
    "class": "Diabetes_binary",
    "ground_truth": "Diabetes_binary",
    "groundtruth": "Diabetes_binary",
    "true_label": "Diabetes_binary",
    "truelabel": "Diabetes_binary",
    "status": "Diabetes_binary",
    "diagnosis": "Diabetes_binary",
}

# Labeled sample dataset for local debugging and validation
SAMPLE_LABELED_DATA = [
    {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 34.2,
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
        "Age": 10,
        "Education": 4,
        "Income": 3,
        "Diabetes_binary": 1,
    },
    {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 22.4,
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
        "Diabetes_binary": 0,
    },
    {
        "HighBP": 1,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 27.8,
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
        "MentHlth": 2,
        "PhysHlth": 2,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 5,
        "Income": 6,
        "Diabetes_binary": 0,
    },
    {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 38.5,
        "Smoker": 1,
        "Stroke": 1,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 0,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 1,
        "GenHlth": 5,
        "MentHlth": 20,
        "PhysHlth": 25,
        "DiffWalk": 1,
        "Sex": 0,
        "Age": 11,
        "Education": 3,
        "Income": 2,
        "Diabetes_binary": 1,
    },
    {
        "HighBP": 0,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 29.1,
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
        "MentHlth": 5,
        "PhysHlth": 4,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 8,
        "Education": 5,
        "Income": 6,
        "Diabetes_binary": 0,
    },
    {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 31.0,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 4,
        "MentHlth": 12,
        "PhysHlth": 10,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 9,
        "Education": 4,
        "Income": 4,
        "Diabetes_binary": 1,
    },
]


def generate_labeled_template(file_format: str = "csv", sample: bool = True) -> Tuple[bytes, str, str]:
    """
    Generate blank or sample CSV/Excel template files including the ground-truth target column.

    Returns:
        (file_bytes, mimetype, filename)
    """
    columns_with_target = list(FEATURE_COLUMNS) + ["Diabetes_binary"]

    if sample:
        df = pd.DataFrame(SAMPLE_LABELED_DATA)[columns_with_target]
        filename_prefix = "glucoscreen_test_evaluation_sample"
    else:
        df = pd.DataFrame(columns=columns_with_target)
        filename_prefix = "glucoscreen_test_evaluation_template"

    if file_format.lower() in {"xlsx", "excel"}:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Evaluation Test Set")
            worksheet = writer.sheets["Evaluation Test Set"]
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
        csv_str = df.to_csv(index=False)
        return (
            csv_str.encode("utf-8"),
            "text/csv",
            f"{filename_prefix}.csv",
        )


def parse_and_validate_labeled_file(
    file_bytes: bytes, filename: str, max_rows: int = 20000
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], str, List[str], List[str]]:
    """
    Parse uploaded CSV or Excel file containing feature columns (X) and ground truth target column (y).
    Validates data types, maps column aliases, and returns cleaned X DataFrame, y Series, detected target column name,
    errors, and warnings.

    Returns:
        (cleaned_df, y_series, detected_target_col, validation_errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not file_bytes:
        return None, None, "", ["The uploaded file is empty."], []

    lower_name = filename.lower()
    try:
        if lower_name.endswith(".csv"):
            try:
                df_raw = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        elif lower_name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return (
                None,
                None,
                "",
                ["Unsupported file format. Please upload a .csv or .xlsx file."],
                [],
            )
    except Exception as exc:
        return None, None, "", [f"Failed to parse file: {str(exc)}"], []

    if df_raw.empty or len(df_raw) == 0:
        return None, None, "", ["The uploaded file contains no data rows."], []

    if len(df_raw) > max_rows:
        return (
            None,
            None,
            "",
            [f"File contains {len(df_raw)} rows, which exceeds the maximum limit of {max_rows} rows."],
            [],
        )

    # Normalize column names & map aliases for both features and target
    col_mapping = {}
    detected_target_col = ""

    for col in df_raw.columns:
        norm_key = str(col).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        # Check if target column
        if norm_key in TARGET_ALIASES:
            col_mapping[col] = "Diabetes_binary"
            detected_target_col = str(col)
        elif norm_key in COLUMN_ALIASES:
            col_mapping[col] = COLUMN_ALIASES[norm_key]

    df_renamed = df_raw.rename(columns=col_mapping)

    # Check for missing target column
    if "Diabetes_binary" not in df_renamed.columns:
        errors.append(
            "Missing ground-truth target column (y). Please ensure your file includes a column named "
            "'Diabetes_binary', 'Outcome', 'Target', 'Label', 'Actual', or 'y' containing actual 0/1 classes."
        )
        return None, None, "", errors, warnings

    # Check for missing feature columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df_renamed.columns]
    if missing_cols:
        missing_formatted = ", ".join(f"'{c}'" for c in missing_cols)
        errors.append(
            f"Missing {len(missing_cols)} required feature column(s): {missing_formatted}. "
            f"Please ensure all 21 CDC health indicator features are included."
        )
        return None, None, "", errors, warnings

    # Row-Level Validation and Type Coercion
    cleaned_rows = []
    y_values = []
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

        # 1. Target Column (y) Validation
        y_val = _coerce_binary(row["Diabetes_binary"])
        if y_val is None:
            row_errors.append(
                f"Target value must be binary (0 or 1, True/False, No/Yes) (got '{row['Diabetes_binary']}')"
            )

        # 2. Binary Feature Columns (0 or 1)
        for col in binary_cols:
            val = _coerce_binary(row[col])
            if val is None:
                row_errors.append(f"{col} must be 0/1, True/False, or Yes/No (got '{row[col]}')")
            else:
                clean_row[col] = val

        # 3. Sex (0=Female, 1=Male)
        sex_val = _coerce_sex(row["Sex"])
        if sex_val is None:
            row_errors.append(f"Sex must be 0 (Female) or 1 (Male) (got '{row['Sex']}')")
        else:
            clean_row["Sex"] = sex_val

        # 4. BMI (continuous 10.0 to 100.0)
        bmi_val = _coerce_float(row["BMI"], 10.0, 100.0)
        if bmi_val is None:
            row_errors.append(f"BMI must be a number between 10.0 and 100.0 (got '{row['BMI']}')")
        else:
            clean_row["BMI"] = bmi_val

        # 5. GenHlth (1 to 5)
        gen_val = _coerce_gen_hlth(row["GenHlth"])
        if gen_val is None:
            row_errors.append(f"GenHlth must be between 1 (Excellent) and 5 (Poor) (got '{row['GenHlth']}')")
        else:
            clean_row["GenHlth"] = gen_val

        # 6. MentHlth & PhysHlth (0 to 30)
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

        # 7. Age (1 to 13 or continuous 18-120)
        age_val = _coerce_age(row["Age"])
        if age_val is None:
            row_errors.append(f"Age must be age group (1-13) or years (18-120) (got '{row['Age']}')")
        else:
            clean_row["Age"] = age_val

        # 8. Education (1 to 6)
        edu_val = _coerce_int_range(row["Education"], 1, 6)
        if edu_val is None:
            row_errors.append(f"Education must be between 1 and 6 (got '{row['Education']}')")
        else:
            clean_row["Education"] = edu_val

        # 9. Income (1 to 8)
        inc_val = _coerce_int_range(row["Income"], 1, 8)
        if inc_val is None:
            row_errors.append(f"Income must be between 1 and 8 (got '{row['Income']}')")
        else:
            clean_row["Income"] = inc_val

        if row_errors:
            if len(errors) < 15:
                errors.append(f"Row {row_idx}: {'; '.join(row_errors)}")
            elif len(errors) == 15:
                errors.append("... Additional row validation errors truncated.")
        else:
            cleaned_rows.append(clean_row)
            y_values.append(y_val)

    if errors:
        return None, None, detected_target_col, errors, warnings

    cleaned_df = pd.DataFrame(cleaned_rows)[FEATURE_COLUMNS]
    y_series = pd.Series(y_values, name="Diabetes_binary", dtype=int)
    return cleaned_df, y_series, detected_target_col, [], warnings


def run_model_evaluation(
    df: pd.DataFrame, y_actual: pd.Series, model: Any, scaler: Any
) -> Dict[str, Any]:
    """
    Run the ML evaluation pipeline comparing ground-truth y_actual with model predictions y_pred and probabilities.

    Returns:
        Structured dictionary containing summary metrics, confusion matrix, and row-by-row comparisons.
    """
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    # 1. Feature scaling
    X = df[FEATURE_COLUMNS].copy()
    X_scaled_cols = scaler.transform(X[SCALED_FEATURES])
    X_matrix = X.values.astype(float)
    # Indices in FEATURE_COLUMNS: BMI=3, MentHlth=14, PhysHlth=15
    X_matrix[:, 3] = X_scaled_cols[:, 0]
    X_matrix[:, 14] = X_scaled_cols[:, 1]
    X_matrix[:, 15] = X_scaled_cols[:, 2]

    X_transformed_df = pd.DataFrame(X_matrix, columns=FEATURE_COLUMNS)

    # 2. Model inference
    probabilities = model.predict_proba(X_transformed_df)[:, 1]
    predictions = model.predict(X_transformed_df).astype(int)
    actuals = y_actual.values.astype(int)

    # 3. Compute Confusion Matrix & Counts
    tp = int(np.sum((actuals == 1) & (predictions == 1)))
    tn = int(np.sum((actuals == 0) & (predictions == 0)))
    fp = int(np.sum((actuals == 0) & (predictions == 1)))
    fn = int(np.sum((actuals == 1) & (predictions == 0)))

    matches = tp + tn
    mismatches = fp + fn

    accuracy = round(matches / n_rows * 100, 2) if n_rows > 0 else 0.0
    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    specificity = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0

    if (precision + recall) > 0:
        f1_score = round(2 * (precision * recall) / (precision + recall), 2)
    else:
        f1_score = 0.0

    # 4. Construct Row Objects
    rows_data = []
    actual_pos_count = int(np.sum(actuals == 1))
    actual_neg_count = int(np.sum(actuals == 0))

    for idx in range(n_rows):
        prob = float(probabilities[idx])
        pred_class = int(predictions[idx])
        act_class = int(actuals[idx])
        is_match = bool(pred_class == act_class)
        prob_pct = round(prob * 100, 1)

        if act_class == 1 and pred_class == 1:
            outcome_type = "TP"
            outcome_desc = "True Positive"
        elif act_class == 0 and pred_class == 0:
            outcome_type = "TN"
            outcome_desc = "True Negative"
        elif act_class == 0 and pred_class == 1:
            outcome_type = "FP"
            outcome_desc = "False Positive"
        else:
            outcome_type = "FN"
            outcome_desc = "False Negative"

        if prob < 0.30:
            risk_level = "Low Risk"
            risk_tier = "low"
        elif prob <= 0.60:
            risk_level = "Moderate Risk"
            risk_tier = "mod"
        else:
            risk_level = "High Risk"
            risk_tier = "high"

        original_features = {col: df.iloc[idx][col] for col in FEATURE_COLUMNS}

        rows_data.append(
            {
                "row_id": idx + 1,
                "actual_class": act_class,
                "actual_label": "Diabetes (1)" if act_class == 1 else "No Diabetes (0)",
                "predicted_class": pred_class,
                "predicted_label": "Diabetes (1)" if pred_class == 1 else "No Diabetes (0)",
                "probability": prob,
                "probability_pct": prob_pct,
                "is_match": is_match,
                "outcome_type": outcome_type,
                "outcome_desc": outcome_desc,
                "risk_level": risk_level,
                "risk_tier": risk_tier,
                "features": original_features,
            }
        )

    # 5. Metrics Dictionary
    metrics = {
        "total_records": n_rows,
        "actual_positives": actual_pos_count,
        "actual_negatives": actual_neg_count,
        "predicted_positives": tp + fp,
        "predicted_negatives": tn + fn,
        "matches": matches,
        "mismatches": mismatches,
        "accuracy_pct": accuracy,
        "precision_pct": precision,
        "recall_pct": recall,
        "specificity_pct": specificity,
        "f1_score": f1_score,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp_pct": round(tp / n_rows * 100, 1) if n_rows > 0 else 0.0,
            "tn_pct": round(tn / n_rows * 100, 1) if n_rows > 0 else 0.0,
            "fp_pct": round(fp / n_rows * 100, 1) if n_rows > 0 else 0.0,
            "fn_pct": round(fn / n_rows * 100, 1) if n_rows > 0 else 0.0,
        },
        "avg_risk_score_pct": round(float(np.mean(probabilities)) * 100, 1),
    }

    return {
        "metrics": metrics,
        "rows": rows_data,
    }


def export_evaluation_results(
    results: Dict[str, Any], file_format: str = "csv"
) -> Tuple[bytes, str, str]:
    """
    Generate downloadable CSV or Excel file containing full evaluation results:
    features, actual class, predicted class, probability score, match status, and outcome type.

    Returns:
        (file_bytes, mimetype, filename)
    """
    rows = results.get("rows", [])
    export_records = []

    for r in rows:
        record = {
            "Record ID": r["row_id"],
            "Actual Class (y)": r["actual_class"],
            "Predicted Class (y_hat)": r["predicted_class"],
            "Predicted Probability (%)": r["probability_pct"],
            "Match Status": "MATCH" if r["is_match"] else "MISMATCH",
            "Classification Type": r["outcome_type"],
            "Outcome Description": r["outcome_desc"],
            "Risk Tier": r["risk_level"],
        }
        # Add features
        record.update(r["features"])
        export_records.append(record)

    df_out = pd.DataFrame(export_records)

    if file_format.lower() in {"xlsx", "excel"}:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Evaluation Results")
            worksheet = writer.sheets["Evaluation Results"]
            for col_idx, col in enumerate(df_out.columns, start=1):
                col_letter = worksheet.cell(row=1, column=col_idx).column_letter
                max_len = max(len(str(col)), 10)
                worksheet.column_dimensions[col_letter].width = max_len + 3

        output.seek(0)
        return (
            output.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "glucoscreen_evaluation_results.xlsx",
        )
    else:
        csv_str = df_out.to_csv(index=False)
        return (
            csv_str.encode("utf-8"),
            "text/csv",
            "glucoscreen_evaluation_results.csv",
        )
