"""
routes.py — Main Blueprint for GlucoScreen.

Routes:
    GET  /            → Landing page
    GET  /screening   → Multi-step screening form
    POST /predict     → Run (or dummy-run) the diabetes risk model
    GET  /result      → Display prediction result
"""

import os
import threading
import uuid

import io

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from app.batch_service import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    export_results,
    generate_template,
    parse_and_validate_file,
    run_batch_prediction,
)
from app.evaluation_service import (
    export_evaluation_results,
    generate_labeled_template,
    parse_and_validate_labeled_file,
    run_model_evaluation,
)
from app.chatbot_service import (
    QUICK_PROMPTS,
    WELCOME_MESSAGE,
    generate_mock_chat_response,
)

main = Blueprint("main", __name__)

# =============================================================================
# STEP 1 — IMPORT LIBRARIES FOR MODEL LOADING & PREPROCESSING
# =============================================================================

import joblib
import numpy as np
#
# =============================================================================


# =============================================================================
# STEP 2 — LOAD MODEL & SCALER AT MODULE STARTUP (runs once on boot)
# =============================================================================
# Loading here (at module level, outside any route function) means the files
# are read from disk only ONCE when Flask starts, not on every request.
# This is the standard, performant pattern for serving ML models in Flask.

#
_models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
#loading the model
model_path = os.path.join(_models_dir, 'best_svm_model.pkl')
model = joblib.load(model_path)

#loading the scaler
scaler_path = os.path.join(_models_dir, 'standard_scaler.pkl')
scaler = joblib.load(scaler_path)

#loading the shap explainer
shap_explainer_path = os.path.join(_models_dir, 'shap_explainer.pkl')
shap_explainer = joblib.load(shap_explainer_path)

features_array = np.array([])

# In-memory cache for background SHAP jobs.
# Keys are UUIDs (job_id strings); values are dicts with:
#   {"state": "loading" | "available" | "unavailable",
#    "explanation": <dict> | None}
_shap_cache: dict = {}

# =============================================================================


# =============================================================================
# Helper functions
# =============================================================================
def get_age_group(age):
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
    else:
        return None



# =============================================================================
# EXPLANATION HELPER — Processes real SHAP values into UI-ready dicts
# =============================================================================
# Explanation states returned via the `expl_state` key:
#   "available"   — explanation data is ready to display
#   "loading"     — SHAP job is still running (background thread)
#   "unavailable" — SHAP could not be computed or an error occurred
# =============================================================================

# Maps raw CDC BRFSS column names → human-readable labels shown in the UI.
FEATURE_LABELS = {
    "HighBP":               "High Blood Pressure",
    "HighChol":             "High Cholesterol",
    "CholCheck":            "Cholesterol Check",
    "BMI":                  "Body Mass Index (BMI)",
    "Smoker":               "Smoking History",
    "Stroke":               "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "PhysActivity":         "Physical Activity",
    "Fruits":               "Daily Fruit Consumption",
    "Veggies":              "Daily Vegetable Consumption",
    "HvyAlcoholConsump":    "Heavy Alcohol Consumption",
    "AnyHealthcare":        "Healthcare Coverage",
    "NoDocbcCost":          "Skipped Doctor Due to Cost",
    "GenHlth":              "General Health Rating",
    "MentHlth":             "Days of Poor Mental Health",
    "PhysHlth":             "Days of Poor Physical Health",
    "DiffWalk":             "Difficulty Walking / Stairs",
    "Sex":                  "Sex",
    "Age":                  "Age Group",
    "Education":            "Education Level",
    "Income":               "Household Income",
}


FEATURE_NAMES = list(FEATURE_LABELS.keys())


def _influence_label(bar_pct: int) -> str:
    """Map a relative bar percentage to a human-readable influence tier."""
    if bar_pct >= 65:
        return "Strong influence"
    elif bar_pct >= 30:
        return "Moderate influence"
    return "Small influence"


AGE_GROUP_LABELS = {
    1: "18–24",
    2: "25–29",
    3: "30–34",
    4: "35–39",
    5: "40–44",
    6: "45–49",
    7: "50–54",
    8: "55–59",
    9: "60–64",
    10: "65–69",
    11: "70–74",
    12: "75–79",
    13: "80+",
}

GENHLTH_LABELS = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

EDUCATION_LABELS = {
    1: "Never attended school",
    2: "Elementary",
    3: "Some high school",
    4: "High school graduate",
    5: "Some college",
    6: "College graduate",
}

INCOME_LABELS = {
    1: "< $10,000",
    2: "$10,000–$15,000",
    3: "$15,000–$20,000",
    4: "$20,000–$25,000",
    5: "$25,000–$35,000",
    6: "$35,000–$50,000",
    7: "$50,000–$75,000",
    8: "> $75,000",
}


def _format_value(name: str, raw) -> str:
    """
    Convert a raw numeric feature value to a user-friendly human-readable string.
    """
    if raw is None:
        return "—"

    # Age Group
    if name == "Age":
        try:
            return AGE_GROUP_LABELS.get(int(float(raw)), str(raw))
        except (TypeError, ValueError):
            return str(raw)

    # General Health Rating
    if name == "GenHlth":
        try:
            return GENHLTH_LABELS.get(int(float(raw)), str(raw))
        except (TypeError, ValueError):
            return str(raw)

    # Education Level
    if name == "Education":
        try:
            return EDUCATION_LABELS.get(int(float(raw)), str(raw))
        except (TypeError, ValueError):
            return str(raw)

    # Household Income
    if name == "Income":
        try:
            return INCOME_LABELS.get(int(float(raw)), str(raw))
        except (TypeError, ValueError):
            return str(raw)

    # Sex
    if name == "Sex":
        try:
            return "Male" if int(float(raw)) == 1 else "Female"
        except (TypeError, ValueError):
            return "Male" if raw else "Female"

    # Binary indicators
    binary_features = {
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
    }
    if name in binary_features:
        try:
            return "Yes" if int(float(raw)) else "No"
        except (TypeError, ValueError):
            return "Yes" if raw else "No"

    # Continuous / ordinal fallback (BMI, MentHlth, PhysHlth)
    if name == "BMI":
        try:
            return str(round(float(raw), 1))
        except (TypeError, ValueError):
            return str(raw)

    if name in {"MentHlth", "PhysHlth"}:
        try:
            days = int(round(float(raw)))
            return f"{days} days" if days != 1 else "1 day"
        except (TypeError, ValueError):
            return str(raw)

    try:
        return str(round(float(raw), 1))
    except (TypeError, ValueError):
        return str(raw)


def get_risk_explanation(shap_values, class_index: int, raw_features=None, top_n: int = 5) -> dict:
    """
    Build a UI-ready explanation dict from a real SHAP Explanation object.

    Args:
        shap_values:   shap.Explanation or array returned by shap_explainer(features_array).
        class_index:   Index of the predicted class (0 = No Diabetes, 1 = Diabetes).
        raw_features:  Original unscaled user input values for friendly display.
        top_n:         Maximum number of features to include per side (default 5).

    Returns:
        {
          "expl_state": "available",
          "lowering":   [list of factor dicts — negative SHAP, sorted desc by |shap|],
          "increasing": [list of factor dicts — positive SHAP, sorted desc by |shap|],
        }
    """
    # ── Extract per-feature SHAP values for the predicted class ──────────
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        if vals.ndim == 3:
            sv = vals[0, :, class_index]
        elif vals.ndim == 2:
            sv = vals[0, :]
        else:
            sv = vals
    elif isinstance(shap_values, (list, tuple)):
        arr = shap_values[class_index]
        sv = arr[0] if getattr(arr, "ndim", 1) == 2 else arr
    else:
        sv = np.array(shap_values)
        if sv.ndim >= 2:
            sv = sv[0]

    # Feature names
    if hasattr(shap_values, "feature_names") and shap_values.feature_names is not None:
        feature_names = list(shap_values.feature_names)
    else:
        feature_names = FEATURE_NAMES

    # Raw user values
    if raw_features is not None:
        raw_data = raw_features
    elif hasattr(shap_values, "data") and shap_values.data is not None:
        raw_data = shap_values.data[0] if getattr(shap_values.data, "ndim", 1) == 2 else shap_values.data
    else:
        raw_data = [None] * len(sv)

    # ── Relative bar widths ───────────────────────────────────────────────
    abs_sv = [abs(v) for v in sv]
    max_abs = max(abs_sv) if any(abs_sv) else 1.0        # avoid div-by-zero

    # ── Build sorted factor lists ─────────────────────────────────────────
    lowering = []   # negative SHAP → reduces risk
    increasing = [] # positive SHAP → raises risk

    for feat_name, shap_val, raw_val in zip(feature_names, sv, raw_data):
        if shap_val == 0:
            continue
        bar_pct = round(abs(shap_val) / max_abs * 100)
        row = {
            "name":      FEATURE_LABELS.get(feat_name, feat_name),
            "value":     _format_value(feat_name, raw_val),
            "bar_pct":   bar_pct,
            "influence": _influence_label(bar_pct),
        }
        if shap_val < 0:
            lowering.append((abs(shap_val), row))
        else:
            increasing.append((abs(shap_val), row))

    # Sort each list descending by |shap| and keep top_n entries
    lowering   = [r for _, r in sorted(lowering,   reverse=True)][:top_n]
    increasing = [r for _, r in sorted(increasing, reverse=True)][:top_n]

    # ── Return explanation for the predicted class ─────────────────────
    # If class_index == 0 (No Diabetes) → lower_list shows factors that
    # *reduce* the probability of diabetes; upper_list shows factors that
    # *increase* it.
    # If class_index == 1 (Diabetes) → interpretation flips:
    # lower_list = factors that increased diabetes probability
    # upper_list = factors that decreased diabetes probability
    if class_index == 0:
        lowering, increasing = increasing, lowering

    return {
        "expl_state": "available",
        "lowering":   lowering,
        "increasing": increasing,
    }


# ── Landing Page ─────────────────────────────────────────────────────────────

@main.route("/")
def index():
    """Render the public landing / marketing page."""
    return render_template("index.html")


# ── Screening Form ───────────────────────────────────────────────────────────

@main.route("/screening")
def screening():
    """Render the multi-step diabetes risk screening form."""
    return render_template("screening.html")


# ── Predict (Risk Assessment) ────────────────────────────────────────────────

@main.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON form data from the screening form, run the risk model,
    and return a JSON response with the risk level and score.

    Expected JSON body fields (all CDC Health Indicators):
        age          (int)   : Age of the respondent
        sex          (0|1)   : 0 = Female, 1 = Male
        heightCm     (float) : Height in centimetres
        weightKg     (float) : Weight in kilograms
        highBp       (bool)  : High blood pressure ever diagnosed
        highChol     (bool)  : High cholesterol ever diagnosed
        cholCheck    (bool)  : Cholesterol check in last 5 years
        smoker       (bool)  : Smoked >= 100 cigarettes in lifetime
        stroke       (bool)  : Ever told had a stroke
        heartDisease (bool)  : Coronary heart disease / MI ever diagnosed
        physActivity (bool)  : Physical activity in past 30 days
        fruits       (bool)  : Consumes fruit >= 1 time/day
        veggies      (bool)  : Consumes vegetables >= 1 time/day
        hvyAlcohol   (bool)  : Heavy alcohol consumption
        anyHealthcare(bool)  : Has any healthcare coverage
        noDocbcCost  (bool)  : Couldn't see doctor due to cost in past 12 months
        genHlth      (1-5)   : General health self-rating (1=Excellent … 5=Poor)
        mentHlth     (0-30)  : Days of poor mental health in past 30 days
        physHlth     (0-30)  : Days of poor physical health in past 30 days
        diffWalk     (bool)  : Difficulty walking / climbing stairs
        education    (1-6)   : Education level category
        income       (1-8)   : Income level category
        nickname     (str)   : Optional user display name (session only, not stored)

    Returns JSON:
        {
          "risk_level": "High Risk" | "Low Risk",
          "risk_score": float,   // 0.0 – 1.0 probability
          "message":   str
        }
    """
    data = request.get_json(force=True)

    # ── Store nickname in session (never persisted to DB) ──────────────────
    nickname = (data.get("nickname") or "").strip()
    if nickname:
        session["nickname"] = nickname

    # =========================================================================
    # STEP 3 — EXTRACT & VALIDATE RAW FORM FIELDS
    # =========================================================================
    # The block below reads every field sent by the screening form and casts
    # it to the correct Python type.
    # =========================================================================

    age        = get_age_group(float(data.get("age", 0)))
    sex        = int(bool(data.get("sex", 0)))
    height_cm  = float(data.get("heightCm", 0))
    weight_kg  = float(data.get("weightKg", 10))

    # ── Derived feature: BMI ──────────────────────────────────────────────
    # BMI = weight(kg) / height(m)²
    # The CDC BRFSS dataset uses BMI as a continuous feature, not raw
    # height/weight separately.
    bmi = round(weight_kg / ((height_cm / 100) ** 2)) if height_cm > 0 else 0

    # Binary health indicators — stored as 0 / 1 integers to match
    # the numeric dtype the scaler and model expect.
    high_bp        = int(bool(data.get("highBp",        False)))
    high_chol      = int(bool(data.get("highChol",      False)))
    chol_check     = int(bool(data.get("cholCheck",     False)))
    smoker         = int(bool(data.get("smoker",        False)))
    stroke         = int(bool(data.get("stroke",        False)))
    heart_disease  = int(bool(data.get("heartDisease",  False)))
    phys_activity  = int(bool(data.get("physActivity",  False)))
    fruits         = int(bool(data.get("fruits",        False)))
    veggies        = int(bool(data.get("veggies",       False)))
    hvy_alcohol    = int(bool(data.get("hvyAlcohol",    False)))
    any_healthcare = int(bool(data.get("anyHealthcare", False)))
    no_doc_bc_cost = int(bool(data.get("noDocbcCost",   False)))
    diff_walk      = int(bool(data.get("diffWalk",      False)))

    # Ordinal / scale features
    gen_hlth  = int(data.get("genHlth",   3))
    ment_hlth = float(data.get("mentHlth", 0))
    phys_hlth = float(data.get("physHlth", 0))
    education = int(data.get("education", 4))
    income    = int(data.get("income",    5))

    # =========================================================================
    # STEP 4 — ASSEMBLE THE FEATURE VECTOR
    # =========================================================================
    # ⚠️  IMPORTANT: The ORDER of values in this list must exactly match the
    # column order that your model was trained on.
    #
    # order (CDC BRFSS standard column order):
    #   HighBP, HighChol, CholCheck, BMI, Smoker,
    #   Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
    #   HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
    #   MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    # =========================================================================
    features = [
        high_bp, high_chol, chol_check, bmi, smoker,
        stroke, heart_disease, phys_activity, fruits, veggies,
        hvy_alcohol, any_healthcare, no_doc_bc_cost, gen_hlth,
        ment_hlth, phys_hlth, diff_walk, sex, age, education, income,
    ]
    raw_features = list(features)

    features_to_scale = [bmi, ment_hlth, phys_hlth]

    # =========================================================================
    # STEP 5 — PREPROCESS WITH YOUR SCALER
    # =========================================================================
    # A StandardScaler was fit on the training data.
    # Calling .transform() here applies the SAME mean/std shift to the
    # incoming user values before they reach the model.
    #

    # applying standar scaler to the features tobe scaled
    features_to_scale_array  = np.array(features_to_scale).reshape(1, -1)  # shape: (1, 3)
    features_scaled = scaler.transform(features_to_scale_array)   # apply scaler
    
    # replacing the original feature values with the scaled features
    features[3] = features_scaled[0][0]
    features[14] = features_scaled[0][1]
    features[15] = features_scaled[0][2]

    global features_array
    # converting the features list to a numpy array
    features_array  = np.array(features).reshape(1, -1)  # shape: (1, 21)


    # =========================================================================
    # STEP 6 — RUN THE MODEL & READ THE PREDICTION
    # =========================================================================
    # Classifiers trained with scikit-learn expose .predict_proba(), which
    # returns the probability for each class:
    #   predict_proba()[0][0]  →  probability of class 0 (No Diabetes)
    #   predict_proba()[0][1]  →  probability of class 1 (Diabetes / Pre-diabetes)
    #
    #
    model_prediction = model.predict_proba(features_array)
    risk_score = float(model_prediction[0][1])
    
    # ── Risk level classification (3-tier) ────────────────────────────────
    # Thresholds:  < 0.30  → Low Risk
    #              0.30 – 0.60  → Moderate Risk
    #              > 0.60  → High Risk
    if risk_score < 0.30:
        risk_level = "Low Risk"
    elif risk_score <= 0.60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # ── Response message per risk tier ────────────────────────────────────
    if risk_level == "High Risk":
        message = (
            "Based on the information you provided, you may be at significantly "
            "elevated risk for diabetes. Please consult a healthcare professional "
            "as soon as possible for a formal evaluation and personalised plan."
        )
    elif risk_level == "Moderate Risk":
        message = (
            "Your responses indicate a moderate level of diabetes risk. "
            "This is a good time to make proactive lifestyle changes — "
            "diet, exercise and routine blood sugar checks can make a real difference."
        )
    else:
        message = (
            "Your responses suggest a lower likelihood of diabetes. "
            "Keep up the healthy habits, and continue routine annual check-ups."
        )

    # ── Determine predicted class index ───────────────────────────────────
    # class 0 = No Diabetes, class 1 = Diabetes / Pre-diabetes
    predicted_class = int(model.predict(features_array)[0])

    # ── Launch SHAP computation in a background thread ─────────────────────
    # SHAP on an SVM kernel can take 2-5 minutes; we run it off the
    # request thread so the user is never blocked.
    job_id = str(uuid.uuid4())
    _shap_cache[job_id] = {"state": "loading", "explanation": None}

    def _run_shap(jid: str, feat_arr, cls_idx: int, raw_feats):
        try:
            sv = shap_explainer(feat_arr)
            expl = get_risk_explanation(sv, cls_idx, raw_features=raw_feats)
            _shap_cache[jid] = {"state": "available", "explanation": expl}
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            _shap_cache[jid] = {"state": "unavailable", "explanation": None}

    thread = threading.Thread(
        target=_run_shap,
        args=(job_id, features_array.copy(), predicted_class, raw_features),
        daemon=True,
    )
    thread.start()

    return jsonify(
        {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "message": message,
            "job_id": job_id,
            "predicted_class": predicted_class,
        }
    )


# ── SHAP Status Endpoint ─────────────────────────────────────────────────────

@main.route("/shap_status")
def shap_status():
    """
    Lightweight polling endpoint for the result page.

    Query parameters:
        job_id (str): UUID returned by /predict.

    Returns JSON:
        {
          "state": "loading" | "available" | "unavailable",
          "explanation": <dict> | null   (null when not yet available)
        }
    """
    job_id = request.args.get("job_id", "")
    entry = _shap_cache.get(job_id)
    if entry is None:
        return jsonify({"state": "unavailable", "explanation": None})
    return jsonify({"state": entry["state"], "explanation": entry.get("explanation")})


# ── Result Page ──────────────────────────────────────────────────────────────

@main.route("/result")
def result():
    """
    Display the risk assessment result.

    Query parameters:
        risk_level (str)  : "High Risk" | "Low Risk"
        risk_score (float): 0.0 – 1.0 probability value
    """
    risk_level = request.args.get("risk_level", "Unknown")
    risk_score = request.args.get("risk_score", "0")
    nickname = session.get("nickname", "")
    # class_index: 0 = No Diabetes, 1 = Diabetes/Pre-diabetes
    class_index = int(request.args.get("predicted_class", 1))

    try:
        risk_pct = round(float(risk_score) * 100, 1)
    except ValueError:
        risk_pct = 0.0

    # ── Explanation data ──────────────────────────────────────────────────
    # The SHAP job was launched in the background by /predict and its result
    # stored in _shap_cache keyed by job_id.
    #   "loading"     → thread still running; UI shows a spinner
    #   "available"   → explanation is ready to render
    #   "unavailable" → SHAP failed; UI shows a graceful notice
    job_id = request.args.get("job_id", "")
    cache_entry = _shap_cache.get(job_id)

    if cache_entry is None:
        # Unknown job — treat as unavailable
        explanation = {"expl_state": "unavailable"}
    elif cache_entry["state"] == "loading":
        explanation = {"expl_state": "loading"}
    elif cache_entry["state"] == "available":
        explanation = cache_entry["explanation"]
    else:
        explanation = {"expl_state": "unavailable"}

    return render_template(
        "result.html",
        risk_level=risk_level,
        risk_pct=risk_pct,
        nickname=nickname,
        explanation=explanation,
        job_id=job_id,
    )


# =============================================================================
# BATCH PREDICTION ROUTES
# =============================================================================

# In-memory storage for processed batch prediction results
_batch_cache: dict[str, dict] = {}


@main.route("/batch")
def batch():
    """Render the Batch CSV / Excel Screening page."""
    return render_template(
        "batch.html",
        columns=FEATURE_COLUMNS,
        feature_labels=FEATURE_LABELS,
    )


@main.route("/batch/template")
def batch_template():
    """
    Download a pre-formatted template (blank or sample) in CSV or Excel format.

    Query parameters:
        format (str): 'csv' (default) or 'xlsx' / 'excel'
        sample (bool/str): '1', 'true', or 'yes' to include sample data rows
    """
    file_format = request.args.get("format", "csv").strip().lower()
    sample_param = request.args.get("sample", "0").strip().lower()
    include_sample = sample_param in {"1", "true", "yes"}

    file_bytes, mimetype, filename = generate_template(
        file_format=file_format, sample=include_sample
    )

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@main.route("/batch/predict", methods=["POST"])
def batch_predict():
    """
    Process an uploaded CSV or Excel file, validate all rows, run batch
    predictions using the SVM model & scaler, and return results JSON.
    """
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "errors": ["No file was uploaded. Please choose a CSV or Excel file."],
        }), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "errors": ["No file selected. Please select a valid file to upload."],
        }), 400

    filename = file.filename or "upload.csv"
    try:
        file_bytes = file.read()
    except Exception as exc:
        return jsonify({
            "success": False,
            "errors": [f"Error reading uploaded file: {str(exc)}"],
        }), 400

    # 1. Parse and validate file content
    df_cleaned, errors, warnings = parse_and_validate_file(file_bytes, filename)
    if errors:
        return jsonify({
            "success": False,
            "errors": errors,
            "warnings": warnings,
        }), 400

    # 2. Run vectorized prediction pipeline
    try:
        results = run_batch_prediction(df_cleaned, model, scaler)
    except Exception as exc:
        return jsonify({
            "success": False,
            "errors": [f"Prediction error: {str(exc)}"],
        }), 500

    # 3. Cache results for export download
    batch_id = str(uuid.uuid4())
    # Keep cache from growing unbounded
    if len(_batch_cache) > 50:
        oldest_key = next(iter(_batch_cache))
        _batch_cache.pop(oldest_key, None)

    _batch_cache[batch_id] = results

    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "filename": filename,
        "summary": results["summary"],
        "rows": results["rows"],
        "columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS,
        "warnings": warnings,
    })


@main.route("/batch/export", methods=["GET", "POST"])
def batch_export():
    """
    Export processed batch results to CSV or Excel.

    Accepts batch_id and format ('csv' or 'xlsx') via query parameters or JSON.
    """
    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}
        batch_id = data.get("batch_id")
        file_format = data.get("format", "csv")
    else:
        batch_id = request.args.get("batch_id")
        file_format = request.args.get("format", "csv")

    if not batch_id or batch_id not in _batch_cache:
        return jsonify({
            "error": "Batch session not found or has expired. Please run the batch screening again.",
        }), 404

    results = _batch_cache[batch_id]
    file_bytes, mimetype, filename = export_results(results, file_format=file_format)

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


# =============================================================================
# DEVELOPER-ONLY EVALUATION & BENCHMARKING ROUTES (HIDDEN / LOCALHOST ONLY)
# =============================================================================

# In-memory storage for processed developer evaluation sessions
_eval_cache: dict[str, dict] = {}


def _is_local_request() -> bool:
    """Check if the incoming request is originating from localhost / loopback."""
    remote_addr = (request.remote_addr or "").strip()
    return remote_addr in {"127.0.0.1", "::1", "localhost"}


def _is_debug_mode() -> bool:
    """Check if the Flask app is running in debug mode or testing mode."""
    return bool(
        current_app.debug
        or current_app.config.get("DEBUG", False)
        or current_app.config.get("TESTING", False)
    )


def _check_debug_access() -> None:
    """
    Strict developer-only access guard.
    Ensures route is only accessible when running in debug/testing mode from localhost.
    Aborts with 404 Not Found otherwise so route is completely invisible and non-functional in production.
    """
    if not (_is_debug_mode() and _is_local_request()):
        abort(404)


@main.route("/debug/evaluate", methods=["GET", "POST"])
def debug_evaluate():
    """
    Developer-only route for evaluating a labeled test dataset against the SVM model.

    - GET: Renders the evaluation dashboard.
    - POST: Processes an uploaded test dataset (CSV/XLSX with X and y), runs predictions,
      computes classification metrics (Accuracy, Precision, Recall, Specificity, F1, Confusion Matrix),
      and returns row-by-row comparisons.
    """
    _check_debug_access()

    if request.method == "GET":
        return render_template(
            "debug_evaluate.html",
            columns=FEATURE_COLUMNS,
            feature_labels=FEATURE_LABELS,
        )

    # POST Handling: Process labeled test file
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "errors": ["No file was uploaded. Please upload a labeled CSV or Excel test dataset."],
        }), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "errors": ["No file selected. Please select a valid test dataset to evaluate."],
        }), 400

    filename = file.filename or "test_dataset.csv"
    try:
        file_bytes = file.read()
    except Exception as exc:
        return jsonify({
            "success": False,
            "errors": [f"Error reading uploaded file: {str(exc)}"],
        }), 400

    # 1. Parse and validate features (X) and ground truth target (y)
    df_cleaned, y_actual, target_col, errors, warnings = parse_and_validate_labeled_file(file_bytes, filename)
    if errors:
        return jsonify({
            "success": False,
            "errors": errors,
            "warnings": warnings,
        }), 400

    # 2. Run evaluation pipeline
    try:
        eval_results = run_model_evaluation(df_cleaned, y_actual, model, scaler)
    except Exception as exc:
        return jsonify({
            "success": False,
            "errors": [f"Evaluation error: {str(exc)}"],
        }), 500

    # 3. Cache results for export
    eval_id = str(uuid.uuid4())
    if len(_eval_cache) > 50:
        oldest_key = next(iter(_eval_cache))
        _eval_cache.pop(oldest_key, None)

    _eval_cache[eval_id] = eval_results

    return jsonify({
        "success": True,
        "eval_id": eval_id,
        "filename": filename,
        "target_column": target_col,
        "metrics": eval_results["metrics"],
        "rows": eval_results["rows"],
        "columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS,
        "warnings": warnings,
    })


@main.route("/debug/evaluate/template", methods=["GET"])
def debug_evaluate_template():
    """
    Download a sample or blank labeled test template (CSV/Excel) containing
    both the 21 health indicator feature columns and the ground-truth target column.
    """
    _check_debug_access()

    file_format = request.args.get("format", "csv").strip().lower()
    sample_param = request.args.get("sample", "1").strip().lower()
    include_sample = sample_param in {"1", "true", "yes"}

    file_bytes, mimetype, filename = generate_labeled_template(
        file_format=file_format, sample=include_sample
    )

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@main.route("/debug/evaluate/export", methods=["GET", "POST"])
def debug_evaluate_export():
    """
    Export the complete comparative evaluation results (Actual vs. Predicted,
    Probability scores, Match/Mismatch flags, Error categorizations, and features) to CSV or Excel.
    """
    _check_debug_access()

    if request.method == "POST" and request.is_json:
        data = request.get_json(silent=True) or {}
        eval_id = data.get("eval_id")
        file_format = data.get("format", "csv")
    else:
        eval_id = request.args.get("eval_id")
        file_format = request.args.get("format", "csv")

    if not eval_id or eval_id not in _eval_cache:
        return jsonify({
            "error": "Evaluation session not found or has expired. Please run the evaluation again.",
        }), 404

    results = _eval_cache[eval_id]
    file_bytes, mimetype, filename = export_evaluation_results(results, file_format=file_format)

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


# =============================================================================
# DIA CHATBOT & RAG PIPELINE PLACEHOLDER ROUTES
# =============================================================================

@main.route("/chat", methods=["GET"])
def chat():
    """
    Render the dedicated full-page Dia Diabetes Assistant chat interface.
    """
    return render_template(
        "chat.html",
        quick_prompts=QUICK_PROMPTS,
        welcome_message=WELCOME_MESSAGE,
    )


@main.route("/api/chat", methods=["POST"])
def api_chat():
    """
    API endpoint to handle incoming conversational user messages from the Dia chat interface.

    Request JSON payload:
        {
            "message": str,                # User prompt / inquiry
            "history": list[dict] | None   # Optional conversation history
        }

    Response JSON:
        {
            "text": str,                   # Synthesized response or educational guidance
            "suggestions": list[str],      # Recommended follow-up question chips
            "rag_ready": bool              # Flag indicating if response is from full RAG or mock
        }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Please provide a valid message."}), 400

    if len(user_message) > 1000:
        return jsonify({
            "error": "Message is too long. Please keep your message under 1,000 characters."
        }), 400

    # =========================================================================
    # RAG PIPELINE INTEGRATION ROADMAP (FUTURE IMPLEMENTATION)
    # =========================================================================
    #
    # TODO (RAG Step 1): VECTOR DATABASE & EMBEDDING INDEX INITIALIZATION
    # -------------------------------------------------------------------------
    # 1. Connect to or load the persistent vector database (e.g., ChromaDB, FAISS,
    #    Pinecone, or Qdrant) populated from the scraped medical corpus:
    #    Repository: https://github.com/whyshak/diabetesscraped
    # 2. Source Documents include:
    #    - CDC Diabetes Factsheets and prevention guides
    #    - NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) clinical articles
    #    - Evidence-based dietary guidelines, glycemic index charts, and exercise regimens
    # 3. Use an embedding model (e.g., OpenAI text-embedding-3-small, HuggingFace
    #    sentence-transformers/all-MiniLM-L6-v2, or Vertex AI embeddings) to generate
    #    dense vector representations for chunked documents.
    #
    # Example Target Code:
    #   vector_store = Chroma(
    #       persist_directory="./rag_store/chroma_db",
    #       embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    #   )
    #
    # -------------------------------------------------------------------------
    # TODO (RAG Step 2): CONTEXT RETRIEVAL (SEMANTIC SEARCH)
    # -------------------------------------------------------------------------
    # 1. Generate query embedding for `user_message`.
    # 2. Perform similarity search (e.g., cosine similarity / MMR - Maximal Marginal Relevance)
    #    to fetch top-k (k=3 to 5) most relevant context chunks from the vector database.
    # 3. Apply relevance score thresholds and filter out low-confidence chunks.
    #
    # Example Target Code:
    #   retrieved_docs = vector_store.similarity_search_with_relevance_scores(
    #       query=user_message,
    #       k=4,
    #       score_threshold=0.65
    #   )
    #   context_text = "\n\n".join([doc.page_content for doc, score in retrieved_docs])
    #
    # -------------------------------------------------------------------------
    # TODO (RAG Step 3): PROMPT TEMPLATING & LLM INVOCATION
    # -------------------------------------------------------------------------
    # 1. Construct a clinical-safety-aligned system prompt with strict guardrails:
    #    - Informational & educational tone; never diagnose or prescribe.
    #    - Ground answers strictly in the retrieved context to avoid hallucinations.
    #    - Include medical disclaimer reminders when addressing medication or symptoms.
    # 2. Format prompt combining:
    #    - System instructions
    #    - Retrieved knowledge context
    #    - Recent conversation history (for multi-turn dialogue context)
    #    - User message
    # 3. Call LLM (e.g., Gemini 1.5 Pro/Flash, GPT-4o-mini, Anthropic Claude, or local Ollama).
    #
    # Example Target Code:
    #   system_prompt = (
    #       "You are Dia, an empathetic and knowledgeable diabetes health assistant for GlucoScreen. "
    #       "Answer the user's question using ONLY the provided medical context below. "
    #       "If the answer cannot be determined from the context, state that clearly and advise "
    #       "consulting a doctor. Include relevant suggestions for follow-up questions."
    #   )
    #   response = llm.invoke(messages=[
    #       SystemMessage(content=system_prompt),
    #       *formatted_history,
    #       HumanMessage(content=f"Context:\n{context_text}\n\nQuestion:\n{user_message}")
    #   ])
    #
    # -------------------------------------------------------------------------
    # TODO (RAG Step 4): RESPONSE SYNTHESIS & DYNAMIC SUGGESTIONS
    # -------------------------------------------------------------------------
    # 1. Parse LLM response text and extract synthesized medical advice.
    # 2. Generate 2-4 contextual follow-up suggestions tailored to the topic discussed.
    # 3. Return structured JSON payload to the frontend.
    # =========================================================================

    # ── Current Phase: Safe Educational Mock / Prototype Fallback Response ──
    mock_reply = generate_mock_chat_response(user_message)

    return jsonify({
        "text": mock_reply["text"],
        "suggestions": mock_reply["suggestions"],
        "rag_ready": False,
    })


