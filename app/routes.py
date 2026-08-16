"""
routes.py — Main Blueprint for GlucoScreen.

Routes:
    GET  /            → Landing page
    GET  /screening   → Multi-step screening form
    POST /predict     → Run (or dummy-run) the diabetes risk model
    GET  /result      → Display prediction result
"""

import os
import random

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
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
#loading the model
_models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(_models_dir, 'best_svm_model.pkl')
model = joblib.load(model_path)

#loading the scaler
scaler_path = os.path.join(_models_dir, 'standard_scaler.pkl')
scaler = joblib.load(scaler_path)

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
# EXPLANATION HELPER — Dummy data structure for SHAP-based UI
# =============================================================================
# This function provides mock explanation data so the UI can be built and
# tested independently of the (expensive) SHAP computation.
#
# Explanation states returned via the `expl_state` key:
#   "available"   — explanation data is ready to display
#   "loading"     — SHAP job is still running (async / background)
#   "unavailable" — SHAP could not be computed or an error occurred
#
# TODO (SHAP Integration — Step A):
#   Replace this entire function (or bypass it) once background SHAP
#   computation is wired up.  The expected output schema is:
#
#   {
#     "expl_state": "available",          # str
#     "lowering": [                        # list[dict]  — negative SHAP values
#       {
#         "name":      str,               # human-readable feature name
#         "value":     str,               # formatted user-provided value
#         "bar_pct":   int,               # 0-100, relative bar width
#         "influence": str,               # "Strong" | "Moderate" | "Small"
#       }, ...
#     ],
#     "increasing": [                     # list[dict]  — positive SHAP values
#       { same schema as above }
#     ],
#   }
# =============================================================================

def get_dummy_risk_explanation():
    """
    Return a mock SHAP-explanation dict with three possible states.

    Change `_DEMO_STATE` below to test each UI component state:
        "available"   → shows the two factor sections with dummy data
        "loading"     → shows the preparation spinner/message
        "unavailable" → shows the unavailable notice
    """
    _DEMO_STATE = "available"  # ← change to "loading" or "unavailable" to test other states

    if _DEMO_STATE != "available":
        return {"expl_state": _DEMO_STATE}

    # ── Dummy factor rows ─────────────────────────────────────────────────
    # Each dict mirrors what real SHAP processing will produce.
    # `bar_pct` is calculated as  abs(shap_val) / max_abs_shap * 100
    # and must be in the range 0–100 (relative, not a probability).
    #
    # TODO (SHAP Integration — Step B):
    #   Replace the two lists below with the output of a function that:
    #     1. Loads the precomputed SHAP values from a pickled/cached file
    #        WITHOUT blocking the primary /predict response.
    #     2. Extracts the class-1 SHAP values for this specific prediction.
    #     3. Maps raw feature names to FEATURE_LABELS (see dict below).
    #     4. Formats each raw feature value into a human-readable string.
    #     5. Splits features into `lowering` (shap < 0) and
    #        `increasing` (shap > 0) lists, sorted by abs(shap) desc.
    #     6. Keeps only the top 3–5 features per side.
    #     7. Computes bar_pct = round(abs(shap_val) / max_abs * 100).
    #     8. Maps bar_pct → influence tier:
    #           >= 65  → "Strong influence"
    #           30–64  → "Moderate influence"
    #           < 30   → "Small influence"
    #
    # FEATURE_LABELS = {
    #   "HighBP":              "High Blood Pressure",
    #   "HighChol":            "High Cholesterol",
    #   "CholCheck":           "Cholesterol Check",
    #   "BMI":                 "Body Mass Index (BMI)",
    #   "Smoker":              "Smoking History",
    #   "Stroke":              "Stroke History",
    #   "HeartDiseaseorAttack":"Heart Disease / Attack",
    #   "PhysActivity":        "Physical Activity",
    #   "Fruits":              "Daily Fruit Consumption",
    #   "Veggies":             "Daily Vegetable Consumption",
    #   "HvyAlcoholConsump":   "Heavy Alcohol Consumption",
    #   "AnyHealthcare":       "Healthcare Coverage",
    #   "NoDocbcCost":         "Skipped Doctor Due to Cost",
    #   "GenHlth":             "General Health Rating",
    #   "MentHlth":            "Days of Poor Mental Health",
    #   "PhysHlth":            "Days of Poor Physical Health",
    #   "DiffWalk":            "Difficulty Walking / Stairs",
    #   "Sex":                 "Sex",
    #   "Age":                 "Age Group",
    #   "Education":           "Education Level",
    #   "Income":              "Household Income",
    # }

    return {
        "expl_state": "available",

        # Factors that pulled the risk score DOWN (negative SHAP)
        "lowering": [
            {"name": "Physical Activity",        "value": "Yes",       "bar_pct": 88, "influence": "Strong influence"},
            {"name": "Daily Vegetable Consumption", "value": "Yes",    "bar_pct": 54, "influence": "Moderate influence"},
            {"name": "Heavy Alcohol Consumption", "value": "No",       "bar_pct": 27, "influence": "Small influence"},
        ],

        # Factors that pushed the risk score UP (positive SHAP)
        "increasing": [
            {"name": "Body Mass Index (BMI)",     "value": "31.4",     "bar_pct": 95, "influence": "Strong influence"},
            {"name": "High Blood Pressure",       "value": "Yes",      "bar_pct": 72, "influence": "Strong influence"},
            {"name": "General Health Rating",     "value": "Fair (4)", "bar_pct": 41, "influence": "Moderate influence"},
            {"name": "Age Group",                 "value": "50–54",    "bar_pct": 22, "influence": "Small influence"},
        ],
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

    features_to_scale = [bmi, ment_hlth, phys_hlth]

    # =========================================================================
    # STEP 5 — PREPROCESS WITH YOUR SCALER
    # =========================================================================
    # A StandardScaler was fit on the training data.
    # Calling .transform() here applies the SAME mean/std shift to the
    # incoming user values before they reach the model.
    #

    # applying standar scaler to the features tobe scaled
    features_to_scale_array  = np.array(features_to_scale).reshape(1, -1)  # shape: (1, 21)
    features_scaled = scaler.transform(features_to_scale_array)   # apply scaler
    
    # replacing the original feature values with the scaled features
    features[3] = features_scaled[0][0]
    features[14] = features_scaled[0][1]
    features[15] = features_scaled[0][2]

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

    return jsonify(
        {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "message": message,
        }
    )


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

    try:
        risk_pct = round(float(risk_score) * 100, 1)
    except ValueError:
        risk_pct = 0.0

    # ── Explanation data ─────────────────────────────────────────────────
    # TODO (SHAP Integration — Step C):
    #   Replace get_dummy_risk_explanation() with a call that reads the
    #   precomputed SHAP result for THIS prediction from a cache/queue
    #   (e.g., a file named by session ID, a Redis key, or an in-memory
    #   store).  If the background job is not yet done, return
    #   {"expl_state": "loading"}.  If it errored, return
    #   {"expl_state": "unavailable"}.
    #
    #   Example skeleton:
    #       job_id = request.args.get("job_id", "")
    #       explanation = fetch_shap_result(job_id)   # your async lookup
    #
    explanation = get_dummy_risk_explanation()

    return render_template(
        "result.html",
        risk_level=risk_level,
        risk_pct=risk_pct,
        nickname=nickname,
        explanation=explanation,
    )
