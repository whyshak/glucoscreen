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

    return render_template(
        "result.html",
        risk_level=risk_level,
        risk_pct=risk_pct,
        nickname=nickname,
    )
