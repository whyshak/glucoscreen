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

# ---------------------------------------------------------------------------
# TODO (Model Integration): Load model and scaler once at startup.
#
#   import pickle
#   MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
#   SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
#
#   with open(MODEL_PATH, 'rb') as f:
#       model = pickle.load(f)
#   with open(SCALER_PATH, 'rb') as f:
#       scaler = pickle.load(f)
#
# ---------------------------------------------------------------------------


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
        age         (int)   : Age of the respondent
        sex         (0|1)   : 0 = Female, 1 = Male
        heightCm    (float) : Height in centimetres
        weightKg    (float) : Weight in kilograms
        highBp      (bool)  : High blood pressure ever diagnosed
        highChol    (bool)  : High cholesterol ever diagnosed
        cholCheck   (bool)  : Cholesterol check in last 5 years
        smoker      (bool)  : Smoked >= 100 cigarettes in lifetime
        stroke      (bool)  : Ever told had a stroke
        heartDisease(bool)  : Coronary heart disease / MI ever diagnosed
        physActivity(bool)  : Physical activity in past 30 days
        fruits      (bool)  : Consumes fruit >= 1 time/day
        veggies     (bool)  : Consumes vegetables >= 1 time/day
        hvyAlcohol  (bool)  : Heavy alcohol consumption
        anyHealthcare(bool) : Has any healthcare coverage
        noDocbcCost (bool)  : Couldn't see doctor due to cost in past 12 months
        genHlth     (1-5)   : General health self-rating (1=Excellent … 5=Poor)
        mentHlth    (0-30)  : Days of poor mental health in past 30 days
        physHlth    (0-30)  : Days of poor physical health in past 30 days
        diffWalk    (bool)  : Difficulty walking / climbing stairs
        education   (1-6)   : Education level category
        income      (1-8)   : Income level category
        nickname    (str)   : Optional user display name (session only, not stored)

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

    # ── Extract & validate feature fields ─────────────────────────────────
    age = float(data.get("age", 0))
    sex = int(bool(data.get("sex", 0)))
    height_cm = float(data.get("heightCm", 0))
    weight_kg = float(data.get("weightKg", 0))

    # Derived feature
    bmi = (weight_kg / ((height_cm / 100) ** 2)) if height_cm > 0 else 0.0

    # Binary health indicators (0 or 1)
    high_bp       = int(bool(data.get("highBp", False)))
    high_chol     = int(bool(data.get("highChol", False)))
    chol_check    = int(bool(data.get("cholCheck", False)))
    smoker        = int(bool(data.get("smoker", False)))
    stroke        = int(bool(data.get("stroke", False)))
    heart_disease = int(bool(data.get("heartDisease", False)))
    phys_activity = int(bool(data.get("physActivity", False)))
    fruits        = int(bool(data.get("fruits", False)))
    veggies       = int(bool(data.get("veggies", False)))
    hvy_alcohol   = int(bool(data.get("hvyAlcohol", False)))
    any_healthcare = int(bool(data.get("anyHealthcare", False)))
    no_doc_bc_cost = int(bool(data.get("noDocbcCost", False)))
    diff_walk     = int(bool(data.get("diffWalk", False)))

    # Scale / categorical
    gen_hlth  = int(data.get("genHlth", 3))
    ment_hlth = float(data.get("mentHlth", 0))
    phys_hlth = float(data.get("physHlth", 0))
    education = int(data.get("education", 4))
    income    = int(data.get("income", 5))

    # ── Feature vector in CDC model order ─────────────────────────────────
    # Order must match the feature order your model was trained on.
    # Adjust column order here once you confirm with your training notebook.
    features = [
        high_bp, high_chol, chol_check, bmi, smoker,
        stroke, heart_disease, phys_activity, fruits, veggies,
        hvy_alcohol, any_healthcare, no_doc_bc_cost, gen_hlth,
        ment_hlth, phys_hlth, diff_walk, sex, age, education, income,
    ]

    # ── TODO (Model Integration): Replace dummy block below ───────────────
    #
    # STEP 1 — Scale features:
    #   import numpy as np
    #   features_array = np.array(features).reshape(1, -1)
    #   features_scaled = scaler.transform(features_array)
    #
    # STEP 2 — Predict:
    #   risk_score = float(model.predict_proba(features_scaled)[0][1])
    #   risk_level = "High Risk" if risk_score >= 0.5 else "Low Risk"
    #
    # ── DUMMY implementation (remove when model is ready) ─────────────────
    risk_score = round(random.uniform(0.1, 0.9), 4)
    risk_level = "High Risk" if risk_score >= 0.5 else "Low Risk"
    # ── END DUMMY ──────────────────────────────────────────────────────────

    message = (
        "Based on the information you provided, you may be at elevated risk. "
        "Please consult a healthcare professional for a formal diagnosis."
        if risk_level == "High Risk"
        else "Your responses suggest a lower likelihood of diabetes. "
        "Keep up the healthy habits, and continue routine check-ups."
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

