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
    return render_template("index.html", model_performance=get_model_performance_summary())



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



def get_model_performance_summary() -> dict:
    """
    Returns a plain-English translation of the SVM model evaluation metrics
    and dataset provenance for end-user display in the UI.

    TODO: When updating backend model benchmarks or re-training on new CDC data,
          update the values below or load them dynamically from model metadata.
    """
    return {
        "overall_reliability": {
            "title": "Overall Reliability",
            "badge": "High Accuracy",
            "score_pct": 84,
            "summary": "Evaluates risk patterns correctly for ~84 out of 100 individuals.",
            "description": "Out of 100 people screened, the model accurately categorizes risk patterns for 84 of them based on validated public health survey data.",
        },
        "detection_rate": {
            "title": "Early Risk Detection",
            "badge": "High Sensitivity",
            "score_pct": 81,
            "summary": "Successfully identifies ~81% of individuals at true risk.",
            "description": "The model prioritizes early detection so that individuals at potential risk are flagged promptly for follow-up evaluation.",
        },
        "precautionary_balance": {
            "title": "Targeted Precision",
            "badge": "Safety First",
            "score_pct": 65,
            "summary": "Designed to encourage timely medical advice without missing warning signs.",
            "description": "As a preliminary prescreening tool, the model errs on the side of caution. Moderate/high flags prompt proactive check-ups with a doctor.",
        },
        "dataset_info": {
            "model_type": "Support Vector Machine (RBF Kernel)",
            "sample_size": "253,680 records",
            "source": "CDC BRFSS Survey Dataset",
            "validation_note": "Trained and cross-validated on standardized CDC national health indicator data representing diverse age, demographic, and lifestyle groups.",
        },
    }


# =============================================================================
# CHATBOT RAG ENGINE & ENDPOINTS
# =============================================================================
_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        from app.rag import DiabetesRAGEngine
        _rag_engine = DiabetesRAGEngine()
    return _rag_engine


@main.route("/api/chat", methods=["POST"])
def api_chat():
    """
    RAG-powered Chatbot Endpoint.
    Expects JSON: { "message": "string" }
    Returns JSON: { "answer": "string", "sources": list, "suggested_questions": list }
    """
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "answer": "Please type a question about diabetes risk, symptoms, or health guidance.",
            "sources": [],
            "suggested_questions": [
                "What are the top risk factors for diabetes?",
                "How can I lower my blood sugar naturally?",
                "What does my screening risk score mean?"
            ]
        }), 400

    # Build session dictionary for user-specific context
    user_session = {}
    if "prediction_result" in session:
        res = session["prediction_result"]
        user_session["prediction_score"] = res.get("prob_percent")
        user_session["risk_level"] = res.get("risk_level")

    engine = get_rag_engine()
    result = engine.answer_question(user_message, user_session=user_session)
    return jsonify(result)



@main.route("/api/chat/suggested", methods=["GET"])
def api_chat_suggested():
    """
    Returns initial dynamic suggested prompt pills based on user screening state.
    """
    if "prediction_result" in session:
        res = session["prediction_result"]
        risk_level = res.get("risk_level", "Moderate")
        return jsonify({
            "suggested_questions": [
                f"What does my {risk_level} Risk score mean?",
                "What lifestyle changes should I make first?",
                "What medical tests should I request from my doctor?",
                "What foods help lower blood sugar?"
            ]
        })
    else:
        return jsonify({
            "suggested_questions": [
                "What are the main risk factors for diabetes?",
                "How does the GlucoScreen AI model calculate risk?",
                "What foods lower diabetes risk?",
                "What are early warning signs of prediabetes?"
            ]
        })


