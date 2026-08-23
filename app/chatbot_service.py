"""
chatbot_service.py — Dia Chatbot Assistant Service.

Provides intent scoring, educational responses, and response generation for Dia,
the conversational diabetes assistant. Also defines constants and interfaces for
the upcoming Retrieval-Augmented Generation (RAG) vector store pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DISCLOSURE = (
    "\n\n_This is general educational information, not medical advice. "
    "Always consult your doctor for personal guidance._"
)

INTENTS: List[Dict[str, Any]] = [
    {
        "id": "greeting",
        "keywords": ["hello", "hi", "hey", "good morning", "good evening", "greetings", "howdy", "yo "],
        "reply": (
            "Hello! 👋 I'm Dia, your diabetes assistant. I can answer questions about "
            "diabetes risk, symptoms, diet plans, exercise, blood sugar and more.\n\n"
            "What would you like to know today?"
        ),
        "suggestions": [
            "Am I at risk of diabetes?",
            "What should I eat?",
            "Best exercises for diabetes",
            "What are the symptoms?",
        ],
    },
    {
        "id": "what_is",
        "keywords": [
            "what is diabetes", "define diabetes", "meaning of diabetes",
            "types of diabetes", "type 1", "type 2", "gestational",
            "difference between type",
        ],
        "reply": (
            "**Diabetes** is a chronic condition where your blood glucose (sugar) is too high "
            "because the body can't make or use insulin effectively.\n\n"
            "There are three main types:\n"
            "• **Type 1** — the body produces little or no insulin (autoimmune). Needs daily insulin.\n"
            "• **Type 2** — the body becomes resistant to insulin or doesn't make enough. ~90% of cases and often preventable.\n"
            "• **Gestational** — develops during pregnancy and usually resolves after birth, though it raises future risk."
            + DISCLOSURE
        ),
        "suggestions": ["What causes Type 2?", "What are the symptoms?", "How is it diagnosed?"],
    },
    {
        "id": "symptoms",
        "keywords": [
            "symptom", "signs", "feel", "thirsty", "urinate", "pee",
            "tired", "fatigue", "blurry vision", "weight loss", "warning",
        ],
        "reply": (
            "**Common symptoms of diabetes** include:\n"
            "• Excessive thirst 💧\n"
            "• Frequent urination 🚽\n"
            "• Extreme tiredness / fatigue 😴\n"
            "• Blurred vision 👁️\n"
            "• Unexplained weight loss\n"
            "• Slow-healing cuts or sores\n"
            "• Tingling or numbness in hands/feet\n\n"
            "Type 1 symptoms usually appear quickly; Type 2 can develop gradually and go unnoticed for years. "
            "If you have several of these, please get tested."
            + DISCLOSURE
        ),
        "suggestions": ["How is diabetes diagnosed?", "Should I take the screening test?"],
    },
    {
        "id": "diet",
        "keywords": [
            "diet", "food", "eat", "nutrition", "meal", "carb", "sugar",
            "vegetable", "fruit", "recipe", "what should i eat", "grocery", "snack",
        ],
        "reply": (
            "🥗 **A diabetes-friendly plate** focuses on fibre, lean protein and healthy fats while limiting refined carbs.\n\n"
            "**Foods to enjoy:**\n"
            "• Non-starchy vegetables (leafy greens, broccoli, peppers)\n"
            "• Whole grains (oats, brown rice, quinoa)\n"
            "• Lean protein (fish, chicken, beans, tofu, eggs)\n"
            "• Healthy fats (olive oil, nuts, avocado)\n"
            "• Berries and low-GI fruit in moderation\n\n"
            "**Limit or avoid:**\n"
            "• Sugary drinks, sweets and pastries\n"
            "• White bread, white rice, refined flour\n"
            "• Fried and ultra-processed foods\n"
            "• Fruit juice (even unsweetened)\n\n"
            "**Plate method:** Fill ½ with veggies, ¼ with lean protein, ¼ with complex carbs."
            + DISCLOSURE
        ),
        "suggestions": ["Give me a sample day meal plan", "Can I eat fruit?", "What about drinking water?"],
    },
    {
        "id": "meal_plan",
        "keywords": [
            "meal plan", "sample", "breakfast", "lunch", "dinner",
            "menu", "plan for the day", "what do i eat in a day",
        ],
        "reply": (
            "🍽️ **Sample diabetes-friendly day**\n\n"
            "**Breakfast:** Veggie omelette (2 eggs + spinach + tomato) with a small bowl of oats and berries.\n\n"
            "**Lunch:** Grilled chicken salad with mixed greens, chickpeas, cucumber, olive-oil dressing + 1 slice whole-grain bread.\n\n"
            "**Snack:** A small handful of unsalted almonds + an apple.\n\n"
            "**Dinner:** Baked salmon, roasted broccoli and ½ cup quinoa.\n\n"
            "**Tips:** Eat at consistent times, watch portion sizes, and pair carbs with protein or fibre to avoid spikes."
            + DISCLOSURE
        ),
        "suggestions": ["Best exercises for diabetes", "How do I manage blood sugar?"],
    },
    {
        "id": "exercise",
        "keywords": [
            "exercise", "workout", "physical activity", "gym", "walk", "running",
            "cardio", "strength", "yoga", "fitness", "move", "training", "diabetes exercise",
        ],
        "reply": (
            "🏃 **Exercise is one of the most powerful tools** for managing and preventing diabetes — "
            "it makes your body more sensitive to insulin.\n\n"
            "**Aim for:** 150 minutes of moderate activity per week + 2 strength sessions.\n\n"
            "**Great options:**\n"
            "• Brisk walking (30 min/day) — the easiest start\n"
            "• Cycling or swimming — easy on the joints\n"
            "• Body-weight strength training (squats, push-ups, bands)\n"
            "• Yoga or tai chi — great for stress and balance\n\n"
            "**Safety:** Check blood sugar before/after if on medication, stay hydrated, and keep a fast-acting carb nearby in case of lows. "
            "Even 10-minute walks after meals help a lot!"
            + DISCLOSURE
        ),
        "suggestions": ["How do I start if I'm out of shape?", "Can exercise lower blood sugar?"],
    },
    {
        "id": "blood_sugar",
        "keywords": [
            "blood sugar", "glucose", "sugar level", "normal range", "spike",
            "high sugar", "low sugar", "fasting", "post meal", "reading", "readings",
        ],
        "reply": (
            "🩸 **Blood sugar targets (general, non-pregnant adults):**\n"
            "• **Fasting (before meals):** 80–130 mg/dL (4.4–7.2 mmol/L)\n"
            "• **1–2 hours after meals:** below 180 mg/dL (10 mmol/L)\n"
            "• **A1C:** below 7% for many adults (reflects ~3-month average)\n\n"
            "**To avoid spikes:** eat fibre and protein first, watch portions, move after meals, manage stress and sleep.\n\n"
            "Your personal targets may differ — ask your doctor what's right for you."
            + DISCLOSURE
        ),
        "suggestions": ["What is A1C?", "Foods that don't spike blood sugar"],
    },
    {
        "id": "a1c",
        "keywords": ["a1c", "hba1c", "hba1", "glycated", "three month average"],
        "reply": (
            "**A1C (HbA1c)** measures your average blood sugar over the past 2–3 months.\n\n"
            "• **Below 5.7%** — Normal\n"
            "• **5.7–6.4%** — Prediabetes\n"
            "• **6.5% or higher** — Diabetes\n\n"
            "It's a key diagnostic test because it isn't affected by a single meal. "
            "People with diabetes often aim for **below 7%**, but your doctor will set a personalized goal."
            + DISCLOSURE
        ),
        "suggestions": ["What is prediabetes?", "How is diabetes diagnosed?"],
    },
    {
        "id": "diagnosis",
        "keywords": [
            "diagnos", "test", "testing", "how do i know", "check",
            "confirm", "fpg", "ogtt", "random glucose",
        ],
        "reply": (
            "🧪 **Diabetes is diagnosed with blood tests:**\n"
            "• **Fasting Plasma Glucose (FPG):** ≥ 126 mg/dL after 8h fasting\n"
            "• **A1C:** ≥ 6.5%\n"
            "• **Oral Glucose Tolerance Test (OGTT):** ≥ 200 mg/dL 2h after a sugary drink\n"
            "• **Random glucose:** ≥ 200 mg/dL with symptoms\n\n"
            "Abnormal results are usually confirmed with a second test. You can start by taking our **free risk screening** — "
            "it flags whether you'd benefit from a lab test."
            + DISCLOSURE
        ),
        "suggestions": ["Take the screening test", "What is prediabetes?"],
    },
    {
        "id": "prediabetes",
        "keywords": ["prediabetes", "pre-diabetes", "borderline", "prevent", "reverse"],
        "reply": (
            "**Prediabetes** means blood sugar is above normal but not yet in the diabetes range (A1C 5.7–6.4%). "
            "It's a warning sign — and a powerful opportunity.\n\n"
            "✅ **It can often be reversed** with lifestyle changes:\n"
            "• Lose 5–7% of body weight\n"
            "• 150 min/week of activity\n"
            "• Eat more whole foods, fewer refined carbs\n"
            "• Prioritise sleep and manage stress\n\n"
            "The CDC's Diabetes Prevention Program showed these steps cut progression to Type 2 by ~58%."
            + DISCLOSURE
        ),
        "suggestions": ["Best exercises to prevent diabetes", "What should I eat?"],
    },
    {
        "id": "causes",
        "keywords": [
            "cause", "risk factor", "why do people get", "reason",
            "genetics", "family history", "family",
        ],
        "reply": (
            "**Key risk factors for Type 2 diabetes:**\n"
            "• Overweight / high waist size\n"
            "• Age 45+ (though younger people are increasingly affected)\n"
            "• Family history / genetics\n"
            "• Physical inactivity\n"
            "• High blood pressure & cholesterol\n"
            "• History of gestational diabetes\n"
            "• Poor diet high in refined carbs/sugar\n"
            "• Smoking, stress and poor sleep\n\n"
            "You can't change age or genetics — but weight, activity, diet and sleep are very much in your hands."
            + DISCLOSURE
        ),
        "suggestions": ["Take the screening test", "How do I prevent diabetes?"],
    },
    {
        "id": "weight",
        "keywords": [
            "weight", "lose weight", "bmi", "obesity", "obese",
            "belly fat", "waist", "slim down",
        ],
        "reply": (
            "⚖️ **Weight management** is one of the strongest levers for lowering diabetes risk. "
            "Losing just **5–10% of body weight** can dramatically improve blood sugar.\n\n"
            "**Sustainable approach:**\n"
            "• Create a modest calorie deficit (300–500 kcal/day)\n"
            "• Prioritise protein and fibre to stay full\n"
            "• Add strength training to preserve muscle\n"
            "• Aim for 7–9h sleep (poor sleep raises hunger hormones)\n"
            "• Track progress with habits, not just the scale\n\n"
            "A waistline over 40in (men) / 35in (women) signals higher insulin-resistance risk."
            + DISCLOSURE
        ),
        "suggestions": ["Best exercises for weight loss", "What should I eat?"],
    },
    {
        "id": "complications",
        "keywords": [
            "complication", "danger", "long term", "kidney", "heart", "nerve",
            "eye", "blind", "amputation", "neuropathy", "foot", "stroke",
        ],
        "reply": (
            "⚠️ **Untreated, high blood sugar can damage blood vessels over time.** Long-term complications include:\n"
            "• **Heart disease & stroke** (most common)\n"
            "• **Kidney damage** (nephropathy)\n"
            "• **Nerve damage** (neuropathy) — tingling, pain, foot problems\n"
            "• **Eye damage** (retinopathy) — can lead to blindness\n"
            "• **Slow wound healing** and infections\n\n"
            "**Good news:** tight blood-sugar control, blood pressure and cholesterol management, "
            "regular screenings, and healthy habits prevent or delay most of these."
            + DISCLOSURE
        ),
        "suggestions": ["How do I prevent complications?", "What is A1C?"],
    },
    {
        "id": "medication",
        "keywords": [
            "medication", "medicine", "metformin", "insulin", "drug",
            "pills", "tablet", "prescription", "treatment",
        ],
        "reply": (
            "💊 **Common diabetes treatments** (managed by your doctor):\n"
            "• **Metformin** — usually the first medication for Type 2; improves insulin sensitivity\n"
            "• **Other oral/injectable meds** — GLP-1s, SGLT2 inhibitors, etc., with different mechanisms\n"
            "• **Insulin** — essential for Type 1 and sometimes Type 2\n\n"
            "Medication works best *combined* with lifestyle changes — not instead of them. "
            "Never start, stop or adjust medication without your prescriber."
            + DISCLOSURE
        ),
        "suggestions": ["What should I eat while on metformin?", "Natural ways to lower blood sugar"],
    },
    {
        "id": "hypoglycemia",
        "keywords": [
            "hypoglycemia", "low blood sugar", "hypoglycaemia",
            "shaky", "dizzy", "low sugar", "hypo",
        ],
        "reply": (
            "🔻 **Low blood sugar (hypoglycemia, below 70 mg/dL)** can cause shakiness, sweating, dizziness, "
            "confusion, hunger and a fast heartbeat.\n\n"
            "**The 15-15 rule:**\n"
            "1. Eat **15g fast carbs** (½ cup juice, 3–4 glucose tablets, 1 tbsp sugar)\n"
            "2. Wait **15 minutes**, then re-check\n"
            "3. Repeat if still low\n"
            "4. Once normal, eat a small snack with protein/carbs if your next meal is far off\n\n"
            "Severe lows can cause unconsciousness — inform family about glucagon and call emergency services if needed."
            + DISCLOSURE
        ),
        "suggestions": ["What is a normal blood sugar range?", "Foods that don't spike blood sugar"],
    },
    {
        "id": "screening_help",
        "keywords": [
            "screening", "test myself", "quiz", "questionnaire", "how does this work",
            "assessment", "am i at risk", "take the test", "predict", "tool", "app", "score",
        ],
        "reply": (
            "🧭 **Our screening tool** estimates your diabetes risk using validated health indicators "
            "(age, BMI, blood pressure, activity, diet, family health and more).\n\n"
            "It takes about 3 minutes and gives you:\n"
            "• A personalised risk level\n"
            "• The factors influencing your score\n"
            "• Actionable recommendations\n\n"
            "👉 Tap **\"Start Screening\"** on the navigation bar. Remember, it's an estimate to guide you — "
            "not a diagnosis. Confirm with a blood test from your doctor."
            + DISCLOSURE
        ),
        "suggestions": ["What is a normal blood sugar range?", "What should I eat?"],
    },
    {
        "id": "water",
        "keywords": ["water", "drink water", "hydration", "hydrate", "how much water"],
        "reply": (
            "💧 **Yes — staying hydrated helps!** Water is the best drink for blood sugar control:\n"
            "• It helps your kidneys flush excess sugar through urine\n"
            "• Replaces sugary drinks that spike glucose\n"
            "• Aim for ~6–8 glasses/day (more in heat/exercise)\n\n"
            "Add a slice of lemon, cucumber or mint for flavour. Avoid soda, sweet tea and fruit juice."
            + DISCLOSURE
        ),
        "suggestions": ["What should I drink instead of soda?", "Sample meal plan"],
    },
    {
        "id": "stress_sleep",
        "keywords": [
            "stress", "sleep", "anxiety", "tired", "rest", "insomnia",
            "mental health", "relax", "meditate",
        ],
        "reply": (
            "🧘 **Stress and sleep quietly raise blood sugar.** Stress hormones (cortisol, adrenaline) "
            "push glucose up, and poor sleep reduces insulin sensitivity.\n\n"
            "**Helpful habits:**\n"
            "• 7–9h of consistent sleep\n"
            "• Deep breathing, meditation or yoga (even 5–10 min)\n"
            "• Regular movement and time outdoors\n"
            "• Limit caffeine late in the day & screens before bed\n"
            "• Connect with people you trust\n\n"
            "Managing stress isn't a luxury — it's part of diabetes care."
            + DISCLOSURE
        ),
        "suggestions": ["Best exercises for diabetes", "What should I eat?"],
    },
    {
        "id": "fruit",
        "keywords": ["fruit", "banana", "mango", "apple", "berries", "can i eat fruit", "fruits"],
        "reply": (
            "🍎 **Yes — you can eat fruit!** Fruit contains natural sugar, but its fibre slows absorption.\n\n"
            "**Best choices (lower GI):** berries, apples, pears, cherries, kiwi, oranges, plums.\n\n"
            "**Enjoy in moderation:** bananas, grapes, mango, pineapple (smaller portions, paired with protein/fat).\n\n"
            "**Avoid:** fruit juice and dried fruit with added sugar — these concentrate sugar without fibre."
            + DISCLOSURE
        ),
        "suggestions": ["Sample meal plan", "Foods that don't spike blood sugar"],
    },
    {
        "id": "thanks",
        "keywords": ["thank", "thanks", "appreciate", "cheers", "great", "awesome", "helpful"],
        "reply": (
            "You're very welcome! 😊 I'm glad I could help. Remember: small, consistent steps make a big difference over time.\n\n"
            "Is there anything else you'd like to ask — about diet, exercise, or your screening results?"
        ),
        "suggestions": ["What should I eat?", "Best exercises for diabetes", "Take the screening test"],
    },
    {
        "id": "bye",
        "keywords": ["bye", "goodbye", "see you", "later", "that's all", "good night"],
        "reply": (
            "Take care of yourself! 💚 Wishing you good health. "
            "I'm always here if you have more questions — just come back and chat anytime."
        ),
        "suggestions": ["Start a new chat", "Take the screening test"],
    },
]

FALLBACK_INTENT: Dict[str, Any] = {
    "id": "fallback",
    "keywords": [],
    "reply": (
        "I'm not quite sure I caught that. 🤔 I can help with things like:\n"
        "• Diabetes risk, symptoms & diagnosis\n"
        "• Diet plans and what to eat\n"
        "• Exercises and blood-sugar management\n"
        "• Weight, sleep and prevention\n\n"
        "Try one of the suggestions below, or ask me in your own words!"
    ),
    "suggestions": [
        "Am I at risk of diabetes?",
        "What should I eat?",
        "Best exercises for diabetes",
        "What are the symptoms?",
    ],
}

WELCOME_MESSAGE: Dict[str, Any] = {
    "role": "assistant",
    "text": INTENTS[0]["reply"],
    "suggestions": INTENTS[0]["suggestions"],
}

QUICK_PROMPTS: List[str] = [
    "What are the symptoms of diabetes?",
    "What should I eat for breakfast?",
    "How much should I exercise?",
    "What is a normal blood sugar level?",
    "How can I prevent diabetes?",
]


def score_intent(message: str) -> Dict[str, Any]:
    """
    Score user query against known intent keywords.
    Returns best-matching intent or fallback intent.
    """
    msg = f" {message.lower().strip()} "
    best_intent = None
    best_score = 0.0

    for intent in INTENTS:
        score = 0.0
        for kw in intent["keywords"]:
            if kw in msg:
                # Weight longer multi-word phrases higher
                score += len(kw.split()) + (len(kw) / 10.0)
        if score > 0 and score > best_score:
            best_score = score
            best_intent = intent

    return best_intent if best_intent is not None else FALLBACK_INTENT


def generate_mock_chat_response(message: str) -> Dict[str, Any]:
    """
    Generate mock conversational response for the frontend UI.
    This serves as the temporary placeholder until the full RAG pipeline
    (vector database + embeddings + LLM) is connected.
    """
    matched = score_intent(message)
    return {
        "text": matched["reply"],
        "suggestions": matched.get("suggestions", []),
        "intent_id": matched["id"],
    }
