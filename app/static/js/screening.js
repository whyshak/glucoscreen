/**
 * screening.js — Multi-step form logic for GlucoScreen
 *
 * Manages:
 *  - Step navigation & validation
 *  - Unit toggle (metric / imperial)
 *  - BMI live calculation
 *  - Yes/No pill interactions
 *  - GenHlth scale pills
 *  - Range slider track fill
 *  - Review step population
 *  - fetch('/predict') on submit → redirect to /result
 */

(function () {
  "use strict";

  /* ── State ────────────────────────────────────────────────────────────── */
  const SECTIONS = [
    "About you",
    "Demographics",
    "Health conditions",
    "Lifestyle",
    "Healthcare & Cost",
    "Wellbeing",
    "Background",
    "Review",
  ];
  const TOTAL_DATA_STEPS = SECTIONS.length - 2; // exclude intro (0) and review (last)

  let currentStep = 0;
  let submitting = false;

  // Form data store (mirrors backend field names)
  const form = {
    nickname:      "",
    unit:          "metric",
    heightCm:      "",
    heightFt:      "",
    heightIn:      "",
    weightKg:      "",
    weightLb:      "",
    age:           "",
    sex:           "",
    highBp:        null,
    highChol:      null,
    cholCheck:     null,
    smoker:        null,
    stroke:        null,
    heartDisease:  null,
    physActivity:  null,
    fruits:        null,
    veggies:       null,
    hvyAlcohol:    null,
    diffWalk:      null,
    anyHealthcare: null,
    noDocbcCost:   null,
    genHlth:       "",
    mentHlth:      0,
    physHlth:      0,
    education:     "",
    income:        "",
  };

  /* ── DOM Refs ─────────────────────────────────────────────────────────── */
  const stepPanes       = document.querySelectorAll(".step-pane");
  const progressHeader  = document.getElementById("progressHeader");
  const progressFill    = document.getElementById("progressFill");
  const progressName    = document.getElementById("progressStepName");
  const progressCounter = document.getElementById("progressCounter");
  const btnBack         = document.getElementById("btnBack");
  const btnNext         = document.getElementById("btnNext");
  const btnNextLabel    = document.getElementById("btnNextLabel");
  const btnSpinner      = document.getElementById("btnSpinner");
  const errorAlert      = document.getElementById("errorAlert");
  const loadingOverlay  = document.getElementById("loadingOverlay");

  /* ── Helpers ──────────────────────────────────────────────────────────── */
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  function computeBMI() {
    let h = 0, w = 0;
    if (form.unit === "metric") {
      h = parseFloat(form.heightCm) || 0;
      w = parseFloat(form.weightKg) || 0;
    } else {
      const ft = parseFloat(form.heightFt) || 0;
      const inch = parseFloat(form.heightIn) || 0;
      h = (ft * 12 + inch) * 2.54;
      w = (parseFloat(form.weightLb) || 0) * 0.453592;
    }
    if (h <= 0 || w <= 0) return null;
    return w / ((h / 100) ** 2);
  }

  function bmiCategory(bmi) {
    if (bmi < 18.5) return "Underweight";
    if (bmi < 25)   return "Normal weight";
    if (bmi < 30)   return "Overweight";
    return "Obese";
  }

  function fmtBool(v) {
    if (v === null || v === undefined || v === "") return "—";
    return v ? "Yes" : "No";
  }

  function updateSliderTrack(input) {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--pct", pct + "%");
  }

  /* ── Validation per step ──────────────────────────────────────────────── */
  function canProceed(step) {
    switch (step) {
      case 0: return true;  // intro — always ok
      case 1: {             // demographics
        const hOk = form.unit === "metric"
          ? !!form.heightCm
          : (!!form.heightFt || !!form.heightIn);
        const wOk = form.unit === "metric"
          ? parseFloat(form.weightKg) > 0
          : parseFloat(form.weightLb) > 0;
        const age = parseInt(form.age, 10);
        return age >= 10 && age <= 120 && form.sex !== "" && hOk && wOk;
      }
      case 2: // health conditions
        return [
          form.highBp, form.highChol, form.cholCheck,
          form.smoker, form.stroke, form.heartDisease,
        ].every((v) => v !== null);
      case 3: // lifestyle
        return [
          form.physActivity, form.fruits, form.veggies,
          form.hvyAlcohol, form.diffWalk,
        ].every((v) => v !== null);
      case 4: // healthcare & cost
        return form.anyHealthcare !== null && form.noDocbcCost !== null;
      case 5: // wellbeing
        return form.genHlth !== "";
      case 6: // background
        return form.education !== "" && form.income !== "";
      case 7: return true;  // review — always ok
      default: return true;
    }
  }

  /* ── Render step ──────────────────────────────────────────────────────── */
  function showStep(index) {
    stepPanes.forEach((pane, i) => {
      pane.style.display = i === index ? "block" : "none";
    });

    const isFirst   = index === 0;
    const isLast    = index === SECTIONS.length - 1;
    const dataStep  = clamp(index, 1, TOTAL_DATA_STEPS);

    // Progress header
    if (index === 0) {
      progressHeader.style.display = "none";
    } else {
      progressHeader.style.display = "block";
      const pct = Math.round((index / (SECTIONS.length - 1)) * 100);
      progressFill.style.width   = pct + "%";
      progressName.textContent   = SECTIONS[index];
      progressCounter.textContent = `Step ${index} of ${SECTIONS.length - 2}`;
    }

    // Back button
    btnBack.style.visibility = isFirst ? "hidden" : "visible";

    // Next / Submit button
    if (isLast) {
      btnNextLabel.textContent = "Submit";
      btnNext.classList.add("btn-submit");
    } else {
      btnNextLabel.textContent = "Continue";
      btnNext.classList.remove("btn-submit");
    }
    btnNext.disabled = !canProceed(index);

    // Populate review on step entry
    if (index === SECTIONS.length - 1) populateReview();

    // Error hidden on step change
    errorAlert.style.display = "none";
  }

  /* ── Navigation ───────────────────────────────────────────────────────── */
  btnBack.addEventListener("click", () => {
    if (currentStep > 0) { currentStep--; showStep(currentStep); }
  });

  btnNext.addEventListener("click", () => {
    if (!canProceed(currentStep)) return;
    const isLast = currentStep === SECTIONS.length - 1;
    if (isLast) {
      submitForm();
    } else {
      currentStep++;
      showStep(currentStep);
    }
  });

  /* ── Nickname (Step 0) ────────────────────────────────────────────────── */
  const nicknameInput = document.getElementById("nickname");
  if (nicknameInput) {
    nicknameInput.addEventListener("input", (e) => {
      form.nickname = e.target.value;
    });
  }

  /* ── Unit toggle (Step 1) ─────────────────────────────────────────────── */
  const unitBtns = document.querySelectorAll(".unit-toggle__btn");
  const metricFields   = document.getElementById("metricFields");
  const imperialFields = document.getElementById("imperialFields");

  unitBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      form.unit = btn.dataset.unit;
      unitBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      metricFields.style.display   = form.unit === "metric"   ? "block" : "none";
      imperialFields.style.display = form.unit === "imperial" ? "block" : "none";
      updateBMIDisplay();
      refreshNext();
    });
  });

  // Numeric inputs
  function bindNumericInput(id, key) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", (e) => {
      form[key] = e.target.value;
      updateBMIDisplay();
      refreshNext();
    });
  }
  bindNumericInput("heightCm", "heightCm");
  bindNumericInput("heightFt", "heightFt");
  bindNumericInput("heightIn", "heightIn");
  bindNumericInput("weightKg", "weightKg");
  bindNumericInput("weightLb", "weightLb");
  bindNumericInput("age", "age");

  // Sex toggle
  const sexBtns = document.querySelectorAll(".sex-pill");
  sexBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      form.sex = btn.dataset.val;
      sexBtns.forEach((b) => b.classList.remove("selected-yes"));
      btn.classList.add("selected-yes");
      refreshNext();
    });
  });

  function updateBMIDisplay() {
    const bmi = computeBMI();
    const badge = document.getElementById("bmiBadge");
    if (!badge) return;
    if (bmi) {
      badge.style.display = "inline-flex";
      badge.querySelector(".bmi-val").textContent =
        bmi.toFixed(1) + " — " + bmiCategory(bmi);
    } else {
      badge.style.display = "none";
    }
  }

  /* ── Yes/No pills (generic) ───────────────────────────────────────────── */
  document.querySelectorAll(".yn-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const key = pill.dataset.field;
      const val = pill.dataset.val === "true";
      form[key] = val;

      // Update sibling pills
      const group = pill.closest(".yn-pills");
      group.querySelectorAll(".yn-pill").forEach((p) => {
        p.classList.remove("selected-yes", "selected-no");
      });
      pill.classList.add(val ? "selected-yes" : "selected-no");
      refreshNext();
    });
  });

  /* ── GenHlth scale pills (Step 5) ────────────────────────────────────── */
  document.querySelectorAll(".scale-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      form.genHlth = pill.dataset.val;
      document.querySelectorAll(".scale-pill").forEach((p) => p.classList.remove("selected"));
      pill.classList.add("selected");
      refreshNext();
    });
  });

  /* ── Range sliders ────────────────────────────────────────────────────── */
  function bindSlider(id, key, displayId) {
    const el = document.getElementById(id);
    const display = document.getElementById(displayId);
    if (!el || !display) return;
    updateSliderTrack(el);
    el.addEventListener("input", () => {
      form[key] = parseInt(el.value, 10);
      display.textContent = el.value + " days";
      updateSliderTrack(el);
    });
  }
  bindSlider("mentHlthSlider", "mentHlth", "mentHlthVal");
  bindSlider("physHlthSlider", "physHlth", "physHlthVal");

  /* ── Select dropdowns (Step 6) ────────────────────────────────────────── */
  function bindSelect(id, key) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      form[key] = el.value;
      refreshNext();
    });
  }
  bindSelect("educationSelect", "education");
  bindSelect("incomeSelect", "income");

  /* ── Review population ────────────────────────────────────────────────── */
  function populateReview() {
    const bmi = computeBMI();
    const bmiText = bmi ? bmi.toFixed(1) + " (" + bmiCategory(bmi) + ")" : "—";
    const sexText = form.sex === "1" || form.sex === 1 ? "Male" : form.sex === "0" || form.sex === 0 ? "Female" : "—";

    const GENHLTH_LABELS = { "1": "Excellent", "2": "Very Good", "3": "Good", "4": "Fair", "5": "Poor" };
    const EDU_LABELS = {
      "1": "Never attended school",
      "2": "Elementary",
      "3": "Some high school",
      "4": "High school graduate",
      "5": "Some college",
      "6": "College graduate",
    };
    const INC_LABELS = {
      "1": "< $10,000", "2": "$10,000–$15,000", "3": "$15,000–$20,000",
      "4": "$20,000–$25,000", "5": "$25,000–$35,000",
      "6": "$35,000–$50,000", "7": "$50,000–$75,000", "8": "> $75,000",
    };

    const rows = {
      "rv-nickname":     form.nickname || "Not provided",
      "rv-age":          form.age || "—",
      "rv-sex":          sexText,
      "rv-bmi":          bmiText,
      "rv-highBp":       fmtBool(form.highBp),
      "rv-highChol":     fmtBool(form.highChol),
      "rv-cholCheck":    fmtBool(form.cholCheck),
      "rv-smoker":       fmtBool(form.smoker),
      "rv-stroke":       fmtBool(form.stroke),
      "rv-heartDisease": fmtBool(form.heartDisease),
      "rv-physActivity": fmtBool(form.physActivity),
      "rv-fruits":       fmtBool(form.fruits),
      "rv-veggies":      fmtBool(form.veggies),
      "rv-hvyAlcohol":   fmtBool(form.hvyAlcohol),
      "rv-diffWalk":     fmtBool(form.diffWalk),
      "rv-anyHealthcare": fmtBool(form.anyHealthcare),
      "rv-noDocbcCost":  fmtBool(form.noDocbcCost),
      "rv-genHlth":      GENHLTH_LABELS[form.genHlth] || "—",
      "rv-mentHlth":     form.mentHlth + " days",
      "rv-physHlth":     form.physHlth + " days",
      "rv-education":    EDU_LABELS[form.education] || "—",
      "rv-income":       INC_LABELS[form.income] || "—",
    };

    Object.entries(rows).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (!el) return;
      const isYes = val === "Yes";
      const isNo  = val === "No";
      if (isYes || isNo) {
        el.innerHTML = `<span class="review-badge review-badge--${isYes ? 'yes' : 'no'}">${val}</span>`;
      } else {
        el.textContent = val;
      }
    });
  }

  /* ── Submit ───────────────────────────────────────────────────────────── */
  async function submitForm() {
    if (submitting) return;
    submitting = true;
    errorAlert.style.display = "none";

    // Show loading
    loadingOverlay.classList.add("visible");
    btnNext.disabled = true;
    btnSpinner.style.display = "block";
    btnNextLabel.textContent = "Analysing…";

    const h = form.unit === "metric"
      ? parseFloat(form.heightCm) || 0
      : ((parseFloat(form.heightFt) || 0) * 12 + (parseFloat(form.heightIn) || 0)) * 2.54;
    const w = form.unit === "metric"
      ? parseFloat(form.weightKg) || 0
      : (parseFloat(form.weightLb) || 0) * 0.453592;

    const payload = {
      nickname:      form.nickname.trim() || null,
      age:           parseInt(form.age, 10),
      sex:           parseInt(form.sex, 10) === 1 ? 1 : 0,
      heightCm:      Math.round(h),
      weightKg:      Math.round(w * 10) / 10,
      highBp:        !!form.highBp,
      highChol:      !!form.highChol,
      cholCheck:     !!form.cholCheck,
      smoker:        !!form.smoker,
      stroke:        !!form.stroke,
      heartDisease:  !!form.heartDisease,
      physActivity:  !!form.physActivity,
      fruits:        !!form.fruits,
      veggies:       !!form.veggies,
      hvyAlcohol:    !!form.hvyAlcohol,
      anyHealthcare: !!form.anyHealthcare,
      noDocbcCost:   !!form.noDocbcCost,
      genHlth:       parseInt(form.genHlth, 10) || 3,
      mentHlth:      parseInt(form.mentHlth, 10) || 0,
      physHlth:      parseInt(form.physHlth, 10) || 0,
      diffWalk:      !!form.diffWalk,
      education:     parseInt(form.education, 10) || 4,
      income:        parseInt(form.income, 10) || 5,
    };

    try {
      const res = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || "Server error. Please try again.");
      }

      const data = await res.json();
      // Redirect to result page with query params including job_id & predicted_class
      const params = new URLSearchParams({
        risk_level: data.risk_level,
        risk_score: data.risk_score,
        job_id: data.job_id || "",
        predicted_class: data.predicted_class !== undefined ? data.predicted_class : 1,
      });
      window.location.href = "/result?" + params.toString();

    } catch (err) {
      loadingOverlay.classList.remove("visible");
      btnNext.disabled = false;
      btnSpinner.style.display = "none";
      btnNextLabel.textContent = "Submit";
      errorAlert.textContent = err.message || "Something went wrong.";
      errorAlert.style.display = "flex";
      submitting = false;
    }
  }

  /* ── Refresh next button state ────────────────────────────────────────── */
  function refreshNext() {
    btnNext.disabled = !canProceed(currentStep);
  }

  /* ── Init ─────────────────────────────────────────────────────────────── */
  showStep(0);
  metricFields.style.display   = "block";
  imperialFields.style.display = "none";

})();
