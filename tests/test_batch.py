"""
test_batch.py — Unit and integration tests for Batch CSV/Excel prediction.
"""

import io
import unittest
import pandas as pd
from app import create_app
from app.batch_service import FEATURE_COLUMNS, generate_template


class BatchPredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test-secret-key"
        self.client = self.app.test_client()

    # ── 1. Batch UI & Navigation Tests ─────────────────────────────────────────

    def test_batch_page_renders(self):
        """Test GET /batch renders the batch screening page successfully."""
        response = self.client.get("/batch")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Batch Diabetes Risk Screening", response.data)
        self.assertIn(b"Download Template", response.data)
        self.assertIn(b"Upload Dataset", response.data)

    def test_single_screening_page_intact(self):
        """Verify existing /screening route is still operational."""
        response = self.client.get("/screening")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to GlucoScreen", response.data)

    # ── 2. Template Download Route Tests ──────────────────────────────────────

    def test_download_blank_csv_template(self):
        """Test GET /batch/template?format=csv returns CSV file with 21 headers."""
        response = self.client.get("/batch/template?format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("glucoscreen_batch_template.csv", response.headers.get("Content-Disposition", ""))

        df = pd.read_csv(io.BytesIO(response.data))
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_download_blank_xlsx_template(self):
        """Test GET /batch/template?format=xlsx returns valid Excel file."""
        response = self.client.get("/batch/template?format=xlsx")
        self.assertEqual(response.status_code, 200)
        self.assertIn("spreadsheetml", response.mimetype)
        self.assertIn("glucoscreen_batch_template.xlsx", response.headers.get("Content-Disposition", ""))

        df = pd.read_excel(io.BytesIO(response.data))
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_download_sample_template(self):
        """Test GET /batch/template?format=csv&sample=1 returns pre-filled sample rows."""
        response = self.client.get("/batch/template?format=csv&sample=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("glucoscreen_batch_sample.csv", response.headers.get("Content-Disposition", ""))

        df = pd.read_csv(io.BytesIO(response.data))
        self.assertEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertGreater(len(df), 0)

    # ── 3. Batch Prediction Tests (CSV & Excel) ───────────────────────────────

    def test_batch_predict_valid_csv(self):
        """Test POST /batch/predict with a valid CSV file."""
        csv_bytes, _, _ = generate_template("csv", sample=True)
        data = {
            "file": (io.BytesIO(csv_bytes), "test_cohort.csv"),
        }
        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        self.assertTrue(json_data["success"])
        self.assertIn("batch_id", json_data)
        self.assertIn("summary", json_data)
        self.assertIn("rows", json_data)

        summary = json_data["summary"]
        self.assertEqual(summary["total_records"], 5)
        self.assertIn("high_risk_count", summary)
        self.assertIn("low_risk_count", summary)
        self.assertIn("avg_risk_score_pct", summary)

        rows = json_data["rows"]
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertIn("risk_score", r)
            self.assertIn("risk_level", r)
            self.assertIn(r["risk_level"], ["Low Risk", "Moderate Risk", "High Risk"])
            self.assertIn("features", r)

    def test_batch_predict_valid_xlsx(self):
        """Test POST /batch/predict with a valid Excel file."""
        xlsx_bytes, _, _ = generate_template("xlsx", sample=True)
        data = {
            "file": (io.BytesIO(xlsx_bytes), "test_cohort.xlsx"),
        }
        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["summary"]["total_records"], 5)

    def test_batch_predict_flexible_column_names_and_values(self):
        """Test coercion of friendly names (e.g. high_bp, gender, age in years)."""
        df_custom = pd.DataFrame([
            {
                "high_bp": "yes",
                "high_chol": "1",
                "chol_check": "True",
                "bmi": 29.5,
                "smoker": "no",
                "stroke": "0",
                "heart_disease": "0",
                "phys_activity": "1",
                "fruit": "1",
                "veg": "1",
                "heavy_alcohol": "0",
                "healthcare": "1",
                "no_doc_cost": "0",
                "general_health": "Good",
                "mental_health": 2.0,
                "physical_health": 0.0,
                "diff_walk": "no",
                "gender": "male",
                "age": 45,  # Raw age in years -> maps to age group 6
                "education_level": 5,
                "income_level": 6,
            }
        ])
        csv_bytes = df_custom.to_csv(index=False).encode("utf-8")
        data = {"file": (io.BytesIO(csv_bytes), "custom_cohort.csv")}

        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["summary"]["total_records"], 1)
        self.assertEqual(json_data["rows"][0]["features"]["Age"], 6)

    # ── 4. Validation & Error Handling Tests ────────────────────────────────────

    def test_batch_predict_missing_columns(self):
        """Test POST /batch/predict fails when columns are missing."""
        df_bad = pd.DataFrame([{"BMI": 25.0, "Sex": 1, "Age": 5}])
        csv_bytes = df_bad.to_csv(index=False).encode("utf-8")
        data = {"file": (io.BytesIO(csv_bytes), "incomplete.csv")}

        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])
        self.assertTrue(any("Missing" in err for err in json_data["errors"]))

    def test_batch_predict_invalid_data_type(self):
        """Test POST /batch/predict flags invalid row values."""
        df_invalid = pd.DataFrame([
            {
                "HighBP": 1, "HighChol": 1, "CholCheck": 1,
                "BMI": "INVALID_BMI",
                "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
                "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
                "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
                "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
                "Sex": 1, "Age": 5, "Education": 5, "Income": 7,
            }
        ])
        csv_bytes = df_invalid.to_csv(index=False).encode("utf-8")
        data = {"file": (io.BytesIO(csv_bytes), "bad_data.csv")}

        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])
        self.assertTrue(any("Row 1: BMI must be a number" in err for err in json_data["errors"]))

    def test_batch_predict_empty_file(self):
        """Test POST /batch/predict fails when empty file is uploaded."""
        data = {"file": (io.BytesIO(b""), "empty.csv")}
        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_batch_predict_unsupported_extension(self):
        """Test POST /batch/predict rejects .txt or .pdf files."""
        data = {"file": (io.BytesIO(b"dummy text content"), "dataset.txt")}
        response = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertTrue(any("Unsupported file format" in err for err in json_data["errors"]))

    # ── 5. Results Export Tests ────────────────────────────────────────────────

    def test_export_results_csv_and_xlsx(self):
        """Test GET /batch/export generates downloadable CSV and Excel files."""
        # 1. Run batch prediction first to generate a batch_id
        csv_bytes, _, _ = generate_template("csv", sample=True)
        data = {"file": (io.BytesIO(csv_bytes), "sample.csv")}
        pred_resp = self.client.post("/batch/predict", data=data, content_type="multipart/form-data")
        batch_id = pred_resp.get_json()["batch_id"]

        # 2. Export as CSV
        export_csv_resp = self.client.get(f"/batch/export?batch_id={batch_id}&format=csv")
        self.assertEqual(export_csv_resp.status_code, 200)
        self.assertEqual(export_csv_resp.mimetype, "text/csv")
        df_out_csv = pd.read_csv(io.BytesIO(export_csv_resp.data))
        self.assertIn("Risk Score (%)", df_out_csv.columns)
        self.assertIn("Risk Category", df_out_csv.columns)
        self.assertIn("Predicted Class", df_out_csv.columns)
        self.assertEqual(len(df_out_csv), 5)

        # 3. Export as Excel
        export_xlsx_resp = self.client.get(f"/batch/export?batch_id={batch_id}&format=xlsx")
        self.assertEqual(export_xlsx_resp.status_code, 200)
        self.assertIn("spreadsheetml", export_xlsx_resp.mimetype)
        df_out_xlsx = pd.read_excel(io.BytesIO(export_xlsx_resp.data))
        self.assertIn("Risk Score (%)", df_out_xlsx.columns)
        self.assertEqual(len(df_out_xlsx), 5)

    def test_export_results_invalid_batch_id(self):
        """Test GET /batch/export returns 404 for invalid or non-existent batch_id."""
        response = self.client.get("/batch/export?batch_id=non-existent-id&format=csv")
        self.assertEqual(response.status_code, 404)


    def test_navbar_grouped_dropdown(self):
        """Test navbar contains grouped Start Screening dropdown with single and batch options."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"screeningDropdown", response.data)
        self.assertIn(b"Single Screening", response.data)
        self.assertIn(b"Batch Screening", response.data)

    def test_batch_guide_breakdowns_present(self):
        """Test that Age, GenHlth, Education, and Income value breakdowns are present in batch page."""
        response = self.client.get("/batch")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"18\xe2\x80\x9324 yrs", response.data)  # Age group 1: 18–24 yrs
        self.assertIn(b"80+ yrs", response.data)        # Age group 13: 80+ yrs
        self.assertIn(b"Excellent", response.data)      # GenHlth 1
        self.assertIn(b"College graduate", response.data) # Education 6
        self.assertIn(b"Less than $10,000", response.data) # Income 1


if __name__ == "__main__":
    unittest.main()
