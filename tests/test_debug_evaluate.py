"""
test_debug_evaluate.py — Unit and integration tests for Developer-Only Test Data Evaluation.

Tests security guards, hidden access constraints, labeled dataset parsing (X and y),
vectorized model inference, classification metrics, confusion matrix, and export downloads.
"""

import io
import os
import unittest
import pandas as pd

from app import create_app
from app.batch_service import FEATURE_COLUMNS
from app.evaluation_service import (
    generate_labeled_template,
    parse_and_validate_labeled_file,
    run_model_evaluation,
)


class DebugEvaluateTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["DEBUG"] = True
        self.app.config["SECRET_KEY"] = "test-eval-secret-key"
        self.client = self.app.test_client()

    # ── 1. Security & Environment Guard Tests ─────────────────────────────────

    def test_debug_evaluate_denied_in_production_mode(self):
        """Verify that /debug/evaluate immediately aborts with 404 when debug is disabled (simulating production)."""
        prod_app = create_app()
        prod_app.config["TESTING"] = False
        prod_app.config["DEBUG"] = False
        prod_client = prod_app.test_client()

        response = prod_client.get("/debug/evaluate")
        self.assertEqual(response.status_code, 404)

        # POST also denied
        response_post = prod_client.post("/debug/evaluate")
        self.assertEqual(response_post.status_code, 404)

        # Template & Export also denied
        response_template = prod_client.get("/debug/evaluate/template")
        self.assertEqual(response_template.status_code, 404)

        response_export = prod_client.get("/debug/evaluate/export")
        self.assertEqual(response_export.status_code, 404)

    def test_debug_evaluate_denied_from_non_local_ip(self):
        """Verify that /debug/evaluate aborts with 404 when request originates from an external/remote IP."""
        response = self.client.get(
            "/debug/evaluate",
            environ_base={"REMOTE_ADDR": "198.51.100.24", "HTTP_HOST": "glucoscreen.com"},
        )
        self.assertEqual(response.status_code, 404)

    def test_debug_evaluate_accessible_locally_in_debug_mode(self):
        """Verify that GET /debug/evaluate loads successfully when running locally in debug mode."""
        response = self.client.get("/debug/evaluate")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Test Dataset Evaluation", response.data)
        self.assertIn(b"DEVELOPER MODE ONLY", response.data)
        self.assertIn(b"Upload Labeled Test Dataset", response.data)

    def test_no_ui_links_in_public_templates(self):
        """Verify that no public user-facing templates contain links to /debug/evaluate."""
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "app", "templates")
        public_templates = ["_base.html", "index.html", "screening.html", "batch.html", "result.html"]

        for tmpl_name in public_templates:
            tmpl_path = os.path.join(templates_dir, tmpl_name)
            if os.path.exists(tmpl_path):
                with open(tmpl_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertNotIn(
                        "debug/evaluate",
                        content,
                        f"Found reference to debug/evaluate in public template: {tmpl_name}",
                    )
                    self.assertNotIn(
                        "debug_evaluate",
                        content,
                        f"Found url_for('main.debug_evaluate') in public template: {tmpl_name}",
                    )

    # ── 2. Labeled Template Generation & Download Tests ───────────────────────

    def test_download_sample_labeled_csv_template(self):
        """Test GET /debug/evaluate/template returns CSV with 21 features + Diabetes_binary target column."""
        response = self.client.get("/debug/evaluate/template?format=csv&sample=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("glucoscreen_test_evaluation_sample.csv", response.headers.get("Content-Disposition", ""))

        df = pd.read_csv(io.BytesIO(response.data))
        expected_cols = list(FEATURE_COLUMNS) + ["Diabetes_binary"]
        self.assertEqual(list(df.columns), expected_cols)
        self.assertGreater(len(df), 0)
        self.assertTrue(set(df["Diabetes_binary"].unique()).issubset({0, 1}))

    def test_download_blank_labeled_xlsx_template(self):
        """Test GET /debug/evaluate/template?format=xlsx&sample=0 returns blank Excel template."""
        response = self.client.get("/debug/evaluate/template?format=xlsx&sample=0")
        self.assertEqual(response.status_code, 200)
        self.assertIn("spreadsheetml", response.mimetype)

        df = pd.read_excel(io.BytesIO(response.data))
        expected_cols = list(FEATURE_COLUMNS) + ["Diabetes_binary"]
        self.assertEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 0)

    # ── 3. Evaluation Pipeline & Metrics Tests ─────────────────────────────────

    def test_evaluate_valid_labeled_csv(self):
        """Test POST /debug/evaluate with valid labeled CSV test dataset."""
        csv_bytes, _, _ = generate_labeled_template("csv", sample=True)
        data = {
            "file": (io.BytesIO(csv_bytes), "test_labeled_dataset.csv"),
        }
        response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)

        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("eval_id", json_data)
        self.assertIn("metrics", json_data)
        self.assertIn("rows", json_data)

        metrics = json_data["metrics"]
        self.assertEqual(metrics["total_records"], 6)
        self.assertIn("accuracy_pct", metrics)
        self.assertIn("precision_pct", metrics)
        self.assertIn("recall_pct", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"], 6)

        rows = json_data["rows"]
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertIn("actual_class", r)
            self.assertIn(r["actual_class"], [0, 1])
            self.assertIn("predicted_class", r)
            self.assertIn(r["predicted_class"], [0, 1])
            self.assertIn("probability", r)
            self.assertIn("is_match", r)
            self.assertIn(r["outcome_type"], ["TP", "TN", "FP", "FN"])
            self.assertEqual(r["is_match"], (r["actual_class"] == r["predicted_class"]))

    def test_evaluate_valid_labeled_excel(self):
        """Test POST /debug/evaluate with valid labeled Excel file."""
        xlsx_bytes, _, _ = generate_labeled_template("xlsx", sample=True)
        data = {
            "file": (io.BytesIO(xlsx_bytes), "test_labeled_dataset.xlsx"),
        }
        response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["metrics"]["total_records"], 6)

    def test_target_column_alias_detection(self):
        """Verify that target columns named 'Outcome', 'Target', 'y', or 'actual' are correctly detected."""
        csv_bytes, _, _ = generate_labeled_template("csv", sample=True)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        df = df.rename(columns={"Diabetes_binary": "Outcome"})

        renamed_csv = df.to_csv(index=False).encode("utf-8")
        data = {
            "file": (io.BytesIO(renamed_csv), "test_with_outcome_col.csv"),
        }
        response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["target_column"], "Outcome")

    # ── 4. Validation and Error Handling Tests ─────────────────────────────────

    def test_missing_target_column_error(self):
        """Verify error when uploaded file lacks a ground-truth target column."""
        csv_bytes, _, _ = generate_labeled_template("csv", sample=True)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        df = df.drop(columns=["Diabetes_binary"])  # Remove target

        no_target_csv = df.to_csv(index=False).encode("utf-8")
        data = {
            "file": (io.BytesIO(no_target_csv), "unlabeled_dataset.csv"),
        }
        response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])
        self.assertTrue(any("Missing ground-truth target column" in err for err in json_data["errors"]))

    def test_missing_feature_columns_error(self):
        """Verify error when required feature columns are missing."""
        df = pd.DataFrame({
            "HighBP": [1, 0],
            "BMI": [25.0, 30.0],
            "Diabetes_binary": [0, 1],
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        data = {
            "file": (io.BytesIO(csv_bytes), "incomplete_features.csv"),
        }
        response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])
        self.assertTrue(any("Missing" in err and "feature column" in err for err in json_data["errors"]))

    def test_no_file_uploaded_error(self):
        """Verify error when POST request contains no file payload."""
        response = self.client.post("/debug/evaluate", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["success"])

    # ── 5. Evaluation Results Export Tests ─────────────────────────────────────

    def test_export_evaluation_csv_and_xlsx(self):
        """Test exporting evaluation results to CSV and Excel."""
        csv_bytes, _, _ = generate_labeled_template("csv", sample=True)
        data = {
            "file": (io.BytesIO(csv_bytes), "test_dataset.csv"),
        }
        post_response = self.client.post("/debug/evaluate", data=data, content_type="multipart/form-data")
        eval_id = post_response.get_json()["eval_id"]

        # CSV Export
        csv_export = self.client.get(f"/debug/evaluate/export?eval_id={eval_id}&format=csv")
        self.assertEqual(csv_export.status_code, 200)
        self.assertEqual(csv_export.mimetype, "text/csv")
        self.assertIn("glucoscreen_evaluation_results.csv", csv_export.headers.get("Content-Disposition", ""))

        df_csv = pd.read_csv(io.BytesIO(csv_export.data))
        self.assertIn("Actual Class (y)", df_csv.columns)
        self.assertIn("Predicted Class (y_hat)", df_csv.columns)
        self.assertIn("Match Status", df_csv.columns)
        self.assertIn("Classification Type", df_csv.columns)

        # Excel Export
        xlsx_export = self.client.get(f"/debug/evaluate/export?eval_id={eval_id}&format=xlsx")
        self.assertEqual(xlsx_export.status_code, 200)
        self.assertIn("spreadsheetml", xlsx_export.mimetype)


if __name__ == "__main__":
    unittest.main()
