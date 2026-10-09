import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from medvision.pipeline import clean_and_engineer, icd9_chapter, make_xy, prepare_features, read_mapping
from train import train


class PipelineTests(unittest.TestCase):
    def test_icd9_chapter_mapping(self):
        self.assertEqual(icd9_chapter("250.83"), "Endocrine_metabolic")
        self.assertEqual(icd9_chapter("V45"), "V_codes")
        self.assertEqual(icd9_chapter("E812"), "E_codes")
        self.assertEqual(icd9_chapter("Unknown"), "Unknown")

    def test_cleaning_keeps_unknown_and_drops_audit_exclusions(self):
        row = pd.DataFrame([{"race": "?", "diag_1": "250.83", "weight": "?", "discharge_disposition_id": 1, "num_medications": 4}])
        clean = clean_and_engineer(row)
        self.assertEqual(clean.loc[0, "race"], "Unknown")
        self.assertEqual(clean.loc[0, "diag_1"], "Endocrine_metabolic")
        self.assertNotIn("weight", clean.columns)
        self.assertNotIn("discharge_disposition_id", clean.columns)

    def test_target_encoding_and_group_split_key(self):
        data = pd.DataFrame({"patient_nbr": [10, 10, 20], "readmitted": ["<30", ">30", "NO"], "num_medications": [1, 2, 3]})
        X, y, groups = make_xy(data)
        self.assertEqual(y.tolist(), [1, 0, 0])
        self.assertEqual(groups.tolist(), ["10", "10", "20"])
        self.assertNotIn("patient_nbr", X.columns)

    def test_sectioned_uci_mapping_decodes_admission_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "IDS_mapping.csv"
            path.write_text("admission_type_id,description\n1,Emergency\n2,Urgent\n,\nadmission_source_id,description\n1,Physician Referral\n", encoding="utf-8")
            mappings = read_mapping(path)
            self.assertEqual(mappings["admission_type_id"]["1"], "Emergency")
            self.assertEqual(mappings["admission_source_id"]["1"], "Physician Referral")

    def test_rejects_patient_split_when_identifier_missing(self):
        data = pd.DataFrame({"readmitted_30d": [1, 0], "race": ["Caucasian", "Asian"]})
        with self.assertRaisesRegex(ValueError, "requires patient_nbr"):
            make_xy(data)

    def test_three_model_training_saves_pipeline_and_scores_new_record(self):
        rows = []
        for patient in range(60):
            for encounter in range(4):
                inpatient = (patient + encounter) % 4
                row = {
                    "patient_nbr": patient,
                    "readmitted_30d": int((patient + encounter) % 5 == 0 or inpatient >= 3),
                    "race": ["Caucasian", "AfricanAmerican", "Asian"][patient % 3],
                    "gender": "Female" if patient % 2 else "Male",
                    "age": ["[40-50)", "[60-70)", "[70-80)"][patient % 3],
                    "admission_type": ["Emergency", "Elective"][encounter % 2],
                    "admission_source": "Emergency Room",
                    "time_in_hospital": 1 + (patient + encounter) % 9,
                    "number_inpatient": inpatient,
                    "number_emergency": (patient + encounter) % 3,
                    "number_outpatient": patient % 3,
                    "number_diagnoses": 2 + patient % 8,
                    "num_medications": 3 + encounter + patient % 10,
                    "diag_1": "250.83" if patient % 2 else "414.01",
                    "A1Cresult": "Unknown" if patient % 4 else ">7",
                    "insulin": ["No", "Steady", "Up"][encounter % 3],
                }
                rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            result = train(source, root / "artifacts")
            self.assertIn(result["selected_model"], {"Logistic Regression (balanced)", "Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"})
            self.assertIn("selected_test_metrics", result)
            self.assertTrue((root / "artifacts" / "medvision_pipeline.joblib").exists())
            self.assertTrue((root / "artifacts" / "historical_predictions.csv").exists())
            self.assertTrue((root / "artifacts" / "fairness_metrics.csv").exists())
            import joblib
            bundle = joblib.load(root / "artifacts" / "medvision_pipeline.joblib")
            new_row = {column: np.nan for column in bundle["feature_columns"]}
            new_row.update({"age": "[60-70)", "gender": "Female", "race": "Caucasian", "time_in_hospital": 4, "number_inpatient": 1, "number_emergency": 0, "number_outpatient": 0, "number_diagnoses": 6, "num_medications": 12, "diag_1": "250.83"})
            features = prepare_features(pd.DataFrame([new_row]), bundle["feature_columns"])
            risk = bundle["pipeline"].predict_proba(features)[0, 1]
            self.assertTrue(0 <= risk <= 1)


if __name__ == "__main__":
    unittest.main()
