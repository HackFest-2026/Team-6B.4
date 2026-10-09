# MEDVISION — Hospital Readmission Risk Prediction

MEDVISION is a hospital readmission risk analytics prototype. It trains a readmission classifier on the UCI Diabetes 130-US hospitals dataset and offers separate views for held-out historical encounters and a compact new-encounter form.

## Data

The training inputs are `D:\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv` and `D:\diabetes+130-us+hospitals+for+years+1999-2008\IDS_mapping.csv`. The mapping file decodes admission type and source IDs. The attached feature audit is reflected in the primary-model exclusions and transformations. The separate cleaned v2 CSV is not used for training because it omits `patient_nbr`, which is needed for patient-level splitting.

To use another copy, put it here:

```text
data/diabetic_data.csv
data/IDS_mapping.csv
```

Training defaults to the supplied original `D:` files. To use another copy, pass both paths: `python train.py --data path\to\diabetic_data.csv --mapping path\to\IDS_mapping.csv`.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training writes a serialized full preprocessing/model pipeline, evaluation metrics, held-out encounter predictions, and subgroup fairness summaries under `artifacts/`. The split groups by `patient_nbr`, so a patient's encounters cannot appear in both train and test. The target is `readmitted == '<30'`; both `NO` and `>30` map to 0.

The app uses a green-and-white theme with a subtle animated brand mark and hover-highlighted metric cards. Its Home, Historical records, and New data input pages separate the two prediction use cases. Historical analytics cover only held-out encounters with known outcomes. The risk and medication donut charts are clickable and update the adjacent segment statistics; selecting a high-risk record opens explanation and age, gender, and race fairness cards. The compact new-encounter form scores with the same fitted pipeline and displays an estimate explanation and demographic subgroup comparisons, without presenting an unknown outcome as known. Fields omitted in the form use training-fitted imputation.

## Current scope and next stage

This stage includes deterministic cleaning and ICD-9 chapter features, train-only preprocessing, patient-level development/validation/test splits, five candidate models, validation-based model and threshold selection, standard held-out metrics, and descriptive subgroup metrics. Current held-out performance does not meet a 90% accuracy goal; the dataset is imbalanced, and thresholds that produce high accuracy can severely reduce readmission recall. Next, review calibration and subgroup results, then improve performance through leakage-safe feature/model experiments. This is an educational prototype and is not intended for clinical use.
