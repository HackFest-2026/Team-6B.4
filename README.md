# MEDVISION — Hospital Readmission Risk Prediction

MEDVISION is a hospital readmission risk analytics prototype. It trains a readmission classifier on the UCI Diabetes 130-US hospitals dataset and offers separate views for held-out historical encounters and a compact new-encounter form.

## Data

The repository includes `data/diabetic_data.csv` and `data/IDS_mapping.csv`; training uses these files by default. The mapping file decodes admission type and source IDs. The separate cleaned v2 CSV is not used because it omits `patient_nbr`, which is needed for patient-level splitting.

The included dataset is from Clore, Cios, DeShazo, and Strack, *Diabetes 130-US Hospitals for Years 1999-2008* (UCI Machine Learning Repository, 2014), [DOI: 10.24432/C5230J](https://doi.org/10.24432/C5230J). UCI lists it under the [CC BY 4.0 license](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008). The data includes health and demographic attributes. Generated row-level predictions are kept out of Git.

To train against a different local copy, pass both paths:

```text
data/diabetic_data.csv
data/IDS_mapping.csv
```

`python train.py --data path\to\diabetic_data.csv --mapping path\to\IDS_mapping.csv`

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training writes a serialized full preprocessing/model pipeline, a separate compact-form pipeline, evaluation metrics, held-out encounter predictions, data-quality audit, and subgroup fairness summaries under `artifacts/`. The split groups by `patient_nbr`, so a patient's encounters cannot appear in both train and test. The target is `readmitted == '<30'`; both `NO` and `>30` map to 0. Preprocessing is fit only on development data.

The app uses a green-and-white theme with a subtle animated brand mark and hover-highlighted metric cards. Its Home, Historical records, and New data input pages separate the two prediction use cases. Historical analytics cover only held-out encounters with known outcomes. The risk and medication donut charts are clickable and update the adjacent segment statistics; selecting a high-risk record opens explanation and age, gender, and race fairness cards. The compact new-encounter form scores with a model trained only on the 13 fields the form collects; its fairness cards compare the matching demographic groups against that compact model's own historical holdout, without presenting an unknown outcome as known.

## Current scope and next stage

This stage includes deterministic cleaning and ICD-9 chapter features, train-only preprocessing, patient-level development/validation/test splits, Logistic Regression, Random Forest, and Gradient Boosting candidates, validation selection using average precision, and a recall-weighted F2 decision cutoff. Accuracy is shown alongside precision, recall, F1, average precision, ROC-AUC, and threshold trade-offs because the positive class is only about 11% of encounters; an always-negative model would already score about 89% accuracy while missing every readmission. The holdout results also include age, gender, and race slices. This is an educational prototype and is not intended for clinical use.
