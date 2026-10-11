# MEDVISION — Hospital Readmission Risk Prediction

MEDVISION is a Streamlit prototype for exploring 30-day readmission risk in the UCI Diabetes 130-US Hospitals dataset. It separates historical encounters with known outcomes from new-encounter estimates, whose outcomes are unknown.

## Dataset

The repository includes `data/diabetic_data.csv` and `data/IDS_mapping.csv`. Training uses these paths by default. The mapping file decodes admission type and admission source. The cleaned v2 CSV is not used because it omits `patient_nbr`, which is required to keep each patient's encounters together during splitting.

The dataset is from Clore, Cios, DeShazo, and Strack, *Diabetes 130-US Hospitals for Years 1999-2008* (UCI Machine Learning Repository, 2014), [DOI: 10.24432/C5230J](https://doi.org/10.24432/C5230J). UCI lists it under the [CC BY 4.0 license](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008). The dataset contains health and demographic attributes; the repository includes the required attribution. Generated encounter-level predictions are not committed.

## Setup and run (Windows PowerShell)

Run these commands from the MEDVISION project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --model-profile previous
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --model-profile previous
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To train with CSVs stored elsewhere, pass both file paths:

```powershell
python train.py --data "C:\path\to\diabetic_data.csv" --mapping "C:\path\to\IDS_mapping.csv"
```

## Training workflow

- Target: `<30` is positive (`1`); `NO` and `>30` are negative (`0`).
- The default `previous` profile restores the earlier balanced Logistic Regression configuration (`class_weight="balanced"`, `C=0.5`, `liblinear`) and its validation-selected F1 cutoff. This is the intended demo model.
- A 20% patient-disjoint test set is kept untouched. The remaining development records are split into training and validation groups by `patient_nbr`; preprocessing is fit on training data only, and the final estimators are refit on development data before test evaluation.
- `--model-profile compare` enables the later candidate comparison (balanced/unbalanced Logistic Regression, Random Forest, and Gradient Boosting) with validation average-precision selection and recall-focused F2 thresholding.
- Diagnosis fields are transformed into broad ICD-9 chapters. Weight, maximum glucose, constant medication fields, and discharge disposition are excluded from the primary model; discharge disposition is known after the prediction point and could leak outcome-related information.
- A separate balanced Logistic Regression is trained on only the 13 fields in the new-data form. The form model does not fill omitted inputs with made-up defaults.
- Training regenerates `artifacts/medvision_pipeline.joblib` and the metrics/prediction files. Because artifacts are ignored by Git, run this command after cloning or changing the model profile:

```powershell
python train.py --model-profile previous
```

The previously reported full-model reference results for this demo version were accuracy 68.9%, precision 18.0%, recall 47.4%, F1 26.1%, and ROC-AUC 0.643 at a cutoff around 0.52. The regenerated scores are written to `artifacts/metrics.json`; use those values if they differ. The positive class is only about 11%, so an always-negative classifier can have high accuracy while detecting no readmissions. Accuracy alone is not a useful success claim.

## Dashboard

- **Home:** holdout model metrics, active-model validation snapshot, and interactive outcome/medication distributions.
- **Historical records:** held-out records with known outcomes, risk estimates, explanations, and age, gender, and race context.
- **New data input:** compact 13-field form scored by its dedicated model. The outcome is unknown; fairness cards show aggregate results for similar demographic groups in the compact model's historical holdout, not an outcome for the submitted encounter.

Training writes model pipelines, metrics, held-out predictions, and fairness summaries under `artifacts/`. Those generated files are ignored by Git. Run training before launching the app in a fresh clone. Row-level holdout predictions contain health and demographic fields, so keep them local.

## Limitations

This is an educational analytics prototype, not a clinical decision system. Holdout discrimination is modest and precision is limited. Fairness results are descriptive subgroup comparisons and should be interpreted with group sizes and dataset limitations in mind.
