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
python train.py
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To train with CSVs stored elsewhere, pass both file paths:

```powershell
python train.py --data "C:\path\to\diabetic_data.csv" --mapping "C:\path\to\IDS_mapping.csv"
```

## Training workflow

- Target: `<30` is positive (`1`); `NO` and `>30` are negative (`0`).
- Identifiers are excluded from model features. Patients are grouped by `patient_nbr` for a 20% untouched test split. The remaining development data is split again into a training portion and an internal validation portion for candidate selection and threshold selection. Preprocessing is fit on training data only; the chosen model is refit on development data before final test evaluation.
- Candidate models: balanced and unbalanced Logistic Regression, Random Forest, and Gradient Boosting. Selection uses validation average precision, which is useful for this imbalanced target; ROC-AUC is also reported.
- The default decision threshold is selected on validation using F2 to put more weight on recall. An F1-selected threshold is also reported to show the precision/recall trade-off. The test set is not used to choose the model or either threshold.
- Diagnosis fields are transformed into broad ICD-9 chapters. The source dataset uses ICD-9 diagnosis codes. Weight, maximum glucose, constant medication fields, and discharge disposition are excluded from the primary model; discharge disposition is known after the prediction point and could leak outcome-related information.
- A separate compact model is trained on only the 13 fields entered in the new-data form. It does not score a form submission by filling the full model's omitted features with defaults.

## Latest holdout results

These results are from the saved run in `artifacts/metrics.json`. Positive cases make up 11.6% of the test encounters, so an always-negative classifier already has 88.4% accuracy while detecting no readmissions. Accuracy should not be interpreted on its own.

| Model and cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-feature model, recall-focused F2 | 35.7% | 13.4% | 83.4% | 23.1% | 0.635 | 0.210 |
| Full-feature model, F1 alternative | 76.4% | 20.2% | 35.6% | 25.8% | 0.635 | 0.210 |
| Compact form model, recall-focused F2 | 30.6% | 12.9% | 86.9% | 22.4% | 0.633 | 0.209 |
| Compact form model, F1 alternative | 78.5% | 21.3% | 31.8% | 25.5% | 0.633 | 0.209 |

The recall-focused cutoff finds more positive cases but flags many encounters that are not readmitted. The F1 alternative reduces false alerts but misses more positive cases. The current results do not support claiming 90% useful predictive accuracy; the high accuracy of an always-negative baseline comes from class imbalance, not readmission detection.

## Dashboard

- **Home:** holdout model metrics, candidate comparison, and interactive outcome/medication distributions.
- **Historical records:** held-out records with known outcomes, risk estimates, explanations, and age, gender, and race context.
- **New data input:** compact 13-field form scored by its dedicated model. The outcome is unknown; fairness cards show aggregate results for similar demographic groups in the compact model's historical holdout, not an outcome for the submitted encounter.

Training writes model pipelines, metrics, held-out predictions, and fairness summaries under `artifacts/`. Those generated files are ignored by Git. Run training before launching the app in a fresh clone. Row-level holdout predictions contain health and demographic fields, so keep them local.

## Limitations

This is an educational analytics prototype, not a clinical decision system. Holdout discrimination is modest, and the recall-oriented cutoff has low precision. Fairness results are descriptive subgroup comparisons and should be interpreted with group sizes and dataset limitations in mind.
