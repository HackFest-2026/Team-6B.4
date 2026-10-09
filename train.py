from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from medvision.pipeline import build_preprocessor, make_xy, read_mapping


# These are precisely the fields collected by the compact new-encounter form.
# A separate estimator is trained on them so omitted form fields are not
# silently treated as if the user supplied them.
COMPACT_FEATURES = [
    "age", "gender", "race", "admission_type_id", "admission_source_id",
    "time_in_hospital", "number_inpatient", "number_emergency",
    "number_outpatient", "number_diagnoses", "num_medications",
    "a1cresult", "insulin",
]


def score_metrics(y_true, y_pred, probability):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probability = np.asarray(probability)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probability)) if has_both_classes else None,
        "average_precision": float(average_precision_score(y_true, probability)) if np.any(y_true == 1) else None,
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def fairness_report(test: pd.DataFrame, y_true, y_pred, probability) -> pd.DataFrame:
    rows = []
    for attr in ("age", "gender", "race"):
        if attr not in test:
            continue
        series = test[attr].fillna("Unknown").astype(str).reset_index(drop=True)
        for group in sorted(series.unique()):
            ix = np.flatnonzero(series.to_numpy() == group)
            row = {"attribute": attr, "group": group, "n": int(len(ix))}
            row.update(score_metrics(np.asarray(y_true)[ix], np.asarray(y_pred)[ix], np.asarray(probability)[ix]))
            row["predicted_positive_rate"] = float(np.mean(np.asarray(y_pred)[ix]))
            row["mean_predicted_risk"] = float(np.mean(np.asarray(probability)[ix]))
            rows.append(row)
    return pd.DataFrame(rows)


def _choose_threshold(y_true, probabilities, beta: float):
    """Choose an F-beta cutoff on validation data; beta > 1 weights recall."""
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), np.asarray(probabilities)])
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        pred = (probabilities >= threshold).astype("int8")
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        beta_sq = beta * beta
        score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) if (precision + recall) else 0.0
        candidate = (float(score), -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return best[2]


def _baseline(X: pd.DataFrame, indices) -> dict:
    result = {}
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            center = X.iloc[indices][column].median()
            result[column] = float(center) if pd.notna(center) else 0.0
        else:
            modes = X.iloc[indices][column].dropna().astype(str).mode()
            result[column] = str(modes.iloc[0]) if not modes.empty else "Unknown"
    return result


def _fit_candidates(X, y, fit_idx, val_idx):
    estimators = {
        "Logistic Regression (balanced)": LogisticRegression(max_iter=1800, class_weight="balanced", solver="liblinear", C=0.5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1800, solver="liblinear", C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=70, learning_rate=0.05, max_depth=2, min_samples_leaf=30, subsample=0.8, random_state=42),
    }
    models, reports = {}, {}
    for name, estimator in estimators.items():
        print(f"Fitting {name} on {len(fit_idx):,} development rows...", flush=True)
        candidate = Pipeline([("preprocess", build_preprocessor(X.iloc[fit_idx])), ("model", estimator)])
        candidate.fit(X.iloc[fit_idx], y[fit_idx])
        probability = candidate.predict_proba(X.iloc[val_idx])[:, 1]
        reports[name] = score_metrics(y[val_idx], (probability >= 0.5).astype("int8"), probability)
        models[name] = candidate
    # Average precision is more informative than ROC-AUC with the roughly
    # 11% positive class. AUC remains available for comparison.
    winner = max(reports, key=lambda name: (reports[name]["average_precision"], reports[name]["auc"] or -1))
    return estimators, models, reports, winner


def train(data_path: Path, out: Path, mapping_path: Path | None = None, test_size: float = 0.2):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    raw = pd.read_csv(data_path, low_memory=False)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    target_column = "readmitted_30d" if "readmitted_30d" in raw else "readmitted"
    mappings = read_mapping(mapping_path)
    X, y, groups = make_xy(raw, mappings)

    # Patient-disjoint test set and patient-disjoint inner validation set.
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    dev_idx, test_idx = next(outer.split(X, y, groups))
    if np.intersect1d(groups[dev_idx], groups[test_idx]).size:
        raise AssertionError("Patient-level split leaked patient_nbr across development and test.")
    inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=43)
    fit_rel, val_rel = next(inner.split(X.iloc[dev_idx], y[dev_idx], groups[dev_idx]))
    fit_idx, val_idx = dev_idx[fit_rel], dev_idx[val_rel]
    if np.intersect1d(groups[fit_idx], groups[val_idx]).size or np.intersect1d(groups[val_idx], groups[test_idx]).size:
        raise AssertionError("Patient-level validation split contains overlapping patients.")

    full_estimators, full_models, full_validation, full_winner = _fit_candidates(X, y, fit_idx, val_idx)
    full_validation_probability = full_models[full_winner].predict_proba(X.iloc[val_idx])[:, 1]
    full_threshold_f1 = _choose_threshold(y[val_idx], full_validation_probability, beta=1.0)
    full_threshold_f2 = _choose_threshold(y[val_idx], full_validation_probability, beta=2.0)

    compact_columns = [column for column in COMPACT_FEATURES if column in X.columns]
    if len(compact_columns) != len(COMPACT_FEATURES):
        missing = sorted(set(COMPACT_FEATURES) - set(compact_columns))
        raise ValueError(f"Compact-form training fields missing from source dataset: {missing}")
    compact_X = X[compact_columns]
    compact_estimators, compact_models, compact_validation, compact_winner = _fit_candidates(compact_X, y, fit_idx, val_idx)
    compact_validation_probability = compact_models[compact_winner].predict_proba(compact_X.iloc[val_idx])[:, 1]
    compact_threshold_f1 = _choose_threshold(y[val_idx], compact_validation_probability, beta=1.0)
    compact_threshold_f2 = _choose_threshold(y[val_idx], compact_validation_probability, beta=2.0)

    # Refit only the selected model for each use case on development data.
    full_pipeline = Pipeline([("preprocess", build_preprocessor(X.iloc[dev_idx])), ("model", clone(full_estimators[full_winner]))])
    full_pipeline.fit(X.iloc[dev_idx], y[dev_idx])
    compact_pipeline = Pipeline([("preprocess", build_preprocessor(compact_X.iloc[dev_idx])), ("model", clone(compact_estimators[compact_winner]))])
    compact_pipeline.fit(compact_X.iloc[dev_idx], y[dev_idx])

    full_probability = full_pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    full_prediction = (full_probability >= full_threshold_f2).astype("int8")
    full_prediction_f1 = (full_probability >= full_threshold_f1).astype("int8")
    compact_probability = compact_pipeline.predict_proba(compact_X.iloc[test_idx])[:, 1]
    compact_prediction = (compact_probability >= compact_threshold_f2).astype("int8")
    compact_prediction_f1 = (compact_probability >= compact_threshold_f1).astype("int8")
    full_test_metrics = score_metrics(y[test_idx], full_prediction, full_probability)
    full_test_metrics_f1 = score_metrics(y[test_idx], full_prediction_f1, full_probability)
    compact_test_metrics = score_metrics(y[test_idx], compact_prediction, compact_probability)
    compact_test_metrics_f1 = score_metrics(y[test_idx], compact_prediction_f1, compact_probability)
    full_test_metrics_at_half = {}
    for name, candidate in full_models.items():
        probability = candidate.predict_proba(X.iloc[test_idx])[:, 1]
        full_test_metrics_at_half[name] = score_metrics(y[test_idx], (probability >= 0.5).astype("int8"), probability)

    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": full_pipeline,
        "model_name": full_winner,
        "feature_columns": X.columns.tolist(),
        "threshold": full_threshold_f2,
        "threshold_f1": full_threshold_f1,
        "mappings": mappings,
        "explanation_baseline": _baseline(X, dev_idx),
        "compact_pipeline": compact_pipeline,
        "compact_model_name": compact_winner,
        "compact_feature_columns": compact_columns,
        "compact_threshold": compact_threshold_f2,
        "compact_threshold_f1": compact_threshold_f1,
        "compact_explanation_baseline": _baseline(compact_X, dev_idx),
    }
    joblib.dump(bundle, out / "medvision_pipeline.joblib")

    test_prevalence = float(np.mean(y[test_idx]))
    summary = {
        "selected_model": full_winner,
        "target": "readmitted (<30=1; NO and >30=0)",
        "split": "Patient-disjoint development/validation/test via GroupShuffleSplit",
        "model_selection": "Validation average precision; ROC-AUC reported too",
        "threshold_selection": "Validation F2 (recall-weighted); F1 alternative also reported; test set not used for tuning",
        "decision_threshold": float(full_threshold_f2),
        "decision_threshold_f1_alternative": float(full_threshold_f1),
        "development_rows": int(len(dev_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "development_patients": int(pd.Series(groups[dev_idx]).nunique()),
        "validation_patients": int(pd.Series(groups[val_idx]).nunique()),
        "test_patients": int(pd.Series(groups[test_idx]).nunique()),
        "test_positive_rate": test_prevalence,
        "test_majority_baseline_accuracy": float(max(test_prevalence, 1 - test_prevalence)),
        "validation_metrics_at_0_5": full_validation,
        "test_metrics_at_0_5": full_test_metrics_at_half,
        "selected_test_metrics": full_test_metrics,
        "selected_test_metrics_at_f1_threshold": full_test_metrics_f1,
        "compact_form_model": {
            "selected_model": compact_winner,
            "features": compact_columns,
            "decision_threshold": float(compact_threshold_f2),
            "decision_threshold_f1_alternative": float(compact_threshold_f1),
            "validation_metrics_at_0_5": compact_validation,
            "selected_test_metrics": compact_test_metrics,
            "selected_test_metrics_at_f1_threshold": compact_test_metrics_f1,
        },
        "data_audit": {
            "rows": int(len(raw)),
            "columns": int(len(raw.columns)),
            "unique_patients": int(raw["patient_nbr"].nunique()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "duplicate_encounter_ids": int(raw["encounter_id"].duplicated().sum()) if "encounter_id" in raw else None,
            "target_counts": {str(key): int(value) for key, value in raw[target_column].value_counts(dropna=False).items()},
            "target_positive_rate": float(np.mean(y)),
            "source_missing_values_top10": {str(key): int(value) for key, value in raw.isna().sum().sort_values(ascending=False).head(10).items()},
            "effective_missing_values_top10": {
                str(key): int(value)
                for key, value in raw.apply(
                    lambda column: column.isna() | column.astype("string").str.strip().str.lower().isin({"?", "unknown/invalid", "none", "nan", ""})
                ).sum().sort_values(ascending=False).head(10).items()
            },
            "documented_exclusions": ["weight (mostly missing)", "max_glu_serum (mostly unmeasured)", "examide/citoglipton (constant)", "discharge_disposition_id (post-discharge leakage risk)"],
            "diagnosis_encoding": "ICD-9 chapters; source dataset diagnosis codes are ICD-9, not ICD-10",
        },
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    source = raw.copy()
    source.columns = [str(column).strip().lower() for column in source.columns]
    target = "readmitted_30d" if "readmitted_30d" in source else "readmitted"
    valid = pd.to_numeric(source[target], errors="coerce").isin([0, 1]) if target == "readmitted_30d" else source[target].astype("string").str.strip().isin(["<30", ">30", "NO"])
    source = source.loc[valid].reset_index(drop=True)
    hidden = {"encounter_id", "patient_nbr", target}
    display_columns = [column for column in source.columns if column not in hidden]
    hist = source.iloc[test_idx][display_columns].copy()
    hist["actual_readmitted_within_30_days"] = y[test_idx]
    hist["risk_probability"] = full_probability
    hist["predicted_readmitted_within_30_days"] = full_prediction
    hist["risk_category"] = pd.cut(full_probability, [0.0, 0.3, 0.7, 1.000001], labels=["Lower", "Moderate", "Higher"], right=False, include_lowest=True).astype(str)
    hist.to_csv(out / "historical_predictions.csv", index=False)
    fairness_report(hist, y[test_idx], full_prediction, full_probability).to_csv(out / "fairness_metrics.csv", index=False)
    compact_hist = hist[[column for column in ("age", "gender", "race") if column in hist]].copy()
    compact_hist["actual_readmitted_within_30_days"] = y[test_idx]
    compact_hist["risk_probability"] = compact_probability
    compact_hist["predicted_readmitted_within_30_days"] = compact_prediction
    compact_hist["risk_category"] = pd.cut(compact_probability, [0.0, 0.3, 0.7, 1.000001], labels=["Lower", "Moderate", "Higher"], right=False, include_lowest=True).astype(str)
    compact_hist.to_csv(out / "compact_historical_predictions.csv", index=False)
    fairness_report(compact_hist, y[test_idx], compact_prediction, compact_probability).to_csv(out / "compact_fairness_metrics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=project_root / "data" / "diabetic_data.csv")
    parser.add_argument("--mapping", type=Path, default=project_root / "data" / "IDS_mapping.csv")
    parser.add_argument("--out", type=Path, default=project_root / "artifacts")
    args = parser.parse_args()
    try:
        print(json.dumps(train(args.data, args.out, args.mapping if args.mapping.exists() else None), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
