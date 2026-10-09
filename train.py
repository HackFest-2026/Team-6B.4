from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from medvision.pipeline import build_preprocessor, make_xy, read_mapping


def score_metrics(y_true, y_pred, probability):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
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


def _choose_f1_threshold(y_true, probabilities):
    candidates = np.linspace(0.02, 0.98, 97)
    scored = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype("int8")
        result = score_metrics(y_true, pred, probabilities)
        scored.append((result["f1"], -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[2]


def train(data_path: Path, out: Path, mapping_path: Path | None = None, test_size: float = 0.2):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    raw = pd.read_csv(data_path, low_memory=False)
    mappings = read_mapping(mapping_path)
    X, y, groups = make_xy(raw, mappings)

    # Reserve a patient-disjoint test set. Model selection and threshold tuning
    # use only a second patient-disjoint split inside the development data.
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    dev_idx, test_idx = next(outer.split(X, y, groups))
    if np.intersect1d(groups[dev_idx], groups[test_idx]).size:
        raise AssertionError("Patient-level split leaked patient_nbr across development and test.")
    inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=43)
    fit_rel, val_rel = next(inner.split(X.iloc[dev_idx], y[dev_idx], groups[dev_idx]))
    fit_idx, val_idx = dev_idx[fit_rel], dev_idx[val_rel]
    if np.intersect1d(groups[fit_idx], groups[val_idx]).size or np.intersect1d(groups[val_idx], groups[test_idx]).size:
        raise AssertionError("Patient-level validation split contains overlapping patients.")

    estimators = {
        "Logistic Regression (balanced)": LogisticRegression(max_iter=1800, class_weight="balanced", solver="liblinear", C=0.5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1800, solver="liblinear", C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=260, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=2, min_samples_leaf=20, random_state=42),
    }
    validation_scores, validation_models = {}, {}
    for name, estimator in estimators.items():
        candidate = Pipeline([("preprocess", build_preprocessor(X.iloc[fit_idx])), ("model", estimator)])
        candidate.fit(X.iloc[fit_idx], y[fit_idx])
        probability = candidate.predict_proba(X.iloc[val_idx])[:, 1]
        validation_scores[name] = score_metrics(y[val_idx], (probability >= 0.5).astype("int8"), probability)
        validation_models[name] = candidate
    winner = max(validation_scores, key=lambda name: validation_scores[name]["auc"] or -1)

    # Select the decision threshold on validation data only, then measure once
    # on the untouched outer test set.
    validation_probability = validation_models[winner].predict_proba(X.iloc[val_idx])[:, 1]
    threshold = _choose_f1_threshold(y[val_idx], validation_probability)
    estimator = clone(estimators[winner])
    final_pipeline = Pipeline([("preprocess", build_preprocessor(X.iloc[dev_idx])), ("model", estimator)])
    final_pipeline.fit(X.iloc[dev_idx], y[dev_idx])
    test_probability = final_pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    test_prediction = (test_probability >= threshold).astype("int8")
    test_metrics = score_metrics(y[test_idx], test_prediction, test_probability)
    test_metrics_at_half = {}
    for name, candidate in validation_models.items():
        # These comparison scores use the same untouched test patients; they are
        # descriptive only and do not affect model selection.
        probability = candidate.predict_proba(X.iloc[test_idx])[:, 1]
        test_metrics_at_half[name] = score_metrics(y[test_idx], (probability >= 0.5).astype("int8"), probability)

    baseline = {}
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            center = X.iloc[dev_idx][column].median()
            baseline[column] = float(center) if pd.notna(center) else 0.0
        else:
            modes = X.iloc[dev_idx][column].dropna().astype(str).mode()
            baseline[column] = str(modes.iloc[0]) if not modes.empty else "Unknown"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipeline, "model_name": winner, "feature_columns": X.columns.tolist(), "threshold": threshold, "mappings": mappings, "explanation_baseline": baseline}, out / "medvision_pipeline.joblib")
    test_prevalence = float(np.mean(y[test_idx]))
    summary = {
        "selected_model": winner,
        "target": "readmitted (<30=1; NO and >30=0)",
        "split": "Patient-disjoint development/validation/test via GroupShuffleSplit",
        "threshold_selection": "Validation F1; test set not used for tuning",
        "decision_threshold": float(threshold),
        "development_rows": int(len(dev_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "development_patients": int(pd.Series(groups[dev_idx]).nunique()),
        "validation_patients": int(pd.Series(groups[val_idx]).nunique()),
        "test_patients": int(pd.Series(groups[test_idx]).nunique()),
        "test_positive_rate": test_prevalence,
        "test_majority_baseline_accuracy": float(max(test_prevalence, 1 - test_prevalence)),
        "validation_metrics_at_0_5": validation_scores,
        "test_metrics_at_0_5": test_metrics_at_half,
        "selected_test_metrics": test_metrics,
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
    hist["risk_probability"] = test_probability
    hist["predicted_readmitted_within_30_days"] = test_prediction
    hist["risk_category"] = pd.cut(test_probability, [0.0, 0.3, 0.7, 1.000001], labels=["Lower", "Moderate", "Higher"], right=False, include_lowest=True).astype(str)
    hist.to_csv(out / "historical_predictions.csv", index=False)
    fairness_report(hist, y[test_idx], test_prediction, test_probability).to_csv(out / "fairness_metrics.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(r"D:\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"))
    parser.add_argument("--mapping", type=Path, default=Path(r"D:\diabetes+130-us+hospitals+for+years+1999-2008\IDS_mapping.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        print(json.dumps(train(args.data, args.out, args.mapping if args.mapping.exists() else None), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
