from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMNS = ["encounter_id", "patient_nbr"]
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]


def read_mapping(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Read sectioned UCI IDS_mapping.csv into feature-to-code maps."""
    result: dict[str, dict[str, str]] = {}
    if path is None or not Path(path).exists():
        return result
    current = None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            values = [value.strip() for value in row]
            if not any(values):
                continue
            key = values[0].lower()
            if key in {"admission_type_id", "admission_source_id", "discharge_disposition_id"}:
                current = key
                result.setdefault(current, {})
                continue
            if current and len(values) >= 2 and values[0].isdigit():
                result[current][values[0]] = values[1]
    return result


def icd9_chapter(value: Any) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "?", "unknown", "none", "nan"}:
        return "Unknown"
    code = str(value).strip().upper()
    if code.startswith("V"):
        return "V_codes"
    if code.startswith("E"):
        return "E_codes"
    match = re.match(r"^(\d{3})(?:\.\d+)?$", code)
    if not match:
        return "Unknown"
    number = int(match.group(1))
    chapters = [(1,139,"Infectious"),(140,239,"Neoplasms"),(240,279,"Endocrine_metabolic"),(280,289,"Blood"),(290,319,"Mental"),(320,389,"Nervous_sense_organs"),(390,459,"Circulatory"),(460,519,"Respiratory"),(520,579,"Digestive"),(580,629,"Genitourinary"),(630,679,"Pregnancy"),(680,709,"Skin"),(710,739,"Musculoskeletal"),(740,759,"Congenital"),(760,779,"Perinatal"),(780,799,"Symptoms_signs"),(800,999,"Injury_poisoning")]
    return next((label for low, high, label in chapters if low <= number <= high), "Unknown")


def clean_and_engineer(frame: pd.DataFrame, mappings: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    for col in data.select_dtypes(include=["object", "string"]).columns:
        s = data[col].astype("string").str.strip()
        data[col] = s.mask(s.str.lower().isin({"?", "unknown/invalid", "none", "nan", ""}), "Unknown").fillna("Unknown").astype(object)
    for col, mapping in (mappings or {}).items():
        if col in data:
            data[col] = data[col].map(lambda value: mapping.get(str(value), str(value)) if pd.notna(value) else "Unknown")
    for col in DIAGNOSIS_COLUMNS:
        if col in data:
            data[col] = data[col].map(icd9_chapter)
    if "age" in data:
        data["age"] = data["age"].astype("string").str.replace(" ", "", regex=False).fillna("Unknown").astype(object)
    # Apply the supplied audit's explicit primary-model exclusions.
    data.drop(columns=[c for c in ("weight", "max_glu_serum", "examide", "citoglipton", "discharge_disposition_id") if c in data], inplace=True)
    return data


def prepare_features(frame: pd.DataFrame, feature_columns: list[str] | None = None, mappings: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    data.drop(columns=[c for c in ID_COLUMNS if c in data], errors="ignore", inplace=True)
    data = clean_and_engineer(data, mappings)
    if feature_columns is not None:
        for col in feature_columns:
            if col not in data:
                data[col] = np.nan
        data = data.reindex(columns=feature_columns)
    return data


def make_xy(frame: pd.DataFrame, mappings: dict[str, dict[str, str]] | None = None):
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    if "readmitted_30d" in data:
        y_raw = data.pop("readmitted_30d")
        y = pd.to_numeric(y_raw, errors="coerce")
    elif "readmitted" in data:
        label = data.pop("readmitted").astype("string").str.strip()
        y = label.map({"<30": 1, ">30": 0, "NO": 0})
    else:
        raise ValueError("Expected 'readmitted_30d' or raw 'readmitted' target column.")
    keep = y.isin([0, 1])
    data, y = data.loc[keep].copy(), y.loc[keep].astype("int8")
    if "patient_nbr" not in data:
        raise ValueError("Patient-level splitting requires patient_nbr. The supplied cleaned CSV omits it; provide a version retaining patient_nbr for splitting only.")
    groups = data["patient_nbr"].astype("string").to_numpy()
    X = prepare_features(data, mappings=mappings)
    return X.reset_index(drop=True), y.to_numpy(), groups


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)], remainder="drop", verbose_feature_names_out=False)
