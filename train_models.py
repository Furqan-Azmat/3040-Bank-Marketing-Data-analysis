from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(data_paths: List[Path]) -> pd.DataFrame:
    """Load dataset from the first existing path; drop duration column if present."""
    for path in data_paths:
        if path.exists():
            df = pd.read_csv(path, sep=';')
            if 'duration' in df.columns:
                df = df.drop(columns=['duration'])
            return df
    raise FileNotFoundError(f'Could not find dataset in any of: {data_paths}')


def build_metadata(df_subset: pd.DataFrame, features: List[str]) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, list]]:
    """Compute defaults (median/mode), types, and allowed options for the given feature list."""
    cat_cols = df_subset.select_dtypes(include=['object']).columns.tolist()
    feature_defaults: Dict[str, object] = {}
    feature_types: Dict[str, str] = {}
    feature_options: Dict[str, list] = {}
    for col in features:
        if col in cat_cols:
            feature_defaults[col] = df_subset[col].mode(dropna=True)[0]
            feature_types[col] = 'categorical'
            feature_options[col] = sorted(df_subset[col].dropna().unique().tolist())
        else:
            feature_defaults[col] = float(df_subset[col].median())
            feature_types[col] = 'numeric'
            feature_options[col] = []
    return feature_defaults, feature_types, feature_options


def train_call_model(df: pd.DataFrame) -> Tuple[LogisticRegression, Dict[str, int]]:
    target = (df['y'] == 'yes').astype(int)
    X = df[['campaign']]
    X_train, _, y_train, _ = train_test_split(
        X, target, test_size=0.2, stratify=target, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    meta = {
        'calls_default': int(X['campaign'].median()),
        'calls_max': int(X['campaign'].max()),
    }
    return model, meta


def train_top7_model(df: pd.DataFrame, top_features: List[str]) -> Tuple[Pipeline, Dict[str, object], Dict[str, str], Dict[str, list]]:
    y = (df['y'] == 'yes').astype(int)
    X_full = df.drop(columns=['y'])
    cat = X_full.select_dtypes(include=['object']).columns.tolist()
    use_cat = [c for c in top_features if c in cat]
    use_num = [c for c in top_features if c not in use_cat]

    preproc = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore'), use_cat),
            ('num', Pipeline([('scaler', StandardScaler())]), use_num),
        ],
        remainder='drop',
    )

    X_train, _, y_train, _ = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline(
        [
            ('preprocess', preproc),
            ('model', LogisticRegression(max_iter=300, class_weight='balanced')),
        ]
    )
    model.fit(X_train[top_features], y_train)

    defaults, types, options = build_metadata(X_full, top_features)
    return model, defaults, types, options


def train_q2_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, object], Dict[str, str], Dict[str, list], List[str]]:
    y = (df['y'] == 'yes').astype(int)
    X_full = df.drop(columns=['y'])
    feature_list = list(X_full.columns)

    cat_cols = X_full.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X_full.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('numeric', 'passthrough', num_cols),
        ],
        remainder='drop',
    )

    X_train, _, y_train, _ = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

    clf = Pipeline(
        steps=[
            ('preprocess', preprocess),
            ('model', LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1)),
        ]
    )
    clf.fit(X_train, y_train)

    defaults, types, options = build_metadata(X_full, feature_list)
    return clf, defaults, types, options, feature_list


def main() -> None:
    data_paths = [
        Path('Data/bank-additional-full.csv'),
        Path('bank-additional-full.csv'),
        Path('ITEC3040-Final-Project/Data/bank-additional-full.csv'),
    ]
    top_features = [
        'euribor3m',
        'age',
        'campaign',
        'nr.employed',
        'pdays',
        'emp.var.rate',
        'cons.conf.idx',
    ]

    df = load_dataset(data_paths)

    call_model, call_meta = train_call_model(df)
    top7_model, top7_defaults, top7_types, top7_options = train_top7_model(df, top_features)
    q2_model, q2_defaults, q2_types, q2_options, q2_features = train_q2_model(df)

    joblib.dump(
        {
            'model': call_model,
            'defaults': call_meta,
        },
        'call_model.joblib',
    )

    joblib.dump(
        {
            'model': top7_model,
            'feature_list': top_features,
            'defaults': top7_defaults,
            'types': top7_types,
            'options': top7_options,
        },
        'top7_model.joblib',
    )

    joblib.dump(
        {
            'model': q2_model,
            'feature_list': q2_features,
            'defaults': q2_defaults,
            'types': q2_types,
            'options': q2_options,
        },
        'q2_model.joblib',
    )

    print('Saved: call_model.joblib, top7_model.joblib, q2_model.joblib')


if __name__ == '__main__':
    main()
