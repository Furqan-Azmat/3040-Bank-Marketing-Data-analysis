import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, Tuple, List

# Run the line below in the terminal to launch the GUI: 
# py(or python) combined_ui_prediction_gui.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths to search for the dataset
DATA_PATHS = [
    Path('Data/bank-additional-full.csv'),
    Path('bank-additional-full.csv'),
    Path('ITEC3040-Final-Project/Data/bank-additional-full.csv'),
]

# Top features identified in term_deposit_feature_combo (best-k selection)
TOP_FEATURES = [
    'emp.var.rate',
    'cons.price.idx',
    'month',
    'contact',
    'euribor3m',
    'nr.employed',
    'pdays',
    'day_of_week',
    'poutcome',
    'previous',
    'job',
    'cons.conf.idx',
]

CALL_THRESHOLD = 0.50
FULL_THRESHOLD = 0.20  # matches the interactive threshold used in the notebook
Q2_THRESHOLD = 0.50    # default logistic cutoff from Question 2
# Weighted blend (call-count is weakest signal, term_deposit is strongest)
CALL_WEIGHT = 0.15
FULL_WEIGHT = 0.50
Q2_WEIGHT = 0.35


def locate_data_file() -> Path:
    for path in DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError('bank-additional-full.csv not found in expected locations.')


def load_dataset() -> pd.DataFrame:
    path = locate_data_file()
    df = pd.read_csv(path, sep=';')
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
    return df


def train_call_model(df: pd.DataFrame) -> Tuple[LogisticRegression, int, int]:
    target = (df['y'] == 'yes').astype(int)
    X = df[['campaign']]
    X_train, _, y_train, _ = train_test_split(
        X, target, test_size=0.2, stratify=target, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    default_calls = int(X['campaign'].median())
    max_calls = int(X['campaign'].max())
    return model, default_calls, max_calls


def build_feature_metadata(df: pd.DataFrame, feature_list: List[str]) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, list], List[str], List[str]]:
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    use_cat = [c for c in feature_list if c in cat_cols]
    use_num = [c for c in feature_list if c in num_cols]

    feature_defaults: Dict[str, object] = {}
    feature_types: Dict[str, str] = {}
    feature_options: Dict[str, list] = {}
    for col in feature_list:
        if col in use_cat:
            feature_defaults[col] = df[col].mode(dropna=True)[0]
            feature_types[col] = 'categorical'
            feature_options[col] = sorted(df[col].dropna().unique().tolist())
        else:
            feature_defaults[col] = float(df[col].median())
            feature_types[col] = 'numeric'
            feature_options[col] = []

    return feature_defaults, feature_types, feature_options, use_cat, use_num


def train_feature_model(df: pd.DataFrame, feature_list: List[str]) -> Tuple[Pipeline, Dict[str, object], Dict[str, str], Dict[str, list]]:
    y = (df['y'] == 'yes').astype(int)
    X = df.drop(columns=['y'])

    feature_defaults, feature_types, feature_options, use_cat, use_num = build_feature_metadata(X, feature_list)

    preprocessor = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore'), use_cat),
            ('num', Pipeline([('scaler', StandardScaler())]), use_num),
        ],
        remainder='drop',
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        [
            ('preprocess', preprocessor),
            ('model', LogisticRegression(max_iter=300, class_weight='balanced')),
        ]
    )
    model.fit(X_train[feature_list], y_train)

    return model, feature_defaults, feature_types, feature_options


def train_q2_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, object], Dict[str, str], Dict[str, list], List[str]]:
    """Full-feature logistic (Question 2 best model): all columns except y (duration already dropped)."""
    y = (df['y'] == 'yes').astype(int)
    X = df.drop(columns=['y'])
    feature_list = list(X.columns)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    feature_defaults, feature_types, feature_options, _, _ = build_feature_metadata(X, feature_list)

    preprocess = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('numeric', 'passthrough', num_cols),
        ],
        remainder='drop',
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = Pipeline(steps=[
        ('preprocess', preprocess),
        ('model', LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1)),
    ])
    clf.fit(X_train, y_train)

    return clf, feature_defaults, feature_types, feature_options, feature_list


def build_gui():
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        messagebox.showerror('Error', str(exc))
        return

    call_model, default_calls, max_calls = train_call_model(df)
    full_model, feature_defaults, feature_types, feature_options = train_feature_model(df, TOP_FEATURES)
    q2_model, q2_defaults, q2_types, q2_options, q2_features = train_q2_model(df)

    root = tk.Tk()
    root.title('Term Deposit Prediction (UI + Call Model)')
    root.geometry('820x900')

    # Basic styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', padding=2)
    style.configure('Section.TLabelframe', padding=10)
    style.configure('Section.TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
    style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#0b6fa4')
    style.configure('Result.TLabel', font=('Consolas', 11))
    style.configure('TButton', padding=(12, 6))

    main_frame = ttk.Frame(root, padding=18)
    main_frame.pack(fill='both', expand=True)
    main_frame.columnconfigure(0, weight=1)

    ttk.Label(main_frame, text='Term Deposit Predictor', style='Title.TLabel').grid(row=0, column=0, sticky='w')
    ttk.Label(
        main_frame,
        text='Combine call-count model (Question 1), term_deposit feature model, and a full-feature Question 2 model summary.',
    ).grid(row=1, column=0, sticky='w', pady=(0, 12))

    # Call count input
    calls_frame = ttk.LabelFrame(main_frame, text='Call-count model (q1_training)', style='Section.TLabelframe')
    calls_frame.grid(row=2, column=0, sticky='ew', pady=(0, 12))
    calls_frame.columnconfigure(1, weight=1)
    ttk.Label(calls_frame, text='Number of calls (campaign):').grid(row=0, column=0, sticky='w')
    calls_var = tk.IntVar(value=default_calls)
    calls_spin = ttk.Spinbox(
        calls_frame,
        from_=0,
        to=max_calls if max_calls > 0 else 50,
        textvariable=calls_var,
        width=12,
    )
    calls_spin.grid(row=0, column=1, sticky='w', pady=4, padx=(8, 0))

    # Full feature inputs
    features_frame = ttk.LabelFrame(main_frame, text='Top-feature model (term_deposit_feature_combo)', style='Section.TLabelframe')
    features_frame.grid(row=3, column=0, sticky='ew', pady=(0, 12))
    features_frame.columnconfigure(1, weight=1)

    widgets: Dict[str, tk.Widget] = {}
    for idx, feature in enumerate(TOP_FEATURES):
        ttk.Label(features_frame, text=feature).grid(row=idx, column=0, sticky='w', padx=(0, 8), pady=3)
        if feature_types[feature] == 'categorical':
            var = tk.StringVar(value=str(feature_defaults[feature]))
            combo = ttk.Combobox(features_frame, textvariable=var, values=feature_options[feature], width=24)
            combo.state(['readonly'])
            combo.grid(row=idx, column=1, sticky='ew', pady=3)
            widgets[feature] = combo
        else:
            var = tk.StringVar(value=str(feature_defaults[feature]))
            # Add hint for typical numeric range based on dataset quantiles
            hint = ''
            numeric_hints = {
                'emp.var.rate': 'typical 1.10 (1% -3.40, 99% 1.40)',
                'cons.price.idx': 'typical 93.75 (1% 92.20, 99% 94.47)',
                'euribor3m': 'typical 4.86 (1% 0.66, 99% 4.97)',
                'nr.employed': 'typical 5191 (1% 4963, 99% 5228)',
                'pdays': '0 if never contacted; 999 = not previously contacted',
                'previous': 'usually 0 (99% <=2)',
                'cons.conf.idx': 'typical -41.8 (1% -49.5, 99% -26.9)',
            }
            hint = numeric_hints.get(feature, '')
            entry = ttk.Entry(features_frame, textvariable=var, width=18)
            entry.grid(row=idx, column=1, sticky='w', pady=3)
            widgets[feature] = entry
            if hint:
                ttk.Label(features_frame, text=hint, foreground='#555').grid(row=idx, column=2, sticky='w', padx=(8, 0))

    # Results display
    result_frame = ttk.LabelFrame(main_frame, text='Results', style='Section.TLabelframe')
    result_frame.grid(row=5, column=0, sticky='ew', pady=(6, 0))
    result_text = tk.StringVar(value='Fill inputs and click Predict')
    result_label = ttk.Label(result_frame, textvariable=result_text, justify='left', style='Result.TLabel')
    result_label.pack(anchor='w', fill='x')

    def predict():
        # Gather call count
        try:
            call_count = int(calls_var.get())
            if call_count < 0:
                raise ValueError
        except Exception:
            messagebox.showerror('Input error', 'Calls must be a non-negative integer.')
            return

        # Gather feature values
        feature_values: Dict[str, object] = {}
        for name in TOP_FEATURES:
            widget = widgets[name]
            if feature_types[name] == 'categorical':
                feature_values[name] = widget.get()
            else:
                raw = widget.get().strip()
                if raw == '':
                    raw = feature_defaults[name]
                try:
                    feature_values[name] = float(raw)
                except ValueError:
                    messagebox.showerror('Input error', f'{name} must be numeric.')
                    return

        call_input = pd.DataFrame({'campaign': [call_count]})
        call_prob = float(call_model.predict_proba(call_input)[0, 1])

        full_input = pd.DataFrame([feature_values])
        full_prob = float(full_model.predict_proba(full_input[TOP_FEATURES])[0, 1])

        # Build Q2 input using defaults, override with term_deposit inputs where available, and campaign from call input.
        q2_values: Dict[str, object] = dict(q2_defaults)
        for name, val in feature_values.items():
            if name in q2_values:
                q2_values[name] = val
        if 'campaign' in q2_values:
            q2_values['campaign'] = call_count

        q2_input = pd.DataFrame([q2_values])
        q2_prob = float(q2_model.predict_proba(q2_input[q2_features])[0, 1])

        combined_score = (
            CALL_WEIGHT * call_prob
            + FULL_WEIGHT * full_prob
            + Q2_WEIGHT * q2_prob
        )
        call_decision = 'yes' if call_prob >= CALL_THRESHOLD else 'no'
        full_decision = 'yes' if full_prob >= FULL_THRESHOLD else 'no'
        q2_decision = 'yes' if q2_prob >= Q2_THRESHOLD else 'no'
        recommendation = 'Recommend calling this client.' if combined_score >= FULL_THRESHOLD else 'Do not prioritize calling.'

        result_lines = [
            f'Call-count only: P(yes)={call_prob:.3f} | thr {CALL_THRESHOLD:.2f} => {call_decision}',
            f'Full-feature:    P(yes)={full_prob:.3f} | thr {FULL_THRESHOLD:.2f} => {full_decision}',
            f'Question2 model: P(yes)={q2_prob:.3f} | thr {Q2_THRESHOLD:.2f} => {q2_decision} (uses inputs + defaults)',
            f'Combined (wts call {CALL_WEIGHT:.2f}, term_deposit {FULL_WEIGHT:.2f}, Q2 {Q2_WEIGHT:.2f}): {combined_score:.3f}',
            f'Recommendation:  {recommendation}',
        ]
        result_text.set('\n'.join(result_lines))

    btn_frame = ttk.Frame(main_frame)
    btn_frame.grid(row=4, column=0, sticky='ew', pady=8)
    btn_frame.columnconfigure(0, weight=1)
    ttk.Button(btn_frame, text='Predict', command=predict).grid(row=0, column=0, sticky='w')
    ttk.Button(btn_frame, text='Quit', command=root.destroy).grid(row=0, column=1, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    build_gui()
