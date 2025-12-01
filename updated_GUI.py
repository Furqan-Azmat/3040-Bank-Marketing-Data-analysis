import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATHS = [
    Path('Data/bank-additional-full.csv'),
    Path('bank-additional-full.csv'),
    Path('ITEC3040-Final-Project/Data/bank-additional-full.csv'),
]
TOP_FEATURES = [
    'euribor3m',
    'age',
    'campaign',
    'nr.employed',
    'pdays',
    'emp.var.rate',
    'cons.conf.idx',
]
DISPLAY_NAMES = {
    'euribor3m': 'Euro Interbank 3 month Offered Rate',
    'age': 'Age',
    'campaign': 'Calls this campaign',
    'nr.employed': 'Employment level (nr.employed)',
    'pdays': 'Days since last contact (999 = never)',
    'emp.var.rate': 'Employment variation rate',
    'cons.conf.idx': 'Consumer confidence index',
}
DISPLAY_DESCRIPTIONS = {
    'euribor3m': 'Interest rate in Europe for 3-month loans between banks',
    'age': 'Customer age in years.',
    'campaign': 'How many times this customer was contacted in the current campaign.',
    'nr.employed': 'Overall employment level in the economy.',
    'pdays': 'Days since this customer was last contacted; 999 means they were not contacted before.',
    'emp.var.rate': 'Recent change in employment levels; shows whether the job market is growing or shrinking.',
    'cons.conf.idx': 'Consumer confidence; How people feel about the economy.',
}

CALL_THRESHOLD = 0.10
FULL_THRESHOLD = 0.40
Q2_THRESHOLD = 0.50
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


def build_feature_metadata(df: pd.DataFrame, feature_list: List[str]) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, list]]:
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    feature_defaults: Dict[str, object] = {}
    feature_types: Dict[str, str] = {}
    feature_options: Dict[str, list] = {}
    for col in feature_list:
        if col in cat_cols:
            feature_defaults[col] = df[col].mode(dropna=True)[0]
            feature_types[col] = 'categorical'
            feature_options[col] = sorted(df[col].dropna().unique().tolist())
        else:
            feature_defaults[col] = float(df[col].median())
            feature_types[col] = 'numeric'
            feature_options[col] = []

    return feature_defaults, feature_types, feature_options


def train_feature_model(df: pd.DataFrame, feature_list: List[str]) -> Tuple[Pipeline, Dict[str, object], Dict[str, str], Dict[str, list]]:
    y = (df['y'] == 'yes').astype(int)
    X = df.drop(columns=['y'])

    feature_defaults, feature_types, feature_options = build_feature_metadata(X, feature_list)
    use_cat = [c for c in feature_list if c in X.select_dtypes(include=['object']).columns]
    use_num = [c for c in feature_list if c not in use_cat]

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
    y = (df['y'] == 'yes').astype(int)
    X = df.drop(columns=['y'])
    feature_list = list(X.columns)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    feature_defaults, feature_types, feature_options = build_feature_metadata(X, feature_list)

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


def attach_tooltip(widget: tk.Widget, text: str) -> None:
    """Simple hover tooltip for feature info."""
    if not text:
        return
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    tooltip.attributes('-topmost', True)
    lbl = ttk.Label(
        tooltip,
        text=text,
        background='#ffffe0',
        relief='solid',
        borderwidth=1,
        padding=4,
        wraplength=260,
        justify='left',
    )
    lbl.pack()

    def enter(_event: object) -> None:
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        tooltip.geometry(f'+{x}+{y}')
        tooltip.deiconify()

    def leave(_event: object) -> None:
        tooltip.withdraw()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


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
    root.title('Term Deposit Prediction')
    root.geometry('1100x900')

    bg = '#f5f7fa'
    card_bg = '#ffffff'
    accent = '#0b6fa4'
    root.configure(background=bg)
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', padding=2, background=bg, font=('Segoe UI', 10))
    style.configure('Body.TLabel', background=bg, font=('Segoe UI', 10))
    style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground=accent, background=bg)
    style.configure('Section.TLabelframe', padding=12, background=card_bg)
    style.configure('Section.TLabelframe.Label', font=('Segoe UI', 11, 'bold'), foreground=accent, background=card_bg)
    style.configure('Result.TLabel', font=('Consolas', 11), background=card_bg)
    style.configure('Card.TFrame', background=card_bg, relief='ridge', borderwidth=1, padding=12)
    style.configure('Main.TFrame', background=bg)
    style.configure('Accent.TButton', padding=(12, 8), font=('Segoe UI', 10, 'bold'), foreground='#ffffff', background=accent)
    style.map('Accent.TButton', background=[('active', '#095a82')])

    main_frame = ttk.Frame(root, padding=18, style='Main.TFrame')
    main_frame.pack(fill='both', expand=True)
    main_frame.columnconfigure(0, weight=1)

    ttk.Label(main_frame, text='Term Deposit Predictor', style='Title.TLabel').grid(row=0, column=0, sticky='w')
    ttk.Label(
        main_frame,
        text='Predict using call-count (Q1), Q3 top-7 features, and Q2 high/low groups. Defaults shown below are medians/modes.',
        style='Body.TLabel',
    ).grid(row=1, column=0, sticky='w', pady=(0, 10))

    calls_frame = ttk.LabelFrame(main_frame, text='Call-count model (Q1)', style='Section.TLabelframe')
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

    features_frame = ttk.LabelFrame(main_frame, text='Q3 top-7 feature model', style='Section.TLabelframe')
    features_frame.grid(row=3, column=0, sticky='ew', pady=(0, 12))
    features_frame.columnconfigure(1, weight=1)
    ttk.Label(
        features_frame,
        text='Defaults are pre-filled (median/mode). Leave blank to use defaults. All fields are numeric.',
        foreground='#555',
    ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 6))

    widgets: Dict[str, tk.Widget] = {}
    campaign_var: tk.StringVar | None = None
    numeric_hints = {
        'euribor3m': 'typical 0.6–5.0 (econ)',
        'age': '18–95 typical',
        'campaign': 'Calls this campaign; auto-synced from Q1',
        'nr.employed': 'Employment level (econ), typical 4960–5230',
        'pdays': 'Days since last contact; 999 = not previously contacted',
        'emp.var.rate': 'Quarterly employment variation rate (econ); typical -3.4 to 1.4',
        'cons.conf.idx': 'Consumer confidence index (econ); typical -50 to -26',
    }
    for idx, feature in enumerate(TOP_FEATURES, start=1):
        label_txt = DISPLAY_NAMES.get(feature, feature)
        name_lbl = ttk.Label(features_frame, text=label_txt)
        name_lbl.grid(row=idx, column=0, sticky='w', padx=(0, 8), pady=3)
        qmark = ttk.Label(features_frame, text='?', foreground='#0b6fa4', cursor='question_arrow')
        qmark.grid(row=idx, column=3, sticky='w', padx=(4, 0))
        attach_tooltip(qmark, DISPLAY_DESCRIPTIONS.get(feature, ''))
        if feature_types[feature] == 'categorical':
            var = tk.StringVar(value=str(feature_defaults[feature]))
            combo = ttk.Combobox(features_frame, textvariable=var, values=feature_options[feature], width=24)
            combo.state(['readonly'])
            combo.grid(row=idx, column=1, sticky='ew', pady=3)
            widgets[feature] = combo
        else:
            var = tk.StringVar(value=str(feature_defaults[feature]))
            entry = ttk.Entry(features_frame, textvariable=var, width=18)
            entry.grid(row=idx, column=1, sticky='w', pady=3)
            widgets[feature] = entry
            base_hint = numeric_hints.get(feature, '')
            default_txt = f"default {feature_defaults[feature]}"
            hint = f"{base_hint}; {default_txt}" if base_hint else default_txt
            if hint:
                ttk.Label(features_frame, text=hint, foreground='#555').grid(row=idx, column=2, sticky='w', padx=(8, 0))
            if feature == 'campaign':
                campaign_var = var

    def sync_campaign(*_args: object) -> None:
        if campaign_var is None or sync_state['active']:
            return
        try:
            val = calls_var.get()
        except Exception:
            return
        if val == '' or val is None:
            return
        sync_state['active'] = True
        try:
            campaign_var.set(str(val))
        finally:
            sync_state['active'] = False

    def sync_calls(*_args: object) -> None:
        if campaign_var is None or sync_state['active']:
            return
        raw = campaign_var.get().strip()
        if raw == '':
            return
        try:
            num = int(float(raw))
            if num < 0:
                return
        except Exception:
            return
        sync_state['active'] = True
        try:
            calls_var.set(num)
        finally:
            sync_state['active'] = False

    sync_state = {'active': False}
    calls_var.trace_add('write', sync_campaign)
    if campaign_var is not None:
        campaign_var.trace_add('write', sync_calls)
    sync_campaign()

    result_frame = ttk.LabelFrame(main_frame, text='Results', style='Section.TLabelframe')
    result_frame.grid(row=5, column=0, sticky='ew', pady=(6, 0))
    result_container = ttk.Frame(result_frame)
    result_container.grid(row=0, column=0, sticky='nsew')
    result_container.columnconfigure(0, weight=3)
    result_container.columnconfigure(1, weight=2)
    result_text = tk.StringVar(value='Fill inputs and click Predict')
    result_label = ttk.Label(
        result_container,
        textvariable=result_text,
        justify='left',
        style='Result.TLabel',
        padding=(0, 0, 8, 0),
        wraplength=420,
    )
    result_label.grid(row=0, column=0, sticky='nw')
    chart_frame = ttk.Frame(result_container)
    chart_frame.grid(row=0, column=1, sticky='ne', padx=(12, 0))
    result_frame.columnconfigure(0, weight=1)
    chart_holder: Dict[str, object] = {'canvas': None}

    def predict():
        try:
            call_count = int(calls_var.get())
            if call_count < 0:
                raise ValueError
        except Exception:  # noqa: BLE001
            messagebox.showerror('Input error', 'Calls must be a non-negative integer.')
            return

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
        q2_label = 'high-response' if q2_prob >= Q2_THRESHOLD else 'low-response'
        recommendation = 'Recommend calling this client.' if combined_score >= FULL_THRESHOLD else 'Do not prioritize calling.'

        # Visualization: bar chart of probabilities
        labels = ['Calls', 'Top 7', 'High/Low', 'Combined']
        values = [call_prob, full_prob, q2_prob, combined_score]
        colors = ['#4c9be8', '#7dcfb6', '#f5a623', '#0b6fa4']
        fig = Figure(figsize=(5.5, 2.4), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, values, color=colors)
        ax.axhline(FULL_THRESHOLD, color='#888', linestyle='--', linewidth=1, label=f'Threshold {FULL_THRESHOLD:.2f}')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('P(yes)')
        ax.set_title('Predicted probabilities')
        ax.legend(loc='upper right')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        if chart_holder['canvas'] is not None:
            chart_holder['canvas'].get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(anchor='ne', pady=(0, 0))
        chart_holder['canvas'] = canvas
        root.update_idletasks()

        result_lines = [
            f'Call-count only:   P(yes)={call_prob:.3f} | thr {CALL_THRESHOLD:.2f} => {call_decision}',
            f'Top-features:      P(yes)={full_prob:.3f} | thr {FULL_THRESHOLD:.2f} => {full_decision}',
            f'High or Low group: P(yes)={q2_prob:.3f} | thr {Q2_THRESHOLD:.2f} => {q2_decision} ({q2_label})',
            f'Combined (wts call {CALL_WEIGHT:.2f}, top {FULL_WEIGHT:.2f}, Q2 {Q2_WEIGHT:.2f}): {combined_score:.3f}',
            f'Combined meets threshold? {"yes" if combined_score >= FULL_THRESHOLD else "no"} (thr {FULL_THRESHOLD:.2f})',
            f'Recommendation:  {recommendation}',
        ]
        result_text.set('\n'.join(result_lines))

    btn_frame = ttk.Frame(main_frame)
    btn_frame.grid(row=4, column=0, sticky='ew', pady=8)
    btn_frame.columnconfigure(0, weight=1)
    ttk.Button(btn_frame, text='Predict', command=predict, style='Accent.TButton').grid(row=0, column=0, sticky='w')
    ttk.Button(btn_frame, text='Quit', command=root.destroy).grid(row=0, column=1, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    build_gui()
