"""Chronological bank campaign benchmark; no same-call duration or test tuning."""
from pathlib import Path
import hashlib, io, json, math, os, urllib.request, zipfile
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip'
CATEGORICAL = ['contact', 'month', 'day_of_week', 'poutcome']
NUMERIC = ['campaign', 'pdays', 'previous']
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = ['duration', 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

def load_source():
    path = ROOT / 'data' / 'bank-additional-full.csv'
    if not path.exists():
        (ROOT / 'data').mkdir(exist_ok=True)
        source = ROOT / 'data' / 'source.zip'
        if not source.exists():
            urllib.request.urlretrieve(URL, source)
        with zipfile.ZipFile(source) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read('bank-additional.zip'))) as inner:
                path.write_bytes(inner.read('bank-additional/bank-additional-full.csv'))
    frame = pd.read_csv(path, sep=';')
    assert len(frame) == 41188
    assert set(FEATURES + ['y']) <= set(frame)
    assert frame.y.isin(['yes', 'no']).all() and not frame[FEATURES + ['y']].isna().any().any()
    return frame, hashlib.sha256(path.read_bytes()).hexdigest()

def make_pipeline(estimator):
    preprocessor = ColumnTransformer([('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                                      ('numeric', StandardScaler(), NUMERIC)])
    return Pipeline([('preprocess', preprocessor), ('model', estimator)])

def metrics(y, p):
    k = max(1, math.ceil(len(y) * .1))
    top = np.argsort(-p, kind='stable')[:k]
    precision = float(np.asarray(y)[top].mean())
    return {'average_precision': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p)), 'prevalence': float(np.mean(y)),
            'top_10pct_precision': precision, 'top_10pct_lift': precision / float(np.mean(y)),
            'top_10pct_capture': float(np.asarray(y)[top].sum() / np.asarray(y).sum()),
            'top_10pct_records': k}

def calibration_table(y, p):
    bins = np.minimum((p * 10).astype(int), 9)
    return [{'bin': i, 'records': int((bins == i).sum()), 'mean_probability': float(p[bins == i].mean()),
             'observed_rate': float(np.asarray(y)[bins == i].mean())}
            for i in range(10) if (bins == i).sum()]

def main():
    frame, sha = load_source()
    y = (frame.y == 'yes').astype(int)
    n = len(frame)
    a, b, c = int(n * .6), int(n * .7), int(n * .8)
    split = {'train': (0, a), 'selection': (a, b), 'calibration': (b, c), 'test': (c, n)}
    candidates = {'logistic': make_pipeline(LogisticRegression(C=1.0, max_iter=1500)),
                  'random_forest': make_pipeline(RandomForestClassifier(n_estimators=180, max_depth=10,
                                      min_samples_leaf=30, random_state=42, n_jobs=2))}
    validation = {}
    for name, model in candidates.items():
        model.fit(frame.iloc[:a][FEATURES], y.iloc[:a])
        validation[name] = metrics(y.iloc[a:b], model.predict_proba(frame.iloc[a:b][FEATURES])[:, 1])
    chosen = max(candidates, key=lambda name: validation[name]['average_precision'])
    model = candidates[chosen]
    # Fit a separate sigmoid on the untouched calibration block; base model stays frozen.
    cp = np.clip(model.predict_proba(frame.iloc[b:c][FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1e6, max_iter=1500)
    calibrator.fit(np.log(cp / (1 - cp)).reshape(-1, 1), y.iloc[b:c])
    raw = np.clip(model.predict_proba(frame.iloc[c:][FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    calibrated = calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
    test_y = y.iloc[c:].to_numpy()
    prevalence = np.full(len(test_y), y.iloc[:a].mean())
    summary = {'source_rows': n, 'source_sha256': sha, 'source_url': URL,
               'selected_model': chosen, 'features': FEATURES, 'excluded_features': EXCLUDED,
               'split_0_based_end_exclusive': {k: {'start': s, 'end': e, 'records': e - s,
                                                     'positive_rate': float(y.iloc[s:e].mean())} for k, (s, e) in split.items()},
               'selection_scores': validation, 'test_uncalibrated': metrics(test_y, raw),
               'test_calibrated': metrics(test_y, calibrated), 'test_training_prevalence_baseline': metrics(test_y, prevalence),
               'calibration_bins': calibration_table(test_y, calibrated),
               'test_probability_mean': float(calibrated.mean()),
               'protocol': 'Input row order is chronological per UCI; no exact dates or client IDs supplied. 60/10/10/20 ordered blocks. Model chosen by selection AP. Sigmoid fitted only on calibration block. Test untouched until reporting.'}
    out = ROOT / 'outputs'; out.mkdir(exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Publish aggregates, never individual customer features or predictions.
    pd.DataFrame(summary['calibration_bins']).to_csv(out / 'calibration_bins.csv', index=False)
    rows = []
    for fraction in [.05, .1, .2, .3, .5, 1.0]:
        k = math.ceil(len(test_y) * fraction)
        ids = np.argsort(-calibrated, kind='stable')[:k]
        rows.append({'contact_fraction': fraction, 'records': k, 'positive_rate': float(test_y[ids].mean()),
                     'captured_positive_fraction': float(test_y[ids].sum() / test_y.sum())})
    pd.DataFrame(rows).to_csv(out / 'ranking_curve.csv', index=False)
    checks = {'source_row_count': n == 41188, 'disjoint_ordered_blocks': a < b < c < n,
              'duration_and_outcome_excluded': 'duration' not in FEATURES and 'y' not in FEATURES,
              'personal_financial_and_demographic_features_excluded': not bool(set(FEATURES) & set(EXCLUDED)),
              'finite_probabilities_in_unit_interval': bool(np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all()),
              'test_has_both_classes': len(np.unique(test_y)) == 2,
              'model_selection_matches_selection_ap': chosen == max(validation, key=lambda k: validation[k]['average_precision']),
              'calibration_count_reconciles': sum(r['records'] for r in summary['calibration_bins']) == len(test_y)}
    assert all(checks.values())
    (out / 'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for p, label in [(raw, 'Uncalibrated'), (calibrated, 'Separate-block sigmoid')]:
        observed, predicted = calibration_curve(test_y, p, n_bins=8, strategy='quantile')
        axes[0].plot(predicted, observed, marker='o', label=label)
    axes[0].plot([0, 1], [0, 1], '--', color='gray'); axes[0].set(xlabel='Predicted subscription probability', ylabel='Observed subscription rate', title='Held-out reliability'); axes[0].legend(fontsize=8)
    axes[1].plot([r['contact_fraction'] for r in rows], [r['captured_positive_fraction'] for r in rows], marker='o', color='#bf5c24')
    axes[1].plot([0, 1], [0, 1], '--', color='gray'); axes[1].set(xlabel='Fraction ranked highest', ylabel='Fraction of positives captured', title='Ranking, not a profit claim')
    fig.suptitle('Bank response | Chronological holdout, no call-duration shortcut', x=.05, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(out / 'evaluation.svg', metadata={'Date': None}); fig.savefig(out / 'evaluation.png', dpi=140); plt.close(fig)
    print(json.dumps({'chosen': chosen, 'test': summary['test_calibrated'], 'checks_passed': sum(checks.values())}, indent=2))

if __name__ == '__main__': main()
