# Model card

Intended use: audit a historical prediction workflow and demonstrate the difference between ranking, calibration and a business decision. No financial advice, credit decision or production marketing use is intended.

Features: contact, month, day_of_week, poutcome, campaign, pdays, previous. `pdays=999` is the source's not-previously-contacted sentinel, retained as a numerical code; this is an acknowledged modelling limitation. Unknown categorical values are handled by the one-hot encoder without fit-time access to later data. Logistic C=1, max_iter=1500. Forest: 180 trees, depth 10, minimum leaf 30, seed 42. Separate sigmoid calibration uses log-odds, clipped to avoid infinities. Excluded inputs are listed in metrics.json.

Method selection is based on selection-block average precision, appropriate to ranking in an imbalanced cohort. Brier/log loss assess probability quality. ROC AUC is reported alongside AP, not used as a substitute. Top-10% lift divides the outcome rate in the highest-ranked tenth by the actual test rate; tied scores use stable source order. Calibration is monotonic, so ranking is essentially preserved apart from numerical ties.

The large prevalence change is measured, not explained causally. No client IDs exist to enforce entity isolation, so customer overlap cannot be ruled out. Excluding explicit demographics and finance fields reduces unnecessary exposure but is not a fairness audit. A live use case requires available-at-decision-time feature validation, entity-safe evaluation, current consent rules, a prospective randomized test and governance.
