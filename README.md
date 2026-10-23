# P07 — Term-Deposit Call Prioritisation (Bank Marketing)

Ranks bank customers by their probability of subscribing to a term deposit, so that a call centre with a fixed daily call budget can phone the most promising customers first.

## Dataset

- **Bank Marketing (bank-additional-full)** — Moro, Cortez & Rita (2014), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing (licence: CC BY 4.0)
- Version used: [paste the exact link where you downloaded the CSV]
- 41,188 rows × 21 columns; target column `subscribed` (yes / no); positive rate 11.27%.
- Place the file at: `data/bank_marketing_dataset.csv`

## How to run

1. Python 3.10 or later. Install the dependencies:

   ```
   pip install pandas numpy scikit-learn scipy jupyter
   ```

2. Open `analysis.ipynb` and run it top to bottom (**Kernel → Restart & Run All**, or in Colab **Runtime → Run all**).
   - In Colab, first upload the CSV and make sure the path in the loading cell matches where the file is.
   - Runtime: about [X] minutes. The two model searches and the shuffled-label check are the slowest cells.

## Files

| File | Contents |
|---|---|
| `analysis.ipynb` | R1–R5: decision, data checks, train/test split, metric, search over two model families, final test evaluation (scored once). R6: the five checks (baselines, single feature, split stability, shuffled labels, groups and time), placed after the final evaluation as diagnostics that do not change the model. |
| `report.md` | Write-up in six sections: decision, data, metric, search, result, checks and limitations. |
| `README.md` | This file. |

## Key result

Histogram gradient boosting, cross-validated AP **0.466 ± 0.022**, test AP **0.484** (base rate 0.113). Calling the top 10% of customers reaches **46% of all subscribers**, about **4.6×** as many as random calling.
