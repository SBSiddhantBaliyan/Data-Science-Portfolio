# Telco Customer Churn Prediction

Machine learning assignment: predicting customer churn on the IBM Telco Customer
Churn dataset using Decision Trees, rule-based classification, kNN, and ensemble
methods (Random Forest, AdaBoost) — with every modelling decision tied back to
bias-variance / overfitting concepts.

**Author:** Siddhant Baliyan 

## Project structure

```
telco-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Customer Churn dataset (7,043 rows)
├── notebooks/
│   └── telco_churn_assignment.ipynb           # Full analysis, Tasks A–F, all outputs pre-run
├── report/
│   └── telco_churn_report.pdf                 # 4-page summary report
├── requirements.txt
└── README.md
```

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
7,043 customers, 21 attributes (20 features + binary `Churn` target). ~26.5% churn rate.

## What's inside the notebook

| Task | Contents |
|---|---|
| **A** | Missing value / duplicate / outlier analysis, attribute classification (Nominal/Ordinal/Numeric) with reasoning, encoding, standardization, stratified 80/20 train-test split |
| **B** | Decision Tree training, confusion matrix + Accuracy/Precision/Recall/Specificity/NPV, hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`) with train-vs-test accuracy curves tied to bias-variance, tree visualization, 2 traced decision paths in plain language |
| **C** | Rule extraction from the tuned Decision Tree, top 5 rules with coverage/confidence, rule-set accuracy, interpretability-vs-performance discussion |
| **D** | kNN at k = 1, 3, 5, 7, 9, accuracy/error-rate vs. k, empirical scaled-vs-unscaled comparison, bias-variance vs. k |
| **E** | Tuned Random Forest (bagging) and AdaBoost (boosting), compared against Decision Tree and kNN |
| **F** | Final comparison table (auto-generated from results), best model / most interpretable model / deployment risk discussion |

## Results summary

| Model | Accuracy | Precision | Recall | Specificity |
|---|---|---|---|---|
| Decision Tree (baseline) | 0.731 | 0.493 | 0.500 | 0.814 |
| Decision Tree (tuned) | 0.786 | 0.617 | 0.516 | 0.884 |
| kNN (k=9, scaled) | 0.775 | 0.579 | 0.559 | 0.853 |
| Random Forest (tuned) | 0.799 | 0.655 | 0.513 | 0.902 |
| AdaBoost (tuned) | 0.799 | 0.644 | 0.543 | 0.892 |

Ensembles (Random Forest / AdaBoost) give the best raw accuracy; the tuned Decision
Tree and its extracted rules remain the most interpretable for a retention team that
needs to know *why* a customer is flagged. Full discussion in


## Running it

```bash
git clone <this-repo-url>
cd telco-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/telco_churn_assignment.ipynb
```

The notebook is already fully executed — every cell's output (tables, confusion
matrices, plots) is saved in the `.ipynb` file, so you can read it top-to-bottom on
GitHub without re-running anything. Re-run only if you want to reproduce or modify
the analysis.

## Key takeaways

- Fixed 11 hidden missing values in `TotalCharges` (blank strings tied to
  `tenure == 0`) that a naive `isnull()` check misses.
- `Contract` was ordinal-encoded rather than one-hot encoded — it turned out to be
  the strongest churn predictor the tree finds.
- Hyperparameter tuning cut the Decision Tree's train-test accuracy gap from
  **0.267 (baseline) to 0.037 (tuned)**.
- kNN accuracy dropped measurably when features weren't scaled — verified
  empirically, not just asserted.
- Ensembles beat the single tuned tree on accuracy, but the tree/rules remain the
  most business-usable explanation of individual predictions.

## License

MIT — see [LICENSE](LICENSE).
