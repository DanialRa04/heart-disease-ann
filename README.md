# Heart Disease ANN

I rebuilt this project as a cleaner multiclass heart-disease study notebook. My goal here was to keep the ANN idea, but fix the leakage and make the evaluation easier to trust.

## Dataset

I used `data/heart_disease_uci.csv`, which has 920 rows and the original `num` target with classes `0` to `4`. The class balance is uneven, so I track macro F1 instead of accuracy alone.

## Structure

- `codes/ANN.ipynb`: corrected notebook with saved outputs
- `data/heart_disease_uci.csv`: required dataset

## Method

I dropped the patient id, kept the study-site column, and split the data before fitting any preprocessing steps. Numeric columns use median imputation plus scaling, categorical columns use most-frequent imputation plus one-hot encoding, and `SelectKBest` keeps the strongest transformed features inside the pipeline.

I compared a few ANN widths with 4-fold cross-validation on the training split and chose the final network from the best macro F1 score. After that, I fit the selected pipeline on the full train-validation portion and evaluated once on the untouched test split.

## Setup

The notebook runs with Python 3.12 plus `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `jupyter`. From the project root, open `codes/ANN.ipynb` and run all cells in order. The notebook can find `data/heart_disease_uci.csv` from the repo root, the project folder, or the `codes` folder.

## Evaluation

In the saved run, the selected ANN reached about `0.55` test accuracy and `0.38` macro F1. The model is much steadier on classes `0` and `1` than on the rare severe classes, so the confusion matrix matters more than the headline accuracy.

## Limits

This dataset is imbalanced, especially for class `4`, so the notebook still struggles on the rarest cases even after fixing the workflow. I kept the original multiclass target because that matches the project idea, but it is a harder problem than simple disease-vs-no-disease prediction.
