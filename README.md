# Predicting Customer Churn for a Telecom Provider

**DSAI501 – How To Use Big-Data And AI To Make Better Business Decisions (Prof. Hunter Yan)**
Group Project, Option A · Team 16: Kseniia Reznyk, Sofia Angely León Rodriguez, Neel Laxmisan, Rhyday Sadh

## About the project

A telecom company loses some of its customers every month (**churn**). We built a model that predicts which
customers are likely to leave, so a retention team can contact them before they do.

What the notebook covers:
- data cleaning and exploratory data analysis (EDA)
- feature engineering with justification
- two classification models compared: **logistic regression** and a **tuned random forest**
- evaluation with accuracy, precision, recall and F1-score, plus confusion matrices
- choosing the cutoff, feature importance, a fairness check and the model on test cases
- a business view: can the company act on the predictions, and is the model ready to deploy?

**Final model:** random forest with a 0.3 cutoff. On the 1,409 test customers it gets recall 0.773,
precision 0.527, F1 0.627 and accuracy 0.756, so it catches about 3 out of 4 churners.

## Repository contents

| File / folder | What it is |
|---|---|
| `Telco_Churn_Project.ipynb` | The full notebook, from loading the data to the business recommendations. All outputs are saved, so you can read it without re-running it. |
| `churn_model.joblib` | The trained final model. It is a dictionary with the fitted random forest (`model`), the cutoff (`cutoff` = 0.3) and the feature list (`features`). |
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | The raw dataset, exactly as downloaded from Kaggle. |
| `data/telco_clean.csv` | Cleaned data, still readable (7,043 rows × 20 columns). |
| `data/telco_model_ready.csv` | Fully numeric data used for modelling (7,043 rows × 26 columns). |
| `figures/` | All 10 charts from the notebook (fig1–fig10), as PNG files. |

## How to run it

**Google Colab (easiest)**
1. Open `Telco_Churn_Project.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. In the Files panel on the left, create a folder called `data` and upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` into it.
   Or clone the whole repository in the first cell:
   ```
   !git clone https://github.com/s-angely/telco-customer-churn-prediction.git
   %cd telco-customer-churn-prediction
   ```
3. Click **Runtime → Run all**. The notebook rebuilds the clean data, the figures and `churn_model.joblib`.

**On your own computer**
```
git clone https://github.com/s-angely/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook Telco_Churn_Project.ipynb
```

**Using the saved model**
```python
import joblib
saved = joblib.load("churn_model.joblib")
model, cutoff, features = saved["model"], saved["cutoff"], saved["features"]
# X = a DataFrame with the same 25 columns as `features` (see data/telco_model_ready.csv)
prob = model.predict_proba(X[features])[:, 1]
contact = prob >= cutoff   # True = likely to churn, contact the customer
```
The model was trained with Python 3.13 and scikit-learn 1.9.1. With a different scikit-learn version, loading
may show a warning; if it fails, re-run the notebook to create a fresh model file.

## Dataset source documentation

| | |
|---|---|
| **Dataset** | Telco Customer Churn |
| **Source** | Kaggle, uploaded by BlastChar (2018): https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Original creator** | IBM, as a sample dataset for its analytics products |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` (CSV, about 955 KB) |
| **Size** | 7,043 customers (rows) × 21 columns |
| **Date accessed** | 15 October 2026 |
| **Licence / use** | Publicly available sample data, used for educational purposes only. The customers are fictional, so there is no real personal data. |

**Columns:** `customerID`; demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`); account
(`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`); services
(`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
`TechSupport`, `StreamingTV`, `StreamingMovies`); target `Churn` (Yes/No, 26.5% Yes).

### Preprocessing steps applied before analysis
1. Converted `TotalCharges` from text to numbers. Its 11 blank values all belonged to new customers with tenure = 0,
   who had not been billed yet, so they were filled with 0.
2. Checked missing values (none left), duplicate rows (0) and duplicate customer IDs (0).
3. Checked outliers in `tenure`, `MonthlyCharges` and `TotalCharges` with boxplots and the 1.5×IQR rule. None were found, so all rows were kept.
4. Dropped `customerID`, because it is only an identifier.
5. Replaced "No internet service" / "No phone service" with "No" in 7 service columns.
6. Converted Yes/No columns and the target `Churn` to 1/0, and `gender` to 1 = Male, 0 = Female.
7. Created new features: `num_addon_services` (0–6), `auto_payment` (1 = bank transfer or credit card) and
   `new_customer` (1 = tenure of 12 months or less).
8. Dropped `TotalCharges`, because it is strongly correlated with `tenure` (r = 0.83).
9. Created n−1 dummy variables for `Contract`, `InternetService` and `PaymentMethod`
   (reference levels: month-to-month, DSL, bank transfer).
10. Saved `data/telco_clean.csv` and `data/telco_model_ready.csv`.

Scaling (StandardScaler, for logistic regression only) happens later inside the modelling pipeline. It is fitted on
the training data only after the 80/20 stratified train/test split, so nothing leaks from the test set.

## Tools and libraries
Python, Google Colab, pandas, NumPy, matplotlib, seaborn, scikit-learn and joblib. Full APA references are in the project report.
