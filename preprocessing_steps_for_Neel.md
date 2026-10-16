# Dataset Source Documentation – info for Neel

**Dataset:** Telco Customer Churn (IBM sample data set)
**Source:** Kaggle – https://www.kaggle.com/datasets/blastchar/telco-customer-churn (file: WA_Fn-UseC_-Telco-Customer-Churn.csv)
**Original creator:** IBM (sample data for IBM Cognos Analytics / Watson); uploaded to Kaggle by BlastChar (2018)
**Accessed:** 15 October 2026
**Size:** 7,043 rows × 21 columns, CSV (~955 KB)
**Licence / use:** publicly available sample data, used for educational purposes only; fictional customers, no personal data

## Preprocessing steps applied (in Part1_Data_Cleaning_EDA.ipynb)
1. Converted TotalCharges from text to numeric; 11 blank values (all customers with tenure = 0, not billed yet) filled with 0.
2. Checked missing values (none left), duplicate rows (0) and duplicate customer IDs (0).
3. Checked outliers in tenure, MonthlyCharges, TotalCharges with boxplots and the 1.5×IQR rule – none found, all rows kept.
4. Dropped customerID (identifier only).
5. Replaced "No internet service" / "No phone service" with "No" in 7 service columns.
6. Converted Yes/No columns and the target Churn to 1/0; gender to 1 = Male, 0 = Female.
7. Created new features: num_addon_services (0–6), auto_payment (1 = bank transfer/credit card), new_customer (1 = tenure ≤ 12 months).
8. Dropped TotalCharges (r = 0.83 with tenure → multicollinearity).
9. Created n−1 dummy variables for Contract, InternetService, PaymentMethod (reference: month-to-month, DSL, bank transfer).
10. Saved data/telco_clean.csv (readable, 7,043 × 20) and data/telco_model_ready.csv (numeric, 7,043 × 26).
Scaling (StandardScaler) is done in the modelling notebook after the train/test split, fitted on training data only.
