## Fraud Risk Modeling (Credit Card Transactions)



**Author:** Mohanad Alemam



**Start Date:** 14 October 2025 



**Status:** Completed and Production-ready.



---



### Project Overview

This project implements a Fraud Risk Modeling solution using End-to-end Machine Learning (ML) workflow. The solution is designed to detect fraudulent credit card transactions. The aim is early detection of fraud to help financial institutions mitigate Fraud Risk and achieve business objectives.



-----



### Dataset

- **Data and source:** European Credit Card Transactions, covering two days provided by Kaggle.  

- **Class balance and size:** 284,807 transactions with severe class imbalance fraud = 0.173% i.e. 492 fraud vs 284,315 non-fraud.  

- **Features:** 30 numerical features V1 to V28, Time and Amount.



-----------



### Workflow and Methodology

End to end Machine Learning (ML) best practices were applied including data exploration, feature engineering, model training and tuning evaluation, and client facing API. The exploratory Data Analysis (EDA) explored class imbalance and feature behavior. Some features show strong class separation others have low variance. No missing values were observed.



The EDA is followed by targeted Feature Engineering including non-monotonic transformations e.g. absolute and squaring, scaling of low variance features, and creating time based features e.g. hour\_of\_day and time\_segment. The engineered features evidently improved learning hidden patterns as they make 40% of the top ten most important predictors.



**Modeling and Evaluation:** Logistic Regression is deployed as a Baseline accompanied with advanced models including Random Forest, CatBoost, and LightGBM. These selections were based on EDA's results and the nature of these models. Hyperparameter tuning and Out-of-Fold (OOF) evaluation focused on positive/fraud class metrics.



**Calibration and Thresholding:** The production model was selected based on OOF evaluation results. The model's probabilities calibration was conducted including computing Brier score resulting in ~ 0.00037, indicating realistic and reliable probabilities. Then three Fraud Risk thresholds (High ≥0.80, Medium 0.30–0.80, Low <0.30) were calibrated to work as risk levels. The API allows business/users to adjust these thresholds based on domain knowledge and risk appetite.



**Final Production Model Performance Metrics (on test holdout data):** The table below lists the performance metrics for the final production model I.e. LightGBM (tuned). The model's confusion matrix confirms its strong positive class performance. These metrics are satisfactory and an indicator of the project's success in handling the severe class imbalance i.e. only 0.173% fraud prevalence.


| Class               | Precision | Recall | F1-score | Support | BalAcc | PR AUC |
|--------------------|-----------|--------|----------|---------|--------|--------|
| Fraud (Class 1)     | 0.95      | 0.776  | 0.854    | 98      | 0.888  | 0.857  |
| Non-fraud (Class 0) | 1.00      | 1.000  | 1.000    | 56864   | 0.888  | 0.857  |
| Macro Avg           | 0.975     | 0.888  | 0.927    | 56962   | 0.888  | 0.857  |



**Feature Importance and Explainability:** The most important predictors were computed based on gain, highlighting the impact of engineered features.  



**Client-facing API:** A business friendly API was built to return fraud probability, risk level risk indicator, and actionable recommendations. This API was validated using a smoke test across all risk levels.



----



### Project Structure



```

fraud-risk-modeling/

│

├── data/    # Raw and processed datasets

├── notebooks/     # EDA, feature engineering, modeling and analysis

├── src/   # Python code/Scripts data prep, modeling and API

├── results/   # Metrics, plots, and tables

├── trained_models/   # Saved models

├── .gitignore

├── LICENSE

├── README.md

└── requirements.txt

```

-------



### Outputs

- **Executive Summary:** `/results/Fraud_Risk_Modeling_Executive_Summary`  

- **Plots:** `/results/plots/` (PR curves, feature importance, and confusion matrices etc.)  

- **Tables:** `/results/tables/` (OOF/test metrics and feature importance etc.)



--------------



### License

This project is licensed under the MIT License. See LICENSE for details.

