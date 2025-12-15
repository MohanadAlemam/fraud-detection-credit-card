FRAUD RISK MODELING (CREDIT CARD TRANSACTIONS)

Author: Mohanad Alemam
Start Date: 14 October 2025
Status: Completed / Production-ready


PROJECT OVERVIEW

This project implements a Fraud Risk Modeling solution using End-to-end Machine Learning (ML) workflow. The solution is designed to detect fraudulent credit card transactions. The aim is early detection of fraud to help financial institutions mitigating Fraud Risk and achieving business objectives.


DATASET

Data and source: European Credit Card Transactions, covering two days provided by Kaggle.

Class balance and size: 284,807 transactions with severe class imbalance fraud = 0.173% i.e. 492 fraud vs 284,315 non-fraud.

Features: 30 numerical features V1 to V28, Time and Amount.


WORKFLOW AND METHODOLOGY

End-to-end Machine Learning (ML) workflow best practices were followed including (data exploration, feature engineering, model training and tuning evaluation and a client-facing API). The exploratory Data Analysis (EDA) explored class imbalance and feature behavior. Some features show strong class separation others have low variance. No missing values were observed.

The EDA is followed by targeted Feature Engineering including Non-monotonic transformations eg. absolute and squaring, scaling of low-variance features, and creating time-based features (hour_of_day and time_segment). The engineered features significantly improved learning the hidden patterns representing 40% of the most important predictors.

Modeling and Evaluation: Baseline (Logistic Regression) and advanced models (Random Forest, CatBoost, LightGBM) were used based on the EDA and the nature of the task. After that Hyperparameter tuning and Out of Fold (OOF) evaluation focused on positive class metrics/ fraud.

Calibration and Thresholding: The production model was selected based OOF evaluation. Brier score calibration resulted in ~ 0.00037 indicating realistic and reliable probabilities a three-tier Fraud Risk thresholds (High ≥0.80 Medium 0.30–0.80 and Low <0.30) were calibrated to work as risk levels.

Final Production Model Performance Metrics (on test/holdout data): The table hereunder lists the performance metrics for the final production model. The model's confusion matrix confirms strong positive-class performance. These metrics are satisfactory and a key indicator of the project's success in handling the severe class imbalance (fraud prevalence = 0.173%).

Class                  Precision  Recall  F1-score  Support  BalAcc  PR AUC
---------------------------------------------------------------------------
Fraud (Class 1)        0.95       0.776   0.854     98       0.888   0.857
Non-fraud (Class 0)    1.00       1.000   1.000     56864    0.888   0.857
Macro Avg              0.975      0.888   0.927     56962    0.888   0.857

Feature Importance and Explainability: The Most important predictors were computed based on gain, highlighting the impact of engineered features.

Client-facing API: Business-friendly API that returns fraud probability, risk level risk indicator and actionable recommendations validated using smoke test across all risk tiers.


PROJECT STRUCTURE

fraud_risk_modeling/
│
├── data/       # Raw and  processed datasets
├── notebooks/  # EDA, feature engineering modeling and analysis 
├── src/        # Python code/Scripts (data prep, modeling, API)
├── results/    # Metrics plots  and tables
├── trained_models/  # Saved models
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


OUTPUTS

Executive Summary: /results/Fraud_Risk_Modeling_Executive_Summary

Plots: /results/plots/ (PR curves, feature importance and confusion matrices etc)

Tables: /results/tables/ (OOF/test metrics and feature importance etc)


LICENSE

This project is licensed under the MIT License (see LICENSE for details).