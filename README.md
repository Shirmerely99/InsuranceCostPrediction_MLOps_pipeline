# Insurance Cost Prediction
Predict individual medical insurance charges using demographic and health factors. This project builds and compares regression models to estimate insurance costs from `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.

## Dataset
- Source: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle
- Size: 1,338 records
- Target: `charges` — Individual medical costs billed by health insurance
- Features:
  | Feature   | Description                                                  | Type        |
  | --- | --- | --- |
  | `age`     | Age of primary beneficiary                                   | int         |
  | `sex`     | Gender: male, female                                         | categorical |
  | `bmi`     | Body mass index                                              | float       |
  | `children`| Number of dependents covered                                 | int         |
  | `smoker`  | Smoking status: yes, no                                      | categorical |
  | `region`  | Residential area: northeast, southeast, southwest, northwest | categorical |

## Project Overview
The goal is to model the relationship between personal attributes and medical insurance charges. We implement a full ML pipeline: data cleaning, EDA, feature engineering, model training, and evaluation. The project emphasizes interpretability and comparing linear vs. tree-based models for regression tasks.

## Core Tech Stack
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `XGBoost`

## ML Pipeline
1. EDA & Cleaning: Checked distributions, outliers in `bmi`, and correlation with `charges`. No missing values.
2. Feature Engineering: One-hot encoded `sex`, `smoker`, `region`. Scaled numerical features for linear models.
3. Models Trained: Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost Regressor
4. Evaluation Metrics: RMSE, MAE, R² Score. Used 5-fold cross-validation.

## Results
| Model             | RMSE    | MAE     | R²   |
| --- | --- | --- | --- |
| Linear Regression | 5796.28 | 4181.19 | 0.78 |
| Random Forest     | 4568.93 | 2556.41 | 0.86 |
| XGBoost           | 4412.05 | 2491.50 | 0.87 |

**Key Insight:** `smoker` status is the dominant feature. Tree-based models outperformed linear models, with XGBoost achieving the best R² of 0.87. Smokers pay ~3.5x more on average.
