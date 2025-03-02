# Titanic Survival Prediction
## Just Experimenting a bit
This is a simple machine learning project to predict Titanic survival using the Titanic dataset. The project uses **XGBoost** for classification and includes feature engineering, data preprocessing, and hyperparameter tuning to improve model performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)

## Overview

This project explores machine learning techniques to predict passenger survival on the Titanic. The model is trained using various features, including passenger class, age, sex, family size, and others, to predict whether a passenger survived.

## Dataset

The Titanic dataset contains the following columns:

- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Age`: Age of the passenger
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Fare`: Fare paid by the passenger
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- `Sex`: Gender of the passenger
- `Survived`: Target variable (1 if survived, 0 if not)

## Model

The model is built using **XGBoost** (`XGBClassifier`), which is an efficient and scalable implementation of gradient boosting. The following steps were performed:

1. Data preprocessing:
   - Missing values were handled.
   - Categorical features were encoded using OneHotEncoder.
   - New features were engineered (e.g., `FamilySize` and `IsAlone`).

2. Hyperparameter tuning was done using **RandomizedSearchCV** to improve model performance.

## Results

The performance of the model was evaluated using precision, recall, F1-score, and support. Below is the classification report for the model:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.91      | 0.83   | 0.87     | 105     |
| 1     | 0.78      | 0.88   | 0.83     | 74      |
| **Accuracy** |           |        | **0.85** | **179** |
| **Macro avg** | 0.84      | 0.85   | 0.85     | 179     |
| **Weighted avg** | 0.86   | 0.85   | 0.85     | 179     |


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

