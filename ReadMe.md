# Decision Tree Classifier with PCA and StandardScaler

This notebook demonstrates how to preprocess data using **StandardScaler** and reduce dimensionality with **Principal Component Analysis (PCA)** before training a **Decision Tree Classifier**. The classifier's performance is evaluated for different numbers of PCA components.

---

## Table of Contents
1. [Project Overview](#Project-Overview)
2. [Dataset Requirements](#Dataset-Requirements)
3. [Methodology](#Methodology)
4. [Dependencies](#Dependencies)
5. [Steps to Use](#Steps-to-Use)
6. [Results](#Results)

---

## Project Overview

### Goals:
- Scale features using `StandardScaler`.
- Apply PCA to reduce data dimensions.
- Train a `DecisionTreeClassifier` for each PCA configuration.
- Evaluate accuracy to determine the optimal number of PCA components.

---

## Dataset Requirements

Ensure the dataset is:
1. Split into:
   - **Features**: `X_train` (training data), `X_test` (testing data)
   - **Target**: `y_train` (training labels), `y_test` (testing labels)
2. Features are numeric, and the target variable is categorical (binary or multi-class).

---

## Methodology

1. **Scaling**:
   - Standardize features to have mean = 0 and standard deviation = 1 using `StandardScaler`.
2. **PCA**:
   - Incrementally reduce dimensions from 1 to the total number of features.
3. **Model Training**:
   - Train a `DecisionTreeClassifier` for each PCA configuration.
4. **Evaluation**:
   - Output the accuracy of each configuration to assess performance.

---

## Dependencies

Install the required libraries before running the notebook:
```python
!pip install pandas numpy scikit-learn
```

## Steps to Use

1. Load your dataset and split it into:
   - `X_train` (training data)
   - `X_test` (testing data)
   - `y_train` (training labels)
   - `y_test` (testing labels)

2. Copy the following code into a cell in your Jupyter Notebook to preprocess the data, apply PCA, and evaluate the Decision Tree Classifier:

   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score
   import pandas as pd

   # Scale the data
   scale = StandardScaler()
   clf3 = ColumnTransformer(
       transformers=[('scale', scale, X_train.columns)],
       remainder='passthrough'
   )
   x_train_scaled = clf3.fit_transform(X_train)
   x_test_scaled = clf3.transform(X_test)

   # Convert scaled data to DataFrame
   x_train_scaled = pd.DataFrame(x_train_scaled, columns=X_train.columns)
   x_test_scaled = pd.DataFrame(x_test_scaled, columns=X_test.columns)

   # Train and evaluate with PCA
   dtc = DecisionTreeClassifier(random_state=42)
   for i in range(1, len(X_train.columns) + 1):
       pca = PCA(n_components=i, random_state=42)
       x_train_pca = pca.fit_transform(x_train_scaled)
       x_test_pca = pca.transform(x_test_scaled)
       dtc.fit(x_train_pca, y_train)
       y_pred = dtc.predict(x_test_pca)
       print(f'Decision Tree Classifier with {i} PCA components: Accuracy = {accuracy_score(y_test, y_pred):.4f}')

## Results
Decision Tree Classifier with 1 PCA components: Accuracy = 0.03
Decision Tree Classifier with 2 PCA components: Accuracy = 0.06
...
Decision Tree Classifier with 16 PCA components: Accuracy = 0.09
