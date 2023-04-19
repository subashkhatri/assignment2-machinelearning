import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# load data set
data = pd.read_csv("Datasets/D2/healthcare-dataset-stroke-data.csv")
# print(data) checking weather the csv is loaded or not


# Step 1: Loading Data, Data Pre-Processing, EDA
print(data.head())
print(data.info())
print(data.describe())

# handling Missing values

# Handling missing values
data["bmi"].fillna(data["bmi"].mean(), inplace=True)
data["smoking_status"].replace(
    "Unknown", data["smoking_status"].mode()[0], inplace=True
)

# EDA
sns.countplot(x="stroke", data=data)
plt.show()

# Step 2: Feature Engineering, Creating Train, and Test Datasets
data = pd.get_dummies(
    data,
    columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    drop_first=True,
)

X = data.drop("stroke", axis=1)
y = data["stroke"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Apply at least 4 algorithms (Training and Testing)
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}
# Step 4: Generate at least 4 Evaluation Metrics on each algorithm.
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Confusion Matrix: \n{cm}")
    print("\n")


for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)

# Step 5: Comparing the results

# Review the printed evaluation metrics from the previous response's code.
# Based on the evaluation metrics (accuracy, precision, recall, F1 score, and ROC AUC score),
# determine which model performed the best. In this example, we'll assume RandomForestClassifier was the best-performing model.
# Assuming RandomForestClassifier was the best model
best_model = RandomForestClassifier()

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Perform the grid search
grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
evaluate_model("Fine-tuned Random Forest", best_model, X_test, y_test)
# This code snippet assumes that RandomForestClassifier was the best-performing model. It then uses
#  GridSearchCV to fine-tune the hyper parameters of the model. The best parameters and
#   their corresponding score are printed. Finally, the fine-tuned model is evaluated using the same evaluation function used earlier.

# Adjust this code to fine-tune the model you determined to be the best based on the
#  evaluation metrics. If another model performed better in your case, you would need
#  to adjust the parameter grid and use the corresponding estimator.

#The Output generated after the Fine-tuned Random Forest
# Best parameters found:  {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
# Best score found:  0.9547455806172621
# UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# Fine-tuned Random Forest:
# Accuracy: 0.9393346379647749
# Precision: 0.0
# Recall: 0.0
# F1 Score: 0.0
# ROC AUC Score: 0.5
# Confusion Matrix:
# [[960   0]
#  [ 62   0]]
