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

