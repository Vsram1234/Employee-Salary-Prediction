import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV  # for probability estimates
import joblib

# ==============================
# Settings
# ==============================
DATA_PATH = "/Users/pallavipaasamgmail.com/Downloads/adult 3.csv"
PLOTS_DIR = "model_graphs"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================
# Step 1: Load dataset
# ==============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
print(f"Initial dataset shape: {data.shape}")

# ==============================
# Step 2: Clean and preprocess
# ==============================
# Replace missing values
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})

# Remove irrelevant rows
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]

# Remove outliers
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data = data[(data['educational-num'] >= 5) & (data['educational-num'] <= 16)]

# Drop redundant column
if 'education' in data.columns:
    data = data.drop(columns=['education'])

# Encode categorical variables
encoder = LabelEncoder()
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

print(f"Cleaned dataset shape: {data.shape}")

# ==============================
# Step 3: Split data
# ==============================
X = data.drop(columns=['income'])
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Step 4: Define models
# ==============================
# Wrap LinearSVC in CalibratedClassifierCV so we can get probability estimates
linear_svc = CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Linear SVM": linear_svc,
    "Gradient Boosting": GradientBoostingClassifier()
}
results = {}

# ==============================
# Helper functions for plotting
# ==============================
def save_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def save_roc_curve(y_true, probas, title, filename):
    fpr, tpr, _ = roc_curve(y_true, probas, pos_label='>50K')
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

# ==============================
# Step 5: Train, evaluate, and plot
# ==============================
pipelines = {}  # to keep each pipeline for later saving
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probas = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    results[name] = acc
    pipelines[name] = pipe
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    save_confusion_matrix(cm, ["<=50K", ">50K"], f"{name} - Confusion Matrix", f"{name}_confusion.png")

    # Save ROC Curve (if available)
    if probas is not None:
        save_roc_curve(y_test, probas, f"{name} - ROC Curve", f"{name}_roc.png")

# ==============================
# Step 6: Save best pipeline (with scaler)
# ==============================
best_model_name = max(results, key=results.get)
best_pipeline = pipelines[best_model_name]  # includes scaler + model
joblib.dump(best_pipeline, "best_model.pkl")
print(f"\nBest model saved: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# ==============================
# Step 7: Save accuracy comparison bar chart
# ==============================
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), results.values(), color='teal')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"))
plt.close()

print(f"\nAll graphs saved in '{PLOTS_DIR}' folder.")
