import numpy as np
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from load_gse183019 import X, y

matplotlib.use("Agg")   
RANDOM_STATE = 42
os.makedirs("../plots", exist_ok=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Label counts:\n", y.value_counts())
print("Alignment check:", X.shape[0] == y.shape[0])

# train + test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,          # 75% train, 25% test
    stratify=y,
    random_state=RANDOM_STATE,
)

# impute missing values
imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

print("\nTOTAL NaNs in training AFTER imputation:", np.isnan(X_train_imp).sum())
print("TOTAL NaNs in test AFTER imputation:", np.isnan(X_test_imp).sum())

# random Forest with class weights

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",   # important for tumor class
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf.fit(X_train_imp, y_train)

# evaluation
y_pred = rf.predict(X_test_imp)
y_proba = rf.predict_proba(X_test_imp)[:, 1]

acc = (y_pred == y_test).mean()
print(f"\nAccuracy: {acc:.3f}\n")

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.3f}")
except ValueError:
    print("ROC AUC could not be computed")

# plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Tumor"],
    yticklabels=["Normal", "Tumor"]
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("../plots/confusion_matrix.png")
plt.close()