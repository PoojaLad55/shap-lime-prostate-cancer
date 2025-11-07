import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# file paths
data_dir = "../data"
clean_file = os.path.join(data_dir, "GSE183019_TPM_47genes_clean.csv")
raw = pd.read_csv(clean_file, header=None)

# fix headers and labels from your CSV structure
df = raw.iloc[3:, :]                     # data starts after the 3rd row
df.columns = raw.iloc[0, :]              # gene symbols as columns
labels = raw.iloc[1, 1:].values          # second row has "Normal"/"Tumor"

# drop non-numeric description column if present
df = df.drop(columns=["DESCRIPTION"], errors='ignore')

# set gene names as index and convert everything to numeric
df = df.set_index(df.columns[0])
df = df.apply(pd.to_numeric, errors='coerce')

# transpose so each row = sample, each column = gene
df = df.T

# create label vector (0 = Normal, 1 = Tumor)
y = pd.Series(labels, name="Label").map({'Normal': 0, 'Tumor': 1}).astype(int)
y = y.iloc[:len(df)]  # align in case of mismatch

print(f"Loaded data: {df.shape}")
print(f"Labels: {y.value_counts().to_dict()}")

# split data
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# evaluate model
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Tumor'],
            yticklabels=['Normal', 'Tumor'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# feature importance
importances = pd.Series(rf.feature_importances_, index=df.columns)
top_genes = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_genes.values, y=top_genes.index, orient='h')
plt.title("Top 10 Most Important Genes (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
