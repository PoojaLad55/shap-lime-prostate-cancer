import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# file paths
data_dir = "../data"
clean_file = os.path.join(data_dir, "GSE183019_TPM_47genes_clean.csv")

# load data
df = pd.read_csv(clean_file, header=[0,1], index_col=0)
print(f"Loaded data: {df.shape}")

# separate sample IDs and labels
sample_ids = df.columns.get_level_values(0)
labels = df.columns.get_level_values(1)

# quick label summary
label_counts = pd.Series(labels).value_counts()
print("\nLabel distribution:")
print(label_counts)

# visualize class balance
sns.countplot(x=labels)
plt.title("Sample Type Distribution")
plt.xlabel("Sample Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# look at expression distribution
plt.figure(figsize=(10,5))
numeric_df = df.T.apply(pd.to_numeric, errors="coerce")  # convert all to floats, ignore bad ones
sns.boxplot(data=numeric_df, orient='h', fliersize=0.5)
plt.title("Expression Value Distribution (47 genes)")
plt.xlabel("TPM (log scale not applied)")
plt.tight_layout()
plt.show()
