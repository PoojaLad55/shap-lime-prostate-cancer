import pandas as pd
import os
import re

# file paths
data_dir = "../data"
expr_file = os.path.join(data_dir, "GSE183019_processed_TPM.txt")
gene_file = os.path.join(data_dir, "47-PCa-Genes.csv")
meta_file = os.path.join(data_dir, "GSE183019_series_matrix.txt")  # metadata
output_file = os.path.join(data_dir, "GSE183019_TPM_47genes_clean.csv")

# load expression data
expr_df = pd.read_csv(expr_file, sep="\t", index_col=0)
print(f"Matrix shape: {expr_df.shape}")

# load 47 genes
genes47 = pd.read_csv(gene_file)
gene_list = genes47["Gene symbol"].tolist()

# standardize index
expr_df.index = expr_df.index.str.replace(r'\.\d+', '', regex=True)
matched = expr_df.index.intersection(gene_list)
print(f"Matched {len(matched)} of 47 genes")

expr47 = expr_df.loc[matched]

# read metadata to extract sample type
sample_labels = {}
with open(meta_file, "r") as f:
    for line in f:
        if line.startswith("!Sample_characteristics_ch1") and "sample type" in line.lower():
            parts = re.findall(r'"([^"]+)"', line)
            for sample, desc in zip(expr47.columns, parts):
                if "cancer" in desc.lower():
                    sample_labels[sample] = "Tumor"
                else:
                    sample_labels[sample] = "Normal"
            break

# verify label counts
print("Label summary:")
print(pd.Series(list(sample_labels.values())).value_counts())

# assign MultiIndex columns
labels = [sample_labels.get(c, "Unknown") for c in expr47.columns]
expr47.columns = pd.MultiIndex.from_arrays([expr47.columns, labels])

# save cleaned data
expr47 = expr47.loc[:, [lab != "Unknown" for lab in labels]] # remove unknowns
expr47.to_csv(output_file)
