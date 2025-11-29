import pandas as pd

def load_gse183019(path):
    raw = pd.read_csv(path, header=None)

    # row parsing
    sample_ids = raw.iloc[0, 1:].tolist()   # sample names
    labels_raw = raw.iloc[1, 1:].tolist()   # tumor/normal string labels
    gene_names = raw.iloc[3:, 0]            # gene symbols
    expr = raw.iloc[3:, 1:]                 # TPM values

    # convert expression to numeric
    expr.columns = sample_ids
    expr.index = gene_names
    expr = expr.apply(pd.to_numeric, errors='coerce')

    # transpose so rows = samples
    X = expr.T

    # create label Series with matching indices
    y = pd.Series(labels_raw, index=sample_ids, name="label")

    # convert to numeric labels
    y = y.map({"Normal": 0, "Tumor": 1}).astype(int)
    X = X.loc[y.index]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label counts:")
    print(y.value_counts())
    print("\nAlignment check:", (X.index == y.index).all())

    return X, y

X, y = load_gse183019("../data/GSE183019_TPM_47genes_clean.csv")

if __name__ == "__main__":
    X, y = load_gse183019("../data/GSE183019_TPM_47genes_clean.csv")
