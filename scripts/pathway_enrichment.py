import pandas as pd
from gprofiler import GProfiler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def load_ranking(path):
    df = pd.read_csv(path)

    # detect gene column
    gene_col = None
    for c in df.columns:
        if "gene" in c.lower():
            gene_col = c
            break
        if "Unnamed" in c and df[c].dtype == object:
            gene_col = c
            break

    # detect score column
    score_col = None
    for c in df.columns:
        if "score" in c.lower() or "importance" in c.lower():
            score_col = c
            break

    if gene_col is None or score_col is None:
        print("ERROR reading:", path, "columns:", df.columns.tolist())
        raise SystemExit()

    df = df[[gene_col, score_col]]
    df.columns = ["gene", "score"]
    return df.sort_values("score", ascending=False)

# load shap/lime rankings
shap_df = load_ranking("../data/shap_importance.csv")
lime_df = load_ranking("../data/lime_importance.csv")

# define top lists
TOP_SHAP = shap_df["gene"].head(15).tolist()
TOP_LIME = lime_df["gene"].head(10).tolist()
CONSENSUS = list(set(TOP_SHAP).intersection(TOP_LIME))

print("\ntop shap genes:", TOP_SHAP)
print("top lime genes:", TOP_LIME)
print("consensus genes:", CONSENSUS)

gp = GProfiler(return_dataframe=True)

def make_plot(df, name):
    df["logp"] = -np.log10(df["p_value"])

    top = df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top["name"], top["logp"], color="teal")
    plt.xlabel("-log10(p-value)")
    plt.ylabel("Pathway")
    plt.title(f"Top Enriched Pathways for {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"../plots/{name}_top_pathways.png", dpi=220)
    plt.close()

def enrich(glist, name):
    print(f"\nrunning enrichment for {name} ({len(glist)} genes)")

    res = gp.profile(
        organism="hsapiens",
        query=glist,
        sources=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"]
    )

    if res.empty:
        print("no enriched pathways")
        return None

    print(res[["native", "name", "p_value"]].head(10))

    # save CSV
    csv_path = f"../data/{name}_enrichment.csv"
    res.to_csv(csv_path, index=False)
    print("saved:", csv_path)

    make_plot(res, name)
    return res

# run all enrichments
enrich(TOP_SHAP, "SHAP_top15")
enrich(TOP_LIME, "LIME_top10")
enrich(CONSENSUS, "Consensus")

print("\n=== All enrichment complete ===")
