# shap-lime-prostate-cancer
Evaluating robustness of SHAP and LIME in Random Forest classifier for prostate cancer gene expression.

# Overview
This project tests whether model interpretability can be validated instead of simply assumed. Instead of relying on SHAP alone, this project directly compares SHAP and LIME explanations generated from the same Random Forest model trained on the GSE183019 prostate cancer RNA-seq dataset. Stability is quantified using Spearman correlation and Jaccard similarity, and overlapping top-ranked genes are mapped to biological pathways. All code is implemented in Python.

# How to Run
## 1. Create and activate virtual environment
`
python3 -m venv venv`
`
source venv/bin/activate # for linux`
`
venv\Scripts\activate`

## 2. Install dependencies
` 
pip install -r requirements.txt
` 
## 3. Train the Random Forest model
This loads the 47-gene csv, imputes missing values, trains the classifier, and outputs evaluation metrics. Outputs include: (1) Confusion matrix plot, (2) Classification metrics, (3) ROC AUC score.

`
python3 train_model.py
`
## 4. Generate SHAP and LIME explanations
Outputs include: (1) shap_importance.csv, (2) lime_importance.csv, (3) Top-genes bar plots.

`
python explain_shap_lime.py
`
## 5. Compare SHAP vs LIME and run pathway enrichment
Outputs include: (1) sSpearman and Jaccard stability scores, (2) Overlapping consensus gene list.csv, (3) KEGG/GO enrichment tables.

`
python pathway_enrichment.py
`






