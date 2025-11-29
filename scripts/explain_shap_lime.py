import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import spearmanr
from load_gse183019 import X, y
import shap
from lime.lime_tabular import LimeTabularExplainer

matplotlib.use("Agg")
RANDOM_STATE = 42

# basic info
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("label counts:\n", y.value_counts())
print("alignment:", X.shape[0] == y.shape[0])

feature_names = list(X.columns)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# impute
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# model
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train_imp, y_train)

print("\nEval:")
print(classification_report(y_test, rf.predict(X_test_imp)))

# shap
print("\nSHAP")

X_train_df = pd.DataFrame(X_train_imp, columns=feature_names)
explainer = shap.Explainer(rf, X_train_df)
shap_values = explainer(X_train_df).values

mean_abs = np.mean(np.abs(shap_values), axis=0)
mean_abs = mean_abs.sum(axis=1)

shap_importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

print("\nTop SHAP:")
print(shap_importance.head(15))

# shap bar plot
plt.figure(figsize=(8,6))
shap_importance.head(15).plot(kind="bar")
plt.title("Top SHAP Features")
plt.ylabel("Importance Score")
plt.xlabel("Gene")
plt.tight_layout()
plt.savefig("../plots/shap_barplot.png")
plt.close()

# shap summary
shap.summary_plot(shap_values[:,:,1], X_train_df, show=False)
plt.tight_layout()
plt.savefig("../plots/shap_summary_plot.png")
plt.close()

# lime
print("\nLIME")

class_names = ["Normal", "Tumor"]

lime_explainer = LimeTabularExplainer(
    X_train_imp,
    feature_names=feature_names,
    class_names=class_names,
    discretize_continuous=True,
    mode="classification",
    random_state=RANDOM_STATE,
)

N = min(30, X_test_imp.shape[0])
lime_acc = pd.Series(0.0, index=feature_names)

for i in range(N):
    exp = lime_explainer.explain_instance(
        X_test_imp[i],
        rf.predict_proba,
        num_features=len(feature_names)
    )
    for f, w in exp.as_list():
        g = f.split()[0]
        if g in lime_acc.index:
            lime_acc[g] += abs(w)

lime_importance = (lime_acc / N).sort_values(ascending=False)

print("\nTop LIME:")
print(lime_importance.head(10))

# lime barplot
plt.figure(figsize=(8,6))
lime_importance.head(15).plot(kind="bar")
plt.title("Top Lime Features")
plt.ylabel("Importance Score")
plt.xlabel("Gene")
plt.tight_layout()
plt.savefig("../plots/lime_global_barplot.png")
plt.close()

# shap vs lime
shap_al = shap_importance.reindex(feature_names).fillna(0)
lime_al = lime_importance.reindex(feature_names).fillna(0)

rho, p = spearmanr(shap_al.values, lime_al.values)
print(f"\nspearman: rho={rho:.3f} p={p:.3e}")

def jacc(series1, series2, k):
    a = set(series1.head(k).index)
    b = set(series2.head(k).index)
    return len(a & b) / len(a | b)

for k in [5,10,15]:
    print(f"jaccard top-{k}: {jacc(shap_importance, lime_importance, k):.3f}")

# save shap + lime rankings for enrichment
shap_importance.to_csv("../data/shap_importance.csv", header=["score"])
lime_importance.to_csv("../data/lime_importance.csv", header=["score"])
