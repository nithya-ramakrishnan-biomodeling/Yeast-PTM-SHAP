import os
import random
import numpy as np
import pandas as pd
import pyreadr as pr
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import plotly.graph_objects as go
import plotly.io as pio

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import KMeansSMOTE
from xgboost import XGBClassifier
from plotly.colors import qualitative

# Configuration
pio.renderers.default = "browser"
random.seed(42)
ORIGINAL_FEATURES = ['-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8']

def load_data():
    """
    Loads genomic data from .rda and .csv files.
    Returns: (data, ptm_names, gene_names, value_dict, mapper, miller)
    """
    data_dict = pr.read_r(r"/content/drive/MyDrive/project/final_matrix.rda")
    data = np.array(data_dict[None])

    steady_state = pr.read_r(r"/content/drive/MyDrive/project/steady_state_headers.rda")
    ptm_names = list(steady_state['steady_state_headers']['steady_state_headers'])

    names_dict = pr.read_r(r"/content/drive/MyDrive/project/genelist.rda")
    gene_names = np.array(names_dict['genelist1']).flatten()

    dataset = pd.read_csv(r"/content/drive/MyDrive/project/Transrate.csv")
    dataset['Names'] = dataset['Value'].combine_first(dataset['Gene'])
    dataset['Rates_num'] = pd.to_numeric(dataset['Transcription Rates mRNA/hr'], errors='coerce').fillna(0)
    value_dict = dict(zip(dataset['Names'], dataset['Rates_num']))

    mapper = pd.read_csv(r"/content/drive/MyDrive/project/my_list.csv")
    miller = pd.read_csv(r"/content/drive/MyDrive/project/output.csv")

    return data, ptm_names, gene_names, value_dict, mapper, miller

def map_labels(names, value_dict, mapper, miller):
    """Maps gene names to transcription rates using primary and fallback datasets."""
    labels, notfound = [], []
    for idx, gene in enumerate(names):
        try:
            labels.append(value_dict[gene])
        except KeyError:
            try:
                acc = mapper.loc[mapper['Value'] == gene, 'Key'].values[0]
                labels.append(value_dict[acc])
            except:
                try:
                    acc = mapper.loc[mapper['Key'] == gene, 'Value'].values[0]
                    labels.append(value_dict[acc])
                except:
                    val = miller.loc[(miller['Gene'] == gene) | (miller['wt'] == gene), '00-06']
                    if not val.empty:
                        labels.append(val.values[0])
                    else:
                        notfound.append(idx)
    return np.array(labels), notfound

def run_all_ml_models():
    """Performs GridSearch for multiple models across all PTMs and saves F1 scores."""
    data, ptm_names, names, v_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, v_dict, mapper, miller)
    
    binned = np.digitize(labels, [13], right=False)
    cleaned_data = np.delete(data, notfound, axis=1)

    param_grids = {
        "XGBoost": {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.1]},
        "RandomForest": {"n_estimators": [100], "max_depth": [10]},
        "SVC": {"C": [1, 10], "kernel": ["rbf"]},
        "LogisticRegression": {"C": [1], "solver": ["lbfgs"]}
    }

    models = {
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVC": SVC(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results_list = []
    for i in range(len(cleaned_data)):
        dataiter = np.nan_to_num([sub[1:] for sub in cleaned_data[i]], nan=0.0)
        sm = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=100)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = sm.fit_resample(xtr, ytr)

        for name, model in models.items():
            grid = GridSearchCV(model, param_grids[name], scoring="f1", cv=3, n_jobs=-1)
            grid.fit(xtr_eq, ytr_eq)
            f1 = f1_score(yte, grid.best_estimator_.predict(xte))
            results_list.append({"PTM": ptm_names[i], "Model": name, "F1": f1})
            print(f"PTM: {ptm_names[i]} | {name} F1: {f1:.4f}")

    pd.DataFrame(results_list).to_csv("model_f1_scores.csv", index=False)

def run_waterfall_analysis():
    """Generates SHAP data for correctly predicted samples (TP and TN)."""
    data, ptm_names, names, v_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, v_dict, mapper, miller)
    binned = np.digitize(labels, [13], right=False)
    cleaned_data = np.delete(data, notfound, axis=1)
    
    shap_all = []
    for i in range(len(cleaned_data)):
        dataiter = np.nan_to_num([sub[1:] for sub in cleaned_data[i]], nan=0.0)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = KMeansSMOTE(random_state=100).fit_resample(xtr, ytr)

        model = XGBClassifier(random_state=42).fit(xtr_eq, ytr_eq)
        y_pred = model.predict(xte)
        
        explainer = shap.Explainer(model, xtr_eq)
        shap_values = explainer(xte)

        correct_idx = [idx for idx, (t, p) in enumerate(zip(yte, y_pred)) if t == p]
        df = pd.DataFrame(shap_values.values[correct_idx], columns=ORIGINAL_FEATURES)
        df['mod'] = ptm_names[i]
        shap_all.append(df)
        
    pd.concat(shap_all).to_csv("filtered_shap_values.csv", index=False)

def run_lollipop_plots():
    """Calculates top SHAP position frequencies and generates Plotly lollipop visualization."""
    data, ptm_names, names, v_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, v_dict, mapper, miller)
    binned = np.digitize(labels, [13], right=False)
    cleaned_data = np.delete(data, notfound, axis=1)

    tp_freqs = []
    for i in range(len(cleaned_data)):
        dataiter = np.nan_to_num([sub[1:] for sub in cleaned_data[i]], nan=0.0)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = KMeansSMOTE(random_state=100).fit_resample(xtr, ytr)

        model = XGBClassifier(random_state=42).fit(xtr_eq, ytr_eq)
        y_pred = model.predict(xte)
        
        tp_idx = np.where((yte == 1) & (y_pred == 1))[0]
        if len(tp_idx) > 0:
            sv = shap.Explainer(model, xtr_eq)(xte[tp_idx]).values
            max_pos = [ORIGINAL_FEATURES[np.argmax(abs(row))] for row in sv]
            counts = Counter(max_pos)
            tp_freqs.append([counts.get(p, 0) / len(tp_idx) for p in ORIGINAL_FEATURES])
        else:
            tp_freqs.append([0] * len(ORIGINAL_FEATURES))

    # Plotting
    fig = go.Figure()
    colors = (qualitative.Dark24 + qualitative.Set3)
    for i, mod in enumerate(ptm_names):
        fig.add_trace(go.Scatter(x=ORIGINAL_FEATURES, y=tp_freqs[i], mode='markers+lines', 
                                 name=mod, marker=dict(size=8, color=colors[i % len(colors)])))
    
    fig.update_layout(title="Top SHAP Position Frequency (TP)", xaxis_title="Position", yaxis_title="Probability")
    fig.show()

def run_combined_26_mods():
    """Flattens all 26 PTMs into a single matrix to find global feature importance."""
    data, ptm_names, names, v_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, v_dict, mapper, miller)
    binned = np.digitize(labels, [13], right=False)
    
    clean_data = np.delete(data, notfound, axis=1)
    # Shape transformation: (PTM, Gene, Pos) -> (Gene, PTM * Pos)
    reshaped = clean_data[:, :, 1:].transpose(1, 0, 2).reshape(clean_data.shape[1], 26 * 11)
    reshaped = np.nan_to_num(reshaped, nan=0.0)

    xtr, xte, ytr, yte = train_test_split(reshaped, binned, shuffle=True, random_state=100)
    xtr_eq, ytr_eq = KMeansSMOTE(random_state=100).fit_resample(xtr, ytr)

    model = XGBClassifier(random_state=42).fit(xtr_eq, ytr_eq)
    explainer = shap.Explainer(model, xtr_eq)
    shap_values = explainer(xte)

    f_names = [f"{m} P{p}" for m in ptm_names for p in ORIGINAL_FEATURES]
    importance = pd.DataFrame({'feature': f_names, 'val': np.abs(shap_values.values).mean(axis=0)})
    top_10 = importance.sort_values(by='val', ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_10, x='feature', y='val')
    plt.xticks(rotation=45)
    plt.title("Top 10 Global Features (All PTMs Combined)")
    plt.show()

def main():
    print("Starting ML Comparison...")
    run_all_ml_models()
    print("Running SHAP Analysis...")
    run_waterfall_analysis()
    print("Generating Plots...")
    run_lollipop_plots()
    run_combined_26_mods()

if __name__ == "__main__":
    main()
