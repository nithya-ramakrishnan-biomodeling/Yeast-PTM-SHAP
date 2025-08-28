import os
import random
import numpy as np
import pandas as pd
import pyreadr as pr
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import KMeansSMOTE
from xgboost import XGBClassifier
import xgboost
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative

pio.renderers.default = "browser"
random.seed(42)


# ----------------------------------------------------------------
# Data loading functions
# ----------------------------------------------------------------
def load_data():
    """Load all necessary data files."""
    data_dict = pr.read_r(r"/content/drive/MyDrive/project/final_matrix.rda")
    data = np.array(data_dict[None])

    steady_state = pr.read_r(r"/content/drive/MyDrive/project/steady_state_headers.rda")
    steady_state_columns = list(steady_state['steady_state_headers']['steady_state_headers'])

    names_dict = pr.read_r(r"/content/drive/MyDrive/project/genelist.rda")
    names = np.array(names_dict['genelist1']).flatten()

    df_rates = pd.read_csv(r"/content/drive/MyDrive/project/yeast_genes_trans_rates.csv")
    df_rates = df_rates.drop(df_rates.columns[[1, 3, 4]], axis=1)

    dataset = pd.read_csv(r"/content/drive/MyDrive/project/Transrate.csv")
    dataset['Names'] = dataset['Value'].combine_first(dataset['Gene'])
    dataset['Transcription Rates mRNA/hr_number'] = pd.to_numeric(
        dataset['Transcription Rates mRNA/hr'], errors='coerce'
    ).fillna(0)
    value_dict = dict(zip(dataset['Names'], dataset['Transcription Rates mRNA/hr_number']))

    mapper = pd.read_csv(r"/content/drive/MyDrive/project/my_list.csv")
    miller = pd.read_csv(r"/content/drive/MyDrive/project/output.csv")

    return data, steady_state_columns, names, value_dict, mapper, miller


def map_labels(names, value_dict, mapper, miller):
    """Map gene names to transcription rate values."""
    labels, notfound = [], []

    for idx, gene in enumerate(names):
        try:
            labels.append(value_dict[gene])
        except KeyError:
            try:
                acc = mapper.loc[mapper['Value'] == gene, 'Key'].values[0]
                labels.append(value_dict[acc])
            except Exception:
                try:
                    acc = mapper.loc[mapper['Key'] == gene, 'Value'].values[0]
                    labels.append(value_dict[acc])
                except Exception:
                    val = miller.loc[(miller['Gene'] == gene) | (miller['wt'] == gene), '00-06']
                    if not val.empty:
                        labels.append(val.values[0])
                    else:
                        notfound.append(idx)

    return np.array(labels), notfound


# ----------------------------------------------------------------
# ML Models with GridSearch
# ----------------------------------------------------------------
def allMLmodels():
    data, steady_state_columns, names, value_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, value_dict, mapper, miller)

    bins = [13]
    binned = np.digitize(labels, bins, right=False)
    cleaned_data = np.delete(data, notfound, axis=1)

    print("Labels:", len(labels), " | Not found:", len(notfound))

    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    }

    models = {
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVC": SVC(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {name: [] for name in models}
    best_params_per_ptm = {name: [] for name in models}

    for i in range(len(cleaned_data)):
        dataiter = [sublist[1:] for sublist in cleaned_data[i]]
        dataiter = np.nan_to_num(dataiter, nan=0.0)

        sm = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=100)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = sm.fit_resample(xtr, ytr)

        for name, model in models.items():
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring="f1",
                cv=5,
                n_jobs=-1,
            )
            grid.fit(xtr_eq, ytr_eq)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(xte)
            f1 = f1_score(yte, y_pred)

            results[name].append(f1)
            best_params_per_ptm[name].append(grid.best_params_)
            print(f"Iteration {i}, {name} best params: {grid.best_params_}, F1: {f1:.4f}")

    # Save results
    csv_results = [
        {"PTM": ptm, "Model": model, "F1": f1}
        for model, f1_scores in results.items()
        for ptm, f1 in zip(steady_state_columns, f1_scores)
    ]

    results_df = pd.DataFrame(csv_results)
    results_df.to_csv("model_f1_scores.csv", index=False)
    print("Results saved to model_f1_scores.csv")

    # Plot
    plt.figure(figsize=(10, 6))
    for name, f1_scores in results.items():
        plt.plot(steady_state_columns, f1_scores, marker="o", label=name)

    plt.xticks(rotation=90)
    plt.ylabel("F1 Score")
    plt.xlabel("PTM")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------
# Waterfall plots for gene specific contibution
# ----------------------------------------------------------------
def Waterfallplots():
    original_features = ['-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8']

    tn, fp, fn, tp = [], [], [], []
    acc, f1sc = [], []
    test0, test1, pred0, pred1 = [], [], [], []
    shap_values_all = []
    data, steady_state_columns, names, value_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, value_dict, mapper, miller)

    bins = [13]
    binned = np.digitize(labels, bins, right=False)
    cleaned_data = np.delete(data, notfound, axis=1)

    print("Labels:", len(labels), " | Not found:", len(notfound))

    for i in range(len(cleaned_data)):
        dataiter = [sublist[1:] for sublist in cleaned_data[i]]
        dataiter = np.nan_to_num(dataiter, nan=0.0)
        sm = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=100)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = sm.fit_resample(xtr, ytr)

        model = xgboost.XGBClassifier(random_state=42)
        model.fit(xtr_eq, ytr_eq)
        y_pred = model.predict(xte)
        #
        # # Metrics
        true = Counter(yte)
        pred = Counter(y_pred)
        test0.append(true[0])
        test1.append(true[1])
        pred0.append(pred[0])
        pred1.append(pred[1])
        t0, f1_, f0, t1 = confusion_matrix(yte, y_pred).ravel()
        tn.append(t0)
        fp.append(f1_)
        tp.append(t1)
        fn.append(f0)
        # f1sc.append(round(f1_score(yte, y_pred), 2))
        # acc.append(round(accuracy_score(yte, y_pred), 2))
        # print(classification_report(yte,y_pred))
        # SHAP
        explainer = shap.Explainer(model, xtr_eq)
        shap_values = explainer(xte)
        print(shap_values)
        # Full SHAP values for test set
        shap_array = shap_values.values

        # Identify TP and TN indices
        tp_indexes = [idx for idx, (true, pred) in enumerate(zip(yte, y_pred)) if true == 1 and pred == 1]
        tn_indexes = [idx for idx, (true, pred) in enumerate(zip(yte, y_pred)) if true == 0 and pred == 0]

        # Combine TP and TN indices to get all correct points
        tptn_indexes = tp_indexes + tn_indexes
        shap_filtered = shap_array[tptn_indexes]

        # Create filtered SHAP dataframe
        shap_df = pd.DataFrame(shap_filtered, columns=original_features)
        shap_df['mod'] = steady_state_columns[i]
        shap_df['sample_idx'] = tptn_indexes
        shap_values_all.append(shap_df)

        # Waterfall plots
        explainer = shap.Explainer(model, xtr_eq)
        shap_values = explainer(xte)
        xte_df = pd.DataFrame(xte, columns=original_features)

        # Map cleaned gene names
        valid_names = np.delete(names, notfound)
        gene_list = list(valid_names)

        # Get the gene names in this test split
        _, xte_names = train_test_split(gene_list, shuffle=True)  # align with yte
        xte_names = list(xte_names)

        # Identify TP and TN indexes and gene names
        # tp_indexes = [idx for idx, (true, pred) in enumerate(zip(yte, y_pred)) if true == 1 and pred == 1]
        # tn_indexes = [idx for idx, (true, pred) in enumerate(zip(yte, y_pred)) if true == 0 and pred == 0]
        # tp_genes = [xte_names[idx] for idx in tp_indexes]
        # tn_genes = [xte_names[idx] for idx in tn_indexes]
        #
        # # Define subfolders
        # mod_name = steady_state_columns[i].replace("/", "_").replace("\\", "_")
        # tp_folder = os.path.join(waterfall_dir, mod_name, "True_Positive")
        # tn_folder = os.path.join(waterfall_dir, mod_name, "True_Negative")
        # os.makedirs(tp_folder, exist_ok=True)
        # os.makedirs(tn_folder, exist_ok=True)

        # Save TP plots
        # for idx, gene in zip(tp_indexes, tp_genes):
        #     plt.figure()
        #     shap.plots.waterfall(shap_values[idx], show=False)
        #     plt.title(f"{mod_name} - True Positive - {gene}")
        #     plt.savefig(os.path.join(tp_folder, f"{gene}.png"))
        #     plt.close()
        #
        # # Save TN plots
        # for idx, gene in zip(tn_indexes, tn_genes):
        #     plt.figure()
        #     shap.plots.waterfall(shap_values[idx], show=False)
        #     plt.title(f"{mod_name} - True Negative - {gene}")
        #     plt.savefig(os.path.join(tn_folder, f"{gene}.png"))
        #     plt.close()
    # Save metrics
    # a = pd.DataFrame({
    #     'mods': steady_state_columns, 'f1': f1sc, 'acc': acc,
    #     'TRUE 1': test1, 'PRED 1': pred1, 'TRUE 0': test0, 'PRED 0': pred0,
    #     'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    # })
    # print(a.head())
    # a.to_csv(r"C:\Edu\project\THRESH-13 final.csv", index=False)

    # Save all SHAP values
    # all_shap_df = pd.concat(shap_values_all, ignore_index=True)
    # all_shap_df.to_csv(r"C:\Edu\project\THRESH-13-final.csv", index=False)
    # #
    # # # Load SHAP values for plotting and ranking
    # shap_df = pd.read_csv(r"C:\Edu\project\THRESH-13-final.csv")
    # feature_cols = original_features

    # # Collect top 3 features per modification into a list
    # top3_records = []

    # for mod, group in shap_df.groupby("mod"):
    #     mod_df = group[feature_cols]
    #     mean_abs_shap = mod_df.abs().mean().sort_values(ascending=False)
    #     top_features = mean_abs_shap.head(3)

    #     # Store in list
    #     top3_records.append({
    #         'modification': mod,
    #         'top1_feature': top_features.index[0],
    #         'top1_value': top_features.iloc[0],
    #         'top2_feature': top_features.index[1],
    #         'top2_value': top_features.iloc[1],
    #         'top3_feature': top_features.index[2],
    #         'top3_value': top_features.iloc[2]
    #     })

    #     # Plot bar chart
    #     # plt.figure(figsize=(8, 6))
    #     # top_features[::-1].plot(kind='barh', color='darkblue', edgecolor='black')
    #     # plt.xlabel("Mean |SHAP value|")
    #     # plt.ylabel("Feature")
    #     # plt.title(f"SHAP Features - {mod}")
    #     # plt.tight_layout()
    #     # safe_mod = mod.replace("/", "_").replace("\\", "_")
    #     # plt.savefig(os.path.join(shap_plot_dir, f"{safe_mod}_SHAP_top3.png"))
    #     # plt.close()

    # # Convert to DataFrame and save
    # top3_df = pd.DataFrame(top3_records)
    # #top3_df.to_csv(r"C:\Edu\project\projecttop3_features_per_modification.csv", index=False)

# ----------------------------------------------------------------
# Lollipop plots for overall contribution from each gene
# ----------------------------------------------------------------
def lollipop_plots():
    original_features = ['-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8']

    data, steady_state_columns, names, value_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, value_dict, mapper, miller)

    bins = [13]
    tp_pos = []
    tn_pos = []
    req_pos = []
    binned = np.digitize(labels, bins, right=False)
    cleaned_data = np.delete(data, notfound, axis=1)

    print("Labels:", len(labels), " | Not found:", len(notfound))
    for i in range(len(cleaned_data)):
        dataiter = [sublist[1:] for sublist in cleaned_data[i]]
        dataiter = np.nan_to_num(dataiter, nan=0.0)

        sm = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=100)
        xtr, xte, ytr, yte = train_test_split(dataiter, binned, shuffle=True, random_state=100)
        xtr_eq, ytr_eq = sm.fit_resample(xtr, ytr)

        model = xgboost.XGBClassifier(random_state=42)
        model.fit(xtr_eq, ytr_eq)
        y_pred = model.predict(xte)
        explainer = shap.Explainer(model, xtr_eq)
        tp_indices = np.where((yte == 1) & (y_pred == 1))[0]
        tn_indices = np.where((yte == 0) & (y_pred == 0))[0]
        req_indices = np.concatenate([tp_indices, tn_indices])
        shap_values_pos = explainer(xte[tp_indices])
        shap_values_neg = explainer(xte[tn_indices])
        shap_values_req = explainer(xte[req_indices])
        distreq = []
        for j in shap_values_req.values:
            max = np.argmax(abs(j))
            distreq.append(original_features[max])
        print("Correct indices calculated")
        distpos = []
        for j in shap_values_pos.values:
            max = np.argmax(abs(j))
            distpos.append(original_features[max])
        countpos = Counter(distpos)
        distneg = []
        print("true Pos val calculated")
        for j in shap_values_neg.values:
            max = np.argmax(abs(j))
            distneg.append(original_features[max])
        countneg = Counter(distneg)
        print("true Neg values calculated")
        fre_req = [countpos.get(val, 0) / len(shap_values_req.values) for val in original_features]
        fre_pos = [countpos.get(val, 0) / len(shap_values_pos.values) for val in original_features]
        fre_neg = [countneg.get(val, 0) / len(shap_values_neg.values) for val in original_features]
        print("ALL freq calcualted")
        tp_pos.append(fre_pos)
        tn_pos.append(fre_neg)
        req_pos.append(fre_req)
    data = {'position': original_features}
    for col_name, values_list in zip(steady_state_columns, tp_pos):
        data[col_name] = values_list
    df = pd.DataFrame(data)
    df.to_csv(r'/content/drive/MyDrive/project/TP.csv')
    for col_name, values_list in zip(steady_state_columns, tn_pos):
        data[col_name] = values_list
    df = pd.DataFrame(data)
    df.to_csv(r'/content/drive/MyDrive/project/TN.csv')
    for col_name, values_list in zip(steady_state_columns, req_pos):
        data[col_name] = values_list
    df = pd.DataFrame(data)
    df.to_csv(r'/content/drive/MyDrive/project/Correctindices.csv')
    from plotly.subplots import make_subplots  # Import for creating multiple subplots if needed
    steady_state = pr.read_r(r"/content/drive/MyDrive/project/steady_state_headers.rda")
    steady_state_columns = list(steady_state['steady_state_headers']['steady_state_headers'])
    genebody = ['H3K4me', 'H4K20me', 'H3K79me3', 'H3K36me3', 'H4R3me2s', 'H4R3me', 'H3K36me', 'H3K36me2', 'H4K16ac',
                'H3S10ph', 'H2AS129ph']
    promotor = ['H2AK5ac', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K27ac', 'H3K4ac', 'H3K4me2', 'H3K4me3', 'H3K56ac',
                'H3K79me', 'H3K9ac', 'H4K12ac', 'H4K5ac', 'H4K8ac', 'Htz1']
    # 1. Load your data (according to TN and TP samples )
    try:
        df = pd.read_csv(
            r'/content/drive/MyDrive/project/TP.csv')  # Adjust based on data points needed (TN.csv and Correctindices)
    except FileNotFoundError:
        print("CSV not found at the specified path")
    correct_position_order = ['-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8']

    df_melted = df.melt(id_vars=['position'],
                        value_vars=steady_state_columns,
                        var_name='MOD_Type',
                        value_name='MODS_Value')

    df_melted['position'] = pd.Categorical(df_melted['position'], categories=df['position'].unique(), ordered=True)

    from plotly.colors import qualitative

    colors = (
            qualitative.Dark24 +
            [qualitative.Set3[0], qualitative.Set3[1]]  # Add 2 more to reach 26
    )

    mod_types = sorted(df_melted['MOD_Type'].unique())
    if len(mod_types) > len(colors):
        raise ValueError("Not enough unique colors for all MOD_Type values!")

    color_map = {mod: colors[i] for i, mod in enumerate(mod_types)}
    mod_types = df_melted['MOD_Type'].unique()
    color_map = {mod: colors[i % len(colors)] for i, mod in enumerate(mod_types)}

    fig = go.Figure()

    # Loop through each MOD_Type to create lollipop traces
    for mod_type in mod_types:
        df_subset = df_melted[df_melted['MOD_Type'] == mod_type]
        current_color = color_map[mod_type]

        fig.add_trace(go.Scatter(
            x=df_subset['position'],
            y=[0] * len(df_subset),
            mode='lines',
            line=dict(color=current_color, width=2),
            showlegend=False,
            name=mod_type,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=df_subset['position'],
            y=df_subset['MODS_Value'],
            mode='markers',
            marker=dict(symbol='circle', size=10, color=current_color, line=dict(width=1, color='DarkSlateGrey')),
            name=mod_type,
            hoverinfo='x+y+name',
            hovertemplate="<b>%{y:.2f}</b><br>Position: %{x}<br>Type: %{customdata}<extra></extra>",
            customdata=df_subset['MOD_Type']
        ))

    # Adjust Layout
    fig.update_layout(
        title='Top SHAP Position Frequency   - Lollipop Plot',
        title_font_size=20,
        xaxis_title='Nucleosome Position',
        yaxis_title='Probability',
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        yaxis_range=[0, df_melted['MODS_Value'].max() * 0.99],
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey',
        showlegend=True,
        legend_title_text='Modification Type'
    )

    fig.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=correct_position_order,
        ticktext=correct_position_order,
        tickangle=-45,
        tickfont=dict(size=20),
        categoryorder='array',
        categoryarray=correct_position_order
    )

    fig.update_yaxes(
        tickfont=dict(size=15)
    )

    fig.add_annotation(x='3', y=df_melted['MODS_Value'].max() * 0.05,  # Position the annotation dynamically
                       text="TSS", textangle=-90, font=dict(size=20, color="black", style='italic'),
                       showarrow=False)

    fig.write_html(r"C:\Edu\project\26.html")

    fig.show()

# ----------------------------------------------------------------
# Combining all PTMs
# ----------------------------------------------------------------
def all_26mods_combined():
    original_features = ['-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8']

    data, steady_state_columns, names, value_dict, mapper, miller = load_data()
    labels, notfound = map_labels(names, value_dict, mapper, miller)

    bins = [13]
    tp_pos = []
    tn_pos = []
    req_pos = []
    binned = np.digitize(labels, bins, right=False)
    cleaned_data = np.delete(data, notfound, axis=1)
    label = np.array(labels)
    bins = [13]
    binned = np.digitize(label, bins, right=False)
    data = np.delete(data, notfound, axis=1)
    names = np.delete(names, notfound)
    data = np.array([[sub_list[1:] for sub_list in inner_list] for inner_list in data])
    data_t = data.transpose(1, 0, 2)
    data = data_t.reshape(data.shape[1], 26 * 11)
    print(data.shape)
    data = np.nan_to_num(data, nan=0)
    sm = KMeansSMOTE(cluster_balance_threshold=0.001, random_state=100)
    xtr, xte, ytr, yte = train_test_split(data, binned, shuffle=True, random_state=100)
    xtr_eq, ytr_eq = sm.fit_resample(xtr, ytr)

    model = xgboost.XGBClassifier(random_state=42)
    model.fit(xtr_eq, ytr_eq)
    y_pred = model.predict(xte)
    print(f1_score(yte, y_pred))
    print(classification_report(yte, y_pred))
    explainer = shap.Explainer(model, xtr_eq)
    shap_values = explainer(xte)
    original_features = ['-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8']
    shap_values.feature_names = [f"MOD-{i} POS {j}" for i in steady_state_columns for j in original_features]
    feature_importance_df = pd.DataFrame({
        'feature': shap_values.feature_names,
        'mean_abs_shap_value': np.abs(shap_values.values).mean(axis=0)
    })

    # Sort and get the top 10
    top_10_features = feature_importance_df.sort_values(
        by='mean_abs_shap_value',
        ascending=False
    ).head(10)
    print(top_10_features)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_10_features['feature'], y=top_10_features['mean_abs_shap_value'])
    plt.xlabel("Modification and Position", size=12)
    plt.ylabel("Mean absolute SHAP value")
    plt.title("Top 10 features")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    # plt.show()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    # allMLmodels()
    # Waterfallplots()
    # lollipop_plots()
    all_26mods_combined()


if __name__ == "__main__":
    main()

