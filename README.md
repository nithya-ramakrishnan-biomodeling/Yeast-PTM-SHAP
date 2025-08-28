# Yeast-epigenetics
Histone Modifications & Gene Expression Classification
This repository contains code for analyzing yeast gene expression based on histone modification data. The steps are data preprocessing, model training, feature importance with SHAP, and visualization (lollipop, waterfall, and feature importance plots).

#Data
final_matrix.rda - Histone modification signals (matrix).

steady_state_headers.rda - Names of 26 histone modifications.

genelist.rda - List of yeast gene identifiers.

yeast_genes_trans_rates.csv - Transcription rates for yeast genes.

Transrate.csv - Reference dataset for transcription rates with gene mapping.

my_list.csv & output.csv - Mapping files for resolving missing genes.

REQUIRED LIBRARIES --
pip install numpy pandas pyreadr shap matplotlib seaborn scikit-learn imbalanced-learn xgboost plotly


#Analysis.py
1. Data Loading & Preprocessing

load_data()
Loads input .rda and .csv datasets and prepares matrices and gene names.

map_labels(names, value_dict, mapper, miller)
Maps yeast gene names to transcription rate values using multiple sources.

2. Model Training with Grid Search

allMLmodels()

Trains XGBoost, RandomForest, SVM, Logistic Regression.

Uses KMeansSMOTE for balancing classes.

Performs GridSearchCV hyperparameter optimization.

Evaluates models using F1 score.

Saves results to model_f1_scores.csv.

Produces line plots comparing models across histone modifications.

3. SHAP Waterfall Plots

Waterfallplots()

Trains an XGBoost classifier.

Computes SHAP values for test set.

Separates True Positives (TP) and True Negatives (TN).

Prepares waterfall plots to show feature contributions per gene.

Stores SHAP values per modification.
