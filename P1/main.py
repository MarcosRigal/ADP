import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

#------------------------------------------------------------------------------
# Helper function to load and process a dataset from a given collection.
#------------------------------------------------------------------------------
def load_dataset(collection, num_bits=1024):
    docs = list(collection.find({}))
    df_rows = []

    for doc in docs:
        # Create a fingerprint array of 1024 bits initialized to 0.
        fp_row = np.zeros(num_bits, dtype=int)
        
        # Update the array if the document contains a valid fingerprint list.
        if "fingerprint" in doc and doc["fingerprint"]:
            for pos in doc["fingerprint"]:
                if 0 <= pos < num_bits:
                    fp_row[pos] = 1

        # Build a dictionary with "class" and each fingerprint bit.
        row_data = {"class": doc.get("class", 0)}  # default to 0 if "class" is missing
        for i in range(num_bits):
            row_data[f"FP{i+1}"] = fp_row[i]
        df_rows.append(row_data)

    # Convert list of dictionaries to a DataFrame.
    df = pd.DataFrame(df_rows)
    print("Before filtering, DataFrame shape:", df.shape)
    
    # Filter out fingerprint columns that are all zeros.
    filtered_df = df.loc[:, (df != 0).any(axis=0)]
    # If filtering removes all fingerprint columns (leaving only "class"), revert.
    if len(filtered_df.columns) <= 1:
        print("Warning: No non-zero fingerprint columns found; reverting to original DataFrame.")
        filtered_df = df

    print("After filtering, DataFrame shape:", filtered_df.shape)
    return filtered_df

#------------------------------------------------------------------------------
# Helper function to perform grid search for hyperparameter tuning.
#------------------------------------------------------------------------------
def grid_search_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

#------------------------------------------------------------------------------
# Helper function to train and evaluate a model on a DataFrame.
#------------------------------------------------------------------------------
def train_and_evaluate(df):
    # Separate features (X) and target (y)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Print class distribution to assess imbalance.
    print("Class distribution:")
    print(y.value_counts())

    # Split into training and testing sets (using stratification).
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning with GridSearchCV.
    best_rf = grid_search_rf(X_train, y_train)
    
    # Train the tuned classifier on the full training set.
    best_rf.fit(X_train, y_train)

    # Predict on test data.
    y_pred = best_rf.predict(X_test)
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics.
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Plot histogram of predicted probabilities for diagnostics.
    plt.hist(y_pred_prob, bins=20)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.show()
    
    return best_rf, acc, kappa, auc

#------------------------------------------------------------------------------
# 1) Connect to MongoDB.
#------------------------------------------------------------------------------
client = pymongo.MongoClient("mongodb://admin:1234@localhost:27017/")

#------------------------------------------------------------------------------
# 2) Select the CDS16 database and its collections.
#------------------------------------------------------------------------------
db_cds16 = client["CDS16"]
collection_molecules_16 = db_cds16["molecules"]
collection_mfp_counts_16 = db_cds16["mfp_counts"]

print(f"CDS16 - molecules: {collection_molecules_16.count_documents({})} documentos.")
print(f"CDS16 - mfp_counts: {collection_mfp_counts_16.count_documents({})} documentos.")

#------------------------------------------------------------------------------
# 3) Select the CDS29 database and its collections.
#------------------------------------------------------------------------------
db_cds29 = client["CDS29"]
collection_molecules_29 = db_cds29["molecules"]
collection_mfp_counts_29 = db_cds29["mfp_counts"]

print(f"CDS29 - molecules: {collection_molecules_29.count_documents({})} documentos.")
print(f"CDS29 - mfp_counts: {collection_mfp_counts_29.count_documents({})} documentos.")

#------------------------------------------------------------------------------
# 4) Load and process datasets from both CDS16 and CDS29.
#------------------------------------------------------------------------------
df_cds16 = load_dataset(collection_molecules_16)
df_cds29 = load_dataset(collection_molecules_29)

# Ensure that the feature matrix is not empty.
if df_cds16.drop('class', axis=1).empty:
    raise ValueError("Feature matrix for CDS16 is empty. Please check your data ingestion and filtering steps.")
if df_cds29.drop('class', axis=1).empty:
    raise ValueError("Feature matrix for CDS29 is empty. Please check your data ingestion and filtering steps.")

#------------------------------------------------------------------------------
# 5) Train and evaluate the model on both datasets.
#------------------------------------------------------------------------------
rf_cds16, acc_cds16, kappa_cds16, auc_cds16 = train_and_evaluate(df_cds16)
rf_cds29, acc_cds29, kappa_cds29, auc_cds29 = train_and_evaluate(df_cds29)

#------------------------------------------------------------------------------
# 6) Print and compare performance metrics.
#------------------------------------------------------------------------------
comparison_df = pd.DataFrame({
    "Dataset": ["CDS16", "CDS29"],
    "Accuracy": [acc_cds16, acc_cds29],
    "Kappa": [kappa_cds16, kappa_cds29],
    "AUC": [auc_cds16, auc_cds29]
})

print("\nComparison of model performance:")
print(comparison_df)
