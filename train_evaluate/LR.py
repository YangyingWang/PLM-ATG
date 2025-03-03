import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    recall_score, accuracy_score, precision_score,
    confusion_matrix
)
import joblib

def load_data(path_ne, path_po):
    data_ne = pd.read_csv(path_ne, header=None)
    data_po = pd.read_csv(path_po, header=None)

    data_ne['label'] = 0
    data_po['label'] = 1

    data = pd.concat([data_ne, data_po])

    X = data.iloc[:, 1:-1].values
    y = data['label'].values 

    print("Data loaded successfully!")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")

    return X, y

def train(X_train, y_train):
    params = {
        'C': loguniform(1e-3, 1e3),
    }
    lr = LogisticRegression(random_state=42, max_iter=150)
    random_search = RandomizedSearchCV(lr, params, n_iter=200, cv=5, verbose=3, refit=True)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    y_train_pred = random_search.predict(X_train)
    y_train_pred_proba = random_search.predict_proba(X_train)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    train_specificity = tn / (tn + fp)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

    joblib.dump(best_model, "D:/Major/AIProject/ATG/models/based_fasta/LR_AADP_fasta.pkl")
    with open("D:/Major/AIProject/ATG/performance/LR_results.txt", "a") as f:
        f.write(f"-------------------LR OF AADP_FASTA--------------------\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best cross-validation score: {random_search.best_score_}\n")
        f.write(f"-------------------------------------------------\n")
        f.write(f"Training Accuracy: {train_accuracy}\n")
        f.write(f"Training MCC: {train_mcc}\n")
        f.write(f"Training F1 Score: {train_f1}\n")
        f.write(f"Training Recall: {train_recall}\n")
        f.write(f"Training Precision: {train_precision}\n")
        f.write(f"Training Specificity: {train_specificity}\n")
        f.write(f"Training ROC AUC: {train_roc_auc}\n")
        f.write(f"-------------------------------------------------\n")
    
    return best_model

def test(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_specificity = tn / (tn + fp)
    test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)

    with open("D:/Major/AIProject/ATG/performance/LR_results.txt", "a") as f:
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Test MCC: {test_mcc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(f"Test Recall: {test_recall}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"Test Specificity: {test_specificity}\n")
        f.write(f"Test ROC AUC: {test_roc_auc}\n")
        f.write(f"-------------------------------------------------\n\n")

ne_train = 'D:/Major/AIProject/ATG/data/fastadata/aadp_ne_train.csv'
po_train = 'D:/Major/AIProject/ATG/data/fastadata/aadp_po_train.csv'
ne_test = 'D:/Major/AIProject/ATG/data/fastadata/aadp_ne_test.csv'
po_test = 'D:/Major/AIProject/ATG/data/fastadata/aadp_po_test.csv'

print("Autophagy Training with 5-Fold Cross Validation...")
X_train, y_train = load_data(ne_train, po_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = train(X_train, y_train)

print("Autophagy Testing...")
X_test, y_test = load_data(ne_test, po_test)
X_test = scaler.transform(X_test)
test(model, X_test, y_test)
print("test end...")
