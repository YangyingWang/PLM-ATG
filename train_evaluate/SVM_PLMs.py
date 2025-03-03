import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    recall_score, accuracy_score, precision_score,
    confusion_matrix
)
import joblib

def load_data(paths):
    X, y = [], []
    
    for path in paths:
        with np.load(path) as data:
            features = [data[key] for key in data]
            X.extend(features)
            if '_ne' in path:
                y.extend([0] * len(features))
            elif '_po' in path:
                y.extend([1] * len(features))
            else:
                print(f"Warning: Unrecognized file pattern in {path}")

    return np.array(X), np.array(y)

def get_data(path):
    data = np.load(path)
    X = data['features']
    y = data['labels']

    return X, y

def train(X_train, y_train):
    params = {
        'C': loguniform(1e-3, 1e3),
        'gamma': loguniform(1e-4, 1e1)
    }
    svm = SVC(probability=True, random_state=42)
    random_search = RandomizedSearchCV(svm, params, n_iter=100, cv=5, verbose=3, refit=True)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    y_train_pred = random_search.predict(X_train)
    y_train_pred_proba = random_search.predict_proba(X_train)[:, 1]

    # param_grid = {
    #     'C': [1, 3, 5, 8, 10],
    #     'gamma': [0.0001, 0.0002, 0.001],
    # }
    # svm = SVC(probability=True, random_state=42)
    # grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=3, refit=True)
    # grid_search.fit(X_train, y_train)
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_
    # y_train_pred = grid_search.predict(X_train) 
    # y_train_pred_proba = grid_search.predict_proba(X_train)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    train_specificity = tn / (tn + fp)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

    joblib.dump(best_model, "D:/Major/AIProject/ATG/models/SVM_AADP_T5.pkl")
    with open("D:/Major/AIProject/ATG/performance/SVM_PLMs_results.txt", "a") as f:
        f.write(f"-------------------SVM OF AADP_T5--------------------\n")
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

    with open("D:/Major/AIProject/ATG/performance/SVM_PLMs_results.txt", "a") as f:
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Test MCC: {test_mcc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(f"Test Recall: {test_recall}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"Test Specificity: {test_specificity}\n")
        f.write(f"Test ROC AUC: {test_roc_auc}\n")
        f.write(f"-------------------------------------------------\n\n")


train_paths = [
    "D:/Major/AIProject/ATG/data/pretrain/Train/t5_ne.npz",
    "D:/Major/AIProject/ATG/data/pretrain/Train/t5_po.npz"
]
test_paths = [
    "D:/Major/AIProject/ATG/data/pretrain/Test/t5_ne.npz",
    "D:/Major/AIProject/ATG/data/pretrain/Test/t5_po.npz"
]

print("Autophagy Training with 5-Fold Cross Validation...")
print("Start load data...")
# X_train, y_train = load_data(train_paths)
# X_train, y_train = get_data("D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz")
X_train, y_train = get_data("D:/Major/AIProject/ATG/data/aadp_t5_train.npz")
print("Load data over!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = train(X_train, y_train)

print("Autophagy Testing...")
print("Start load data...")
# X_test, y_test = load_data(test_paths)
# X_test, y_test = get_data("D:/Major/AIProject/ATG/data/pretrain/Test/t5_esm2.npz")
X_test, y_test = get_data("D:/Major/AIProject/ATG/data/aadp_t5_test.npz")
print("Load data over!")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

X_test = scaler.transform(X_test)
test(model, X_test, y_test)
print("test end...")
