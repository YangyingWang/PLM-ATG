import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform, randint
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    recall_score, accuracy_score, precision_score,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt

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

def train(X_train, y_train, cnt):
    params = {
        'C': randint(1, 20),
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

    joblib.dump(best_model, f"D:/Major/AIProject/ATG/models/SVM_AADP_ESM2({cnt}).pkl")
    with open("D:/Major/AIProject/ATG/performance/Select_features(aadp_esm).txt", "a") as f:
        f.write(f"-------------------SVM OF AADP_ESM2({cnt})--------------------\n")
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

def test(model, X_test, y_test, cnt):
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

    with open("D:/Major/AIProject/ATG/performance/Select_features(aadp_esm).txt", "a") as f:
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Test MCC: {test_mcc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(f"Test Recall: {test_recall}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"Test Specificity: {test_specificity}\n")
        f.write(f"Test ROC AUC: {test_roc_auc}\n")
        f.write(f"-------------------------------------------------\n\n")
    return test_accuracy, test_f1, test_mcc


sorted_data = np.load("D:/Major/AIProject/ATG/data/sorted_features(aadp_esm).npz")
sorted_features = sorted_data['sorted_features']

print("Start load data...")
# X_train, y_train = get_data("D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz")
X_train, y_train = get_data("D:/Major/AIProject/ATG/data/aadp_esm2_train.npz")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("\nStart load data...")
# X_test, y_test = get_data("D:/Major/AIProject/ATG/data/pretrain/Test/t5_esm2.npz")
X_test, y_test = get_data("D:/Major/AIProject/ATG/data/aadp_esm2_test.npz")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_cnt = [1700, 1500, 1300, 1100, 1000, 800, 600, 400, 200, 100]
test_accs = []
test_f1s = []
test_mccs = []

for cnt in feature_cnt:
    print(f"\nTraining with top {cnt} features...")
    X_selected = X_train[:, sorted_features[:cnt]]
    X_test_selected = X_test[:, sorted_features[:cnt]]
    model = train(X_selected, y_train, cnt)

    acc, f1, mcc = test(model, X_test_selected, y_test, cnt)
    print("test end...")

    test_accs.append(acc)
    test_f1s.append(f1)
    test_mccs.append(mcc)


plt.figure(figsize=(10, 6)) 
plt.plot(feature_cnt, test_accs, label='Test Accuracy', marker='o')
plt.plot(feature_cnt, test_f1s, label='Test F1_Score', marker='x')
plt.plot(feature_cnt, test_mccs, label='Test MCC', marker='s')

plt.xlabel('Number of Features')
plt.ylabel('Performance')
plt.title('Performance vs Number of Features')
plt.grid(True)
plt.legend(loc='best') 
plt.tight_layout()
plt.savefig('D:/Major/AIProject/ATG/fig2/Feature_select(aadp_esm).png')
plt.show()
