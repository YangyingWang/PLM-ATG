import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, matthews_corrcoef, f1_score,
    recall_score, accuracy_score, precision_score,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def get_data(path):
    data = np.load(path)
    X = data['features']
    y = data['labels']

    return X, y

sorted_data = np.load("D:/Major/AIProject/ATG/data/sorted_features(aadp_esm).npz")
sorted_features = sorted_data['sorted_features']

print("Start load data...")
X_train, y_train = get_data("D:/Major/AIProject/ATG/data/aadp_esm2_train.npz")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("\nStart load data...")
X_test, y_test = get_data("D:/Major/AIProject/ATG/data/aadp_esm2_test.npz")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_selected = X_train[:, sorted_features[:400]]
X_test_selected = X_test[:, sorted_features[:400]]
model = joblib.load("D:/Major/AIProject/ATG/models/SVM_AADP_ESM2(400).pkl")

y_test_pred = model.predict(X_test_selected)
y_test_pred_proba = model.predict_proba(X_test_selected)[:, 1]

# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_mcc = matthews_corrcoef(y_test, y_test_pred)
# test_f1 = f1_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
# test_specificity = tn / (tn + fp)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
# roc_auc = auc(fpr, tpr)

# print(f"Accuracy: {test_accuracy}")
# print(f"MCC: {test_mcc}")
# print(f"F1 Score: {test_f1}")
# print(f"Recall: {test_recall}")
# print(f"Precision: {test_precision}")
# print(f"Specificity: {test_specificity}")

# plt.figure()
# plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
# plt.plot([0, 1], [0, 1],'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.savefig('D:/Major/AIProject/ATG/fig2/roc_curves.png')
# plt.show()

label_names = {0: 'Non-ATG', 1: 'ATG'}
y_labels = [label_names[label] for label in y_train]

X_transformed = model.decision_function(X_selected) if hasattr(model, "decision_function") else model.predict_proba(X_selected)

if X_transformed.ndim == 1:
    X_transformed = X_transformed.reshape(-1, 1)

tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42, init='random')
X_embedded = tsne.fit_transform(X_transformed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_labels, palette="viridis", s=80)
plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="", loc="best")
plt.savefig('D:/Major/AIProject/ATG/fig2/t-SNE2.png')
plt.show()