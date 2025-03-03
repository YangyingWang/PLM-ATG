import umap
import umap.plot
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

paths = [
    "D:/Major/AIProject/ATG/data/pretrain/Train/esm2_ne.npz",
    "D:/Major/AIProject/ATG/data/pretrain/Train/esm2_po.npz"
]

print("Start load data...")
X, y = load_data(paths)
print("Load data over!")
print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
label_names = {0: 'Non-ATG', 1: 'ATG'}
y_labels = [label_names[label] for label in y]
# model = joblib.load("D:/Major/AIProject/ATG/models/based_plms/SVM_T5.pkl")

reducer = umap.UMAP(n_components=2, random_state=5)
X_embedded = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_labels, palette="viridis", s=40)

plt.title("UMAP projection of the dataset")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.legend(title="", loc="best")
plt.savefig('D:/Major/AIProject/ATG/fig2/umap1.png')
plt.show()