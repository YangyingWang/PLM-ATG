import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_data(path):
    data = np.load(path)
    X = data['features']
    y = data['labels']

    return X, y

print("Start load data...")
# X_train, y_train = get_data("D:/Major/AIProject/ATG/data/pretrain/Train/t5_esm2.npz")
X_train, y_train = get_data("D:/Major/AIProject/ATG/data/aadp_t5_train.npz")
print("Load data over!")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

rf = RandomForestClassifier(
    max_depth=5, 
    min_samples_leaf=10, 
    min_samples_split=10, 
    n_estimators=350, 
    max_features='sqrt', 
    random_state=42 
)
rf.fit(X_train, y_train)
print("Random Forest model trained successfully.")

explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_train)
print(f"SHAP values structure for RF: {np.array(shap_values_rf).shape}")

shap.summary_plot(shap_values_rf[:, :, 1], X_train, feature_names=[f'Feature {i}' for i in range(X_train.shape[1])], cmap='viridis')

mean_shap_values_rf = np.mean(np.abs(shap_values_rf[:, :, 1]), axis=0)
sorted_features_rf = np.argsort(mean_shap_values_rf)[::-1]

np.savez("D:/Major/AIProject/ATG/data/sorted_features(aadp_t5).npz", 
         sorted_features=sorted_features_rf, features=X_train, labels=y_train)
print("Feature selection based on SHAP values complete. Selected features saved.")
