import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data_test = {
    'AAC': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/aac_ne_test.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/aac_po_test.csv'
    },
    'DPC': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/dpc_ne_test.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/dpc_po_test.csv'
    },
    'AADP': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/aadp_ne_test.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/aadp_po_test.csv'
    }
}

data_train = {
    'AAC': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/aac_ne_train.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/aac_po_train.csv'
    },
    'DPC': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/dpc_ne_train.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/dpc_po_train.csv'
    },
    'AADP': {
        'ne': 'D:/Major/AIProject/ATG/data/pssmdata/aadp_ne_train.csv',
        'po': 'D:/Major/AIProject/ATG/data/pssmdata/aadp_po_train.csv'
    }
}

models = {
    'AAC': {
        'LR': "D:/Major/AIProject/ATG/models/based_pssm/LR_AAC_pssm.pkl",
        'RF': "D:/Major/AIProject/ATG/models/based_pssm/RF_AAC_pssm.pkl",
        'SVM': "D:/Major/AIProject/ATG/models/based_pssm/SVM_AAC_pssm.pkl",
        'KNN': "D:/Major/AIProject/ATG/models/based_pssm/KNN_AAC_pssm.pkl",
        'BiLSTM': "D:/Major/AIProject/ATG/models/based_pssm/BiLSTM_AAC_PSSM.pkl",
        'DNN': "D:/Major/AIProject/ATG/models/based_pssm/DNN_AAC_PSSM.pkl"
    },
    'DPC': {
        'LR': "D:/Major/AIProject/ATG/models/based_pssm/LR_DPC_pssm.pkl",
        'RF': "D:/Major/AIProject/ATG/models/based_pssm/RF_DPC_pssm.pkl",
        'SVM': "D:/Major/AIProject/ATG/models/based_pssm/SVM_DPC_pssm.pkl",
        'KNN': "D:/Major/AIProject/ATG/models/based_pssm/KNN_DPC_pssm.pkl",
        'BiLSTM': "D:/Major/AIProject/ATG/models/based_pssm/BiLSTM_DPC_PSSM.pkl",
        'DNN': "D:/Major/AIProject/ATG/models/based_pssm/DNN_DPC_PSSM.pkl"
    },
    'AADP': {
        'LR': "D:/Major/AIProject/ATG/models/based_pssm/LR_AADP_pssm.pkl",
        'RF': "D:/Major/AIProject/ATG/models/based_pssm/RF_AADP_pssm.pkl",
        'SVM': "D:/Major/AIProject/ATG/models/based_pssm/SVM_AADP_pssm.pkl",
        'BiLSTM': "D:/Major/AIProject/ATG/models/based_pssm/BiLSTM_AADP_PSSM.pkl",
        'DNN': "D:/Major/AIProject/ATG/models/based_pssm/DNN_AADP_PSSM.pkl",
        'KNN': "D:/Major/AIProject/ATG/models/based_pssm/KNN_AADP_pssm.pkl"
    }
}

def load_data(path_ne, path_po):
    data_ne = pd.read_csv(path_ne, header=None)
    data_po = pd.read_csv(path_po, header=None)

    data_ne['label'] = 0
    data_po['label'] = 1

    data = pd.concat([data_ne, data_po])

    features = data.iloc[:, 1:-1].values
    labels = data['label'].values 

    print("Data loaded successfully!")
    print(f"Feature shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")

    return features, labels

class FPDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)
    
class BiLSTM(nn.Module):
    def __init__(self, input_dim=420, hidden_dim=128, num_layers=2, dropout=0.2): 
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        x = torch.sigmoid(self.fc(out_last))
        return x
    
    def evaluate(self, loader, criterion):
        self.eval()

        epoch_loss = 0.0
        arr_labels, arr_labels_hyp, arr_prob = [], [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.float().unsqueeze(1)
                labels = labels.unsqueeze(1).float()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                arr_prob.extend(outputs.cpu().numpy())
                arr_labels_hyp.extend([1 if p > 0.5 else 0 for p in outputs.cpu().numpy()])
                arr_labels.extend(labels.cpu().numpy())

        fpr, tpr, _ = roc_curve(arr_labels, arr_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

class DNNs1(nn.Module):
    def __init__(self, input_dim=20, dropout_rate=0.2):
        super(DNNs1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 15)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(15, 10)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(10, 1)
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output(x))
        return x
    
    def evaluate(self, loader, criterion):
        self.eval()

        epoch_loss = 0.0
        arr_labels, arr_labels_hyp, arr_prob = [], [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.float()
                labels = labels.unsqueeze(1).float()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                arr_prob.extend(outputs.cpu().numpy())
                arr_labels_hyp.extend([1 if p > 0.5 else 0 for p in outputs.cpu().numpy()])
                arr_labels.extend(labels.cpu().numpy())

        fpr, tpr, _ = roc_curve(arr_labels, arr_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
class DNNs2(nn.Module):
    def __init__(self, input_dim=400, dropout_rate=0.2):
        super(DNNs2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 350)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(350, 300)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(300, 250)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(250, 1)
        self.best_val_loss = float('inf')
        self.best_model_state = None        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output(x))
        return x
    
    def evaluate(self, loader, criterion):
        self.eval()

        epoch_loss = 0.0
        arr_labels, arr_labels_hyp, arr_prob = [], [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.float()
                labels = labels.unsqueeze(1).float()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()
                arr_prob.extend(outputs.cpu().numpy())
                arr_labels_hyp.extend([1 if p > 0.5 else 0 for p in outputs.cpu().numpy()])
                arr_labels.extend(labels.cpu().numpy())

        fpr, tpr, _ = roc_curve(arr_labels, arr_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

def plot_roc_curves(models, X_test, y_test, X_train, title, ax):
    for model_name, path in models.items():
        if model_name in ['BiLSTM', 'DNN']:
            test_set = FPDataset(X_test, y_test)
            test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=4)
            
            if model_name == 'BiLSTM':
                model = BiLSTM(input_dim=X_test.shape[1])
            elif model_name == 'DNN':
                if X_test.shape[1] == 20:
                    model = DNNs1(input_dim=20)
                else:
                    model = DNNs2(input_dim=X_test.shape[1])

            # 确保模型在每次循环开始时都处于初始状态
            model.best_val_loss = float('inf')
            model.best_model_state = None

            if torch.cuda.is_available():
                model.cuda()

            criterion = nn.BCELoss()
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            
            fpr, tpr, roc_auc = model.evaluate(test_loader, criterion)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
            
        else:
            if model_name == 'KNN':
                scaler = MinMaxScaler()
            else: scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = joblib.load(path)

            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2) #对角线
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")

if __name__ == '__main__':
    X_aac_test, y_aac_test = load_data(data_test['AAC']['ne'], data_test['AAC']['po'])
    X_dpc_test, y_dpc_test = load_data(data_test['DPC']['ne'], data_test['DPC']['po'])
    X_aadp_test, y_aadp_test = load_data(data_test['AADP']['ne'], data_test['AADP']['po'])

    X_aac_train, _ = load_data(data_train['AAC']['ne'], data_train['AAC']['po'])
    X_dpc_train, _ = load_data(data_train['DPC']['ne'], data_train['DPC']['po'])
    X_aadp_train, _ = load_data(data_train['AADP']['ne'], data_train['AADP']['po'])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # plot_roc_curves(models['AAC'], X_aac_test, y_aac_test ,X_aac_train, 'AAC-PSSM', axs[0])
    # plot_roc_curves(models['DPC'], X_dpc_test, y_dpc_test ,X_dpc_train, 'DPC-PSSM', axs[1])
    plot_roc_curves(models['AADP'], X_aadp_test, y_aadp_test ,X_aadp_train, 'AADP-PSSM', axs[2])

    plt.tight_layout()
    plt.savefig('D:/Major/AIProject/ATG/fig2/roc_curves(dl).png')
    plt.show()