import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    recall_score, accuracy_score, precision_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

split_seed = 53
EPOCHES = 100
LEARNING_RATE = 0.001
MODEL_DIR = "D:/Major/AIProject/ATG/models/based_pssm"

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
        return self.X[idx],self.Y[idx]
    
    def __len__(self):
        return len(self.X)

class DNNs(nn.Module):
    # def __init__(self, input_dim=20, dropout_rate=0.2):
    #     super(DNNs, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, 15)
    #     self.dropout1 = nn.Dropout(dropout_rate)
    #     self.fc2 = nn.Linear(15, 10)
    #     self.dropout2 = nn.Dropout(dropout_rate)
    #     self.output = nn.Linear(10, 1)
    #     self.best_val_loss = float('inf')
    #     self.best_model_state = None

    def __init__(self, input_dim=400, dropout_rate=0.2):
        super(DNNs, self).__init__()
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
  
        auc = roc_auc_score(arr_labels, arr_prob)
        acc = accuracy_score(arr_labels, arr_labels_hyp)
        mcc = matthews_corrcoef(arr_labels, arr_labels_hyp)
        precision = precision_score(arr_labels, arr_labels_hyp)
        f1 = f1_score(arr_labels, arr_labels_hyp)
        sensitivity = recall_score(arr_labels, arr_labels_hyp)
        cm = confusion_matrix(arr_labels, arr_labels_hyp)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        
        result = {'epoch_loss_avg': epoch_loss / len(loader), 
                    'acc' : acc, 
                    'confusion_matrix' : cm,
                    'sensitivity' : sensitivity,
                    'specificity' : specificity,
                    'mcc' : mcc,
                    'auc' : auc,
                    'precision': precision,
                    'f1': f1,
                    'arr_prob': arr_prob,
                    'arr_labels': arr_labels
                     }
        return result
    
    def train_one_epoch(self, criterion, optimizer):
        self.train()
        epoch_loss_train = 0.0

        for inputs, labels in self.train_loader:
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()

        return epoch_loss_train / len(self.train_loader)

    def train_validate(self, train_dataset, validate_dataset):
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=validate_dataset, batch_size=8, shuffle=True, num_workers=4)                                        

        criterion = nn.BCELoss()  # 二元交叉熵损失
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHES):
            print(f"\n############### EPOCH : {epoch+1}")
            epoch_loss = self.train_one_epoch(criterion, optimizer)

            print("Evaluate Train set")
            result = self.evaluate(self.train_loader, criterion)
            print("Train loss: ", result['epoch_loss_avg'])
            print("Train acc: ", result['acc'])

            print("\nEvaluate Val set")
            result_val = self.evaluate(self.val_loader, criterion)
            print("Val loss: ", result_val['epoch_loss_avg'])
            print("Val acc: ", result_val['acc'])

            if self.best_val_loss > result_val['epoch_loss_avg']:
                self.best_val_loss = result_val['epoch_loss_avg']
                self.best_model_state = self.state_dict()


def training_k_fold(features, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    best_overall_model_state = None
    best_overall_val_loss = float('inf')

    for fold,(train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f'\n##########Fold : {fold+1}/5')
        X_train, X_val = features[train_idx], features[val_idx]
        Y_train, Y_val = labels[train_idx], labels[val_idx]

        unique, count = np.unique(Y_train, return_counts = True)
        print("Y_train values: ", unique, "count: ", count)
        unique, count = np.unique(Y_val, return_counts = True)
        print("y_val values: ", unique, "count: ", count)

        train_set = FPDataset(X_train, Y_train)
        val_set = FPDataset(X_val, Y_val)
        
        model = DNNs(input_dim=X_train.shape[1])
        if torch.cuda.is_available():
            model.cuda()
        model.train_validate(train_set, val_set)

        if model.best_val_loss < best_overall_val_loss:
            best_overall_val_loss = model.best_val_loss
            best_overall_model_state = model.best_model_state
    torch.save(best_overall_model_state, os.path.join(MODEL_DIR, "DNN_AADP_PSSM.pkl"))
    print(f"Overall best model saved with val loss {best_overall_val_loss:.4f}")

def test(features, labels):
    test_set = FPDataset(features, labels)
    test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False, num_workers=4)

    model = DNNs(input_dim=features.shape[1])
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.BCELoss()
    model.load_state_dict(torch.load(MODEL_DIR+"/DNN_AADP_PSSM.pkl"))
    result = model.evaluate(test_loader, criterion)

    with open("D:/Major/AIProject/ATG/performance/DNN_results.txt", "a") as f:
        f.write(f"-------------------DNN OF AADP_PSSM--------------------\n")
        f.write(f"Test Accuracy: {result['acc']}\n")
        f.write(f"Test MCC: { result['mcc']}\n")
        f.write(f"Test F1 Score: {result['f1']}\n")
        f.write(f"Test Recall: {result['sensitivity']}\n")
        f.write(f"Test Precision: {result['precision']}\n")
        f.write(f"Test Specificity: {result['specificity']}\n")
        f.write(f"Test ROC AUC: {result['auc']}\n")
        f.write(f"-------------------------------------------------\n\n")

if __name__ == "__main__":
    ne_train = 'D:/Major/AIProject/ATG/data/pssmdata/aadp_ne_train.csv'
    po_train = 'D:/Major/AIProject/ATG/data/pssmdata/aadp_po_train.csv'
    ne_test = 'D:/Major/AIProject/ATG/data/pssmdata/aadp_ne_test.csv'
    po_test = 'D:/Major/AIProject/ATG/data/pssmdata/aadp_po_test.csv'

    print("Autophagy Training with 5-Fold Cross Validation using DNN")
    features_train, labels_train = load_data(ne_train, po_train)
    training_k_fold(features_train, labels_train)

    print("Autophagy Testing...")
    features_test, labels_test = load_data(ne_test, po_test)
    test(features_test, labels_test)
    print("test end...")

