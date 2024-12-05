
import pandas as pd

import torch.nn as nn
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader

from config.variables import *
from train.dataset import TextDataset
from train.conv_lstm import RNNClassifier
from train.trainer import RNNPlus

'''
***********************dealing data**********************************
'''
# read source data
df = pd.read_pickle(path_to_source)

X = df[text_name] # Feature vectors,384 dimensions
y = df[tag_name]  # Labels (e.g., spam/ham)

X = X.to_numpy()
y = y.to_numpy()

with open(path_to_best_parameter) as f:
    hp_params = json.load(f)

batch_size = hp_params['batch_size']
hidden_size = hp_params['hidden_size']
num_epochs = hp_params['num_epochs']
num_layers = 2
learning_rate = hp_params['learning_rate']
dropout = hp_params['dropout']
"""
***************** 进行 K-fold 交叉验证******************************
"""
kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_losses = []
train_losses = []

# 进行 K-fold 交叉验证
for train_idx, test_idx in kf.split(X):
    # 获取训练和验证数据
    """
    ************************* dealing data ****************************
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    """
    ******************* pytorch dataset *************************
    """
    dataset = TextDataset(X_train, y_train, max_sentences, dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, max_sentences, dim, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    """early_stop_flag
    ***********************创建模型*********************
    """
    model = RNNClassifier(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers,num_classes=num_classes, batch_size=batch_size, dropout=dropout,epoch=num_epochs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_pp = RNNPlus(model, criterion, optimizer, device, dataloader, test_dataloader,stop_delta=0.01, stop_patient=7)

    """
    ****************创建模型*********************
    """
    train_loss_array,test_loss_array = model_pp.epochs_t_e(num_epochs)
    test_losses.append(sum(test_loss_array)/len(test_loss_array))
    train_losses.append(sum(train_loss_array)/len(train_loss_array))

df = pd.DataFrame({'train_loss': train_losses, 'test_loss': test_losses})
df.to_pickle(path_to_k_fold)
