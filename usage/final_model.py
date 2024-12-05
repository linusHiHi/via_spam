import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config.variables import *
from train.conv_lstm import RNNClassifier
from train.dataset import TextDataset
from train.trainer import RNNPlus

df = pd.read_pickle(path_to_source)

X = df[text_name]  # Feature vectors,384 dimensions
y = df[tag_name]  # Labels (e.g., spam/ham)

test_loss_array = [100]
while test_loss_array[-1] > 0.13:
    '''
    ***********************dealing data**********************************
    '''
    # read source data


    # 分训练集（后续还要smote 才能使用）
    # returns
    (X_train, X_test,
     y_train, y_test) \
        = (train_test_split(X, y, test_size=0.3, random_state=42))
    # \qwq it is maybe a wrong with the shape of y

    X_train = X_train.to_numpy()

    X_test = X_test.to_numpy()

    '''
    ************************* init pytorch dataset ******************************
    '''
    with open(path_to_best_parameter) as f:
        hp_params = json.load(f)

    batch_size = hp_params['batch_size']
    hidden_size = hp_params['hidden_size']
    num_epochs = hp_params['num_epochs']
    num_layers = 2
    learning_rate = hp_params['learning_rate']
    dropout = hp_params['dropout']

    # Initialize dataset and dataloader
    dataset = TextDataset(X_train, y_train, max_sentences, dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, max_sentences, dim, is_test=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)


    '''
    *************************Model initialization******************************
    '''

    model = RNNClassifier(input_size, hidden_size, num_layers, batch_size, num_epochs, num_classes, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_pp = RNNPlus(model, criterion, optimizer, device, dataloader, test_dataloader)
    train_loss_array, test_loss_array = model_pp.epochs_t_e(num_epochs)
    print(f"final loss: {train_loss_array[-1]}, final test loss: {test_loss_array[-1]}")
    torch.save(model.state_dict(),path_to_trained_model)