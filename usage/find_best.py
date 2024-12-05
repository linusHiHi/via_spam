import optuna.visualization as vis
import optuna
import pandas as pd

import torch.nn as nn
from sklearn.model_selection import train_test_split
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

X = df.drop(columns=[tag_name]) # Feature vectors,384 dimensions
y = df[tag_name]  # Labels (e.g., spam/ham)

# 分训练集（后续还要smote 才能使用）
#returns
(X_train, X_test,
 y_train, y_test) \
    = (train_test_split(X, y, test_size=0.3, random_state=42))
# \qwq it is maybe a wrong with the shape of y


X_train_len = X_train[original_length].to_numpy()
X_train = X_train[text_name].to_numpy()


X_test_len = X_test[original_length].to_numpy()
X_test = X_test[text_name].to_numpy()


del df
del X
del y


def objective(trial):
    # 超参数搜索范围
    # hidden_size_for_fc = trial.suggest_int("hidden_size_for_fc", 8, 32, step=8)
    # num_layers = trial.suggest_int("num_layers", 2, 3)
    num_layers = 2

    hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
    batch_size = trial.suggest_int("batch_size", 16, 32, step=16) # 后面有依赖

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3)
    num_epochs = trial.suggest_int("num_epochs", 40, 70)
    dropout = trial.suggest_float("dropout", 0.0,0.3, step=0.1)


    '''
    ************************* init pytorch dataset ******************************
    '''
    # Initialize dataset and dataloader
    dataset = TextDataset(X_train, y_train, max_sentences, dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train)

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, max_sentences, dim, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test)

    '''
    *************************Model initialization******************************
    '''

    model = RNNClassifier(input_size, hidden_size, num_layers, batch_size, num_epochs, num_classes, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_pp = RNNPlus(model,criterion,optimizer,device,dataloader,test_dataloader)
    train_loss_array,test_loss_array = model_pp.epochs_t_e(num_epochs)

    # p = 0.3
    # return p * train_loss_array[-1]+ (1 - p)*test_loss_array[-1]
    return train_loss_array[-1]


study = optuna.create_study(direction="minimize", study_name="hello")
study.optimize(objective, n_trials=30)

# 输出最佳超参数
print("Best trial:")
print(study.best_trial.params)

import json
# 保存最佳超参数到 JSON 文件
best_trial_params = study.best_trial.params
with open(path_to_best_parameter, 'w') as f:
    f.write(json.dumps(best_trial_params, indent=4))



# 可视化优化历史
fig = vis.plot_optimization_history(study)
fig.show()
