import torch.nn as nn
from torch import autograd, Tensor


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, epoch, num_classes, dropout):
        super(RNNClassifier, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv1d(in_channels=input_size,
                        out_channels=input_size // 2,
                        kernel_size=5,
                        stride=1,
                        padding=2
                        ),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=1),
                #------------------------------------
                nn.Conv1d(in_channels=input_size//2,
                            out_channels=input_size //4,
                            kernel_size=5,
                            stride=1,
                            padding=2
                            ),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=1)
                                )
        # self.fc_ = nn.Linear(hidden_size, hidden_size//2)
        self.rnn = nn.LSTM(input_size // 4, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, num_classes-1)

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = autograd.Variable(x)

        x = self.conv(x)

        x = x.transpose(1, 2)

        x, (_, _) = self.rnn(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        x = x[:, -1, :]

        x = self.fc(x)  # Output layer

        return x.flatten(start_dim=0)
