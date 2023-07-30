import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, otuput_size=64):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(5, 32, 2, 1),
            nn.ReLU(),            
            nn.Conv2d(32, otuput_size, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d((11, 1)),
            nn.Flatten(start_dim=1)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassificationModel, self).__init__()
        self.mfcc_cnn = CNNModel(num_classes)
        self.mfcc_rnn = RNNModel(input_size=13, hidden_size=64, num_layers=2, num_classes=num_classes)

    def forward(self, mfcc):
        cnn_output = self.mfcc_cnn(mfcc)
        rnn_input = cnn_output.view(cnn_output.size(0), 1, -1)
        rnn_output = self.mfcc_rnn(rnn_input)
        return rnn_output