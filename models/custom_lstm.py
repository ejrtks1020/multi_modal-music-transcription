import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1764, hidden_size=256, num_layers=2, output_size=64):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(True),
            # nn.Linear(512, 256),
            # nn.ReLU(True),
            # nn.Linear(256, output_size),
            )

    def forward(self, x):
        # 초기 hidden state와 cell state 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 모델 적용
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 타임스텝의 hidden state를 추출하여 특징 벡터로 변환
        features = self.fc(out[:, -1, :])

        return features


def conv_block(
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               pool=False,
               norm=False):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 
                  out_channels=out_channels, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding)
    )
    if norm:
        block.append(nn.BatchNorm2d(out_channels))
    if pool:
        block.append(nn.MaxPool2d(2, 2))
    return block

class CustomBackbone(nn.Module):
    def __init__(self, output_size):
        super(CustomBackbone, self).__init__()
        self.backbone = nn.Sequential(
            conv_block(5, 32, 3, 1, 1, norm=True),
            conv_block(32, 64, 3, 1, 1, pool=True, norm=True),
            conv_block(64, 64, 3, 1, 1, pool=True, norm=True),
            conv_block(64, 128, 3, 1, 1, pool=True),
            conv_block(128, 256, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),            
            conv_block(256, 256, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),
            conv_block(256, 512, 3, 1, 1),
            nn.MaxPool2d((1, 2), (1,2)),
            conv_block(512, 512, 3, 1, 1),
            nn.MaxPool2d((1, 3), (1,1)),
            conv_block(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x.squeeze())
        return out

class CustomLSTM(nn.Module):
    def __init__(self, min_key=5, max_key=80, lstm_output_size=64, backbone_output_size=64):
        super(CustomLSTM, self).__init__()
        self.num_classes=(max_key-min_key)+1
        self.lstm = LSTMModel(output_size=lstm_output_size)
        self.backbone = CustomBackbone(output_size=backbone_output_size)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size + backbone_output_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, self.num_classes),
            )
        
    def forward(self, x_frames, x_audio):
        x_frames = self.backbone(x_frames)
        x_audio = self.lstm(x_audio)
        x = torch.concatenate((x_frames, x_audio), dim=1)
        # x = x_frames + x_audio
        y = self.fc(x)
        return y

if __name__ == "__main__":
    model = CustomLSTM()
    x_frames = torch.randn(2, 5, 100, 900)
    x_audio = torch.randn(2, 5, 1764)
    print(model(x_frames, x_audio).shape)