import torch.nn as nn
import torch
from .resnet import ResNet, BasicBlock

class LSTMModel(nn.Module):
    def __init__(self, input_size=39, hidden_size=256, num_layers=2, output_size=64):
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
    

class RESTM(nn.Module):
    def __init__(self, pretrained=False, min_key=5, max_key=80, lstm_output_size=64, resnet_output_size=64):
        super(RESTM, self).__init__()
        self.num_classes=(max_key-min_key)+1

        self.lstm = LSTMModel(output_size=lstm_output_size)
        self.resnet = ResNet(BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, num_classes=resnet_output_size)
        if pretrained:
            model_path = '/opt/ml/multi_modal-music-transcription/weights/video_to_roll_best_f1_2.pth' # change to your path
            checkpoint = torch.load(model_path, map_location="cuda:0")
            checkpoint['fc.weight'] = self.resnet.fc.weight
            checkpoint['fc.bias'] = self.resnet.fc.bias
            self.resnet.cuda()
            self.resnet.load_state_dict(checkpoint)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size+resnet_output_size, self.num_classes),
            # nn.BatchNorm1d(256),
            # nn.Dropout(),
            # nn.ReLU(True),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(512),
            # nn.Dropout(),
            # nn.ReLU(True),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 512),
            # nn.ReLU(True),
            # nn.Linear(512, self.num_classes)
            )
    
    def forward(self, x_frames, x_audio):
        x_frames = self.resnet(x_frames)
        x_audio = self.lstm(x_audio)
        x = torch.concatenate((x_frames, x_audio), dim=1)
        # x = x_frames + x_audio
        y = self.fc(x)
        return y


if __name__ == "__main__":
    model = RESTM()
    x_frames = torch.randn(2, 5, 100, 900)
    x_audio = torch.randn(2, 5, 1764)
    print(model(x_frames, x_audio).shape)