import torch.nn as nn
import torch
from .resnet import ResNet, BasicBlock
import torch.nn.functional as F

class AudioCNNModel(nn.Module):
    def __init__(self, input_channels, output_size=256):
        super(AudioCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 1764 // 4, output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResAudio(nn.Module):
    def __init__(self, pretrained=False, min_key=5, max_key=80, audio_cnn_output_size=256, resnet_output_size=256):
        super(ResAudio, self).__init__()
        self.num_classes=(max_key-min_key)+1

        self.audio_cnn = AudioCNNModel(input_channels=5, output_size=audio_cnn_output_size)
        self.resnet = ResNet(BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=256, num_classes=resnet_output_size)
        if pretrained:
            model_path = '/opt/ml/Audeo/Audeo_github/Video2Roll_models/Video2RollNet.pth' # change to your path
            checkpoint = torch.load(model_path, map_location="cuda:0")
            checkpoint['fc.weight'] = self.resnet.fc.weight
            checkpoint['fc.bias'] = self.resnet.fc.bias
            self.resnet.cuda()
            self.resnet.load_state_dict(checkpoint)

        self.fc = nn.Sequential(
            nn.Linear(audio_cnn_output_size + resnet_output_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes))
    
    def forward(self, x_frames, x_audio):
        x_frames = self.resnet(x_frames)
        x_audio = self.audio_cnn(x_audio)
        x = torch.concatenate((x_frames, x_audio), dim=1)
        # x = x_frames + x_audio
        y = self.fc(x)
        return y
    
    
if __name__ == "__main__":
    model = ResAudio()
    x_frames = torch.randn(2, 5, 100, 900)
    x_audio = torch.randn(2, 5, 1764)
    print(model(x_frames, x_audio).shape)