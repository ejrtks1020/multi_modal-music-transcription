import torch.nn as nn
import torch
from .mfcc import CNNModel
from .resnet import ResNet, BasicBlock
    

class MFCCNet(nn.Module):
    def __init__(self, pretrained=False, min_key=5, max_key=80, mfcc_output_size=64, resnet_output_size=64):
        super(MFCCNet, self).__init__()
        self.num_classes=(max_key-min_key)+1

        self.mfcc_net = CNNModel(mfcc_output_size)
        self.resnet = ResNet(BasicBlock, layers=[2, 2, 2, 2], top_channel_nums=512, reduced_channel_nums=64, num_classes=resnet_output_size)
        if pretrained:
            model_path = '/opt/ml/data/model/Video2Roll_bestf1.pth' # change to your path
            checkpoint = torch.load(model_path, map_location="cuda:0")
            checkpoint['fc.weight'] = self.resnet.fc.weight
            checkpoint['fc.bias'] = self.resnet.fc.bias
            self.resnet.cuda()
            self.resnet.load_state_dict(checkpoint)

        self.fc = nn.Sequential(
            nn.Linear(mfcc_output_size+resnet_output_size, (mfcc_output_size+resnet_output_size) * 2),
            nn.BatchNorm1d((mfcc_output_size+resnet_output_size) * 2),
            nn.ReLU(True),

            nn.Linear((mfcc_output_size+resnet_output_size) * 2, (mfcc_output_size+resnet_output_size) * 4),
            nn.BatchNorm1d((mfcc_output_size+resnet_output_size) * 4),
            nn.ReLU(True),

            nn.Linear((mfcc_output_size+resnet_output_size) * 4, (mfcc_output_size+resnet_output_size) * 8),
            nn.BatchNorm1d((mfcc_output_size+resnet_output_size) * 8),
            nn.ReLU(True),

            nn.Linear((mfcc_output_size+resnet_output_size) * 8, self.num_classes)
            )
    
    def forward(self, x_frames, x_audio):
        x_frames = self.resnet(x_frames)
        x_audio = self.mfcc_net(x_audio)
        x = torch.concatenate((x_frames, x_audio), dim=1)
        # x = x_frames + x_audio
        y = self.fc(x)
        return y


if __name__ == "__main__":
    model = MFCCNet()
    x_frames = torch.randn(2, 5, 100, 900)
    x_audio = torch.randn(2, 5, 13, 3)
    print(model(x_frames, x_audio).shape)