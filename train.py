from dataset import ImageAudioDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
from models import RESTM, ResAudio, CustomLSTM, MFCCNet
from trainer import Solver
import torch.nn as nn
from dataset import MultilabelBalancedRandomSampler

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    train_dataset = ImageAudioDataset(subset='train')
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=8)
    test_dataset = ImageAudioDataset(subset='test')
    test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)
    net = MFCCNet(pretrained=True)
    net.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
    solver = Solver(train_data_loader, test_data_loader, net, criterion, optimizer, scheduler, epochs=100)
    solver.train()
