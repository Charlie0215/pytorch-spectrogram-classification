import torch
# torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tran_f
from torchvision import transforms

from dataloader import urbanSoundLoader, ravdessLoader
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa
import cv2
from trainer import BaseTrainer, multiheadTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ravdess', help='class of dataset.')
parser.add_argument('--model', type=str, default='squeezenet1_0', help='class of network.')
parser.add_argument('--input_type', type=str, default='all', help='type of input features')
args = parser.parse_args()

def main():
    #--------------------------------------------------------------------------#
    # Configure dataloader.
    #--------------------------------------------------------------------------#
    if args.dataset == 'urbansound':
        from config import urbansound_train_config as config
        train_data_loader = DataLoader(urbanSoundLoader(test_folder='8'), 
            batch_size=config.batch_size, shuffle=True, num_workers=1)
        val_data_loader = DataLoader(urbanSoundLoader(test_folder='8', is_train=False), 
            batch_size=1, shuffle=False, num_workers=1)  
    elif args.dataset == 'ravdess':
        from config import ravdess_train_config as config
        train_data_loader = DataLoader(ravdessLoader(test_folder=[11, 12, 23, 24], input_type=args.input_type), 
            batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_data_loader = DataLoader(ravdessLoader(test_folder=[11, 12, 23, 24], input_type=args.input_type, is_train=False), 
            batch_size=1, shuffle=False, num_workers=0)  
    else:
        raise ValueError("Unsupport dataset {version}: "
                        "urbansound, "
                        "ravdess"
                        " expected".format(version=args.dataset))
    #--------------------------------------------------------------------------#
    # Configure model. 
    #--------------------------------------------------------------------------#
    if args.model == 'mobilenet':
        from models import mobilenet_v2 as network
        net = network(in_channels=1, num_classes=8)
    elif args.model == 'squeezenet1_0':
        if args.input_type == 'spectrogram':
            from models import squeezenet1_0 as network
            net = network(pretrained=False)
        elif args.input_type == 'scm':
            from models import squeezenet_multi as network
            net = network(num_input=3, growth_rate=32)
        elif args.input_type == 'all':
            from models import squeezenet_multi as network
            net = network(num_input=7, growth_rate=16)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    #------------------------------------------------------------------------#
    # Configure model, optimizer, schedular, loss, and trainer.
    #------------------------------------------------------------------------#
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    if config.pretrain == True:
        net.load_state_dict(torch.load('./weight/epoch_183.pkl')['model_state'])
        print('weight loaded.')
    else:
        print('initial training')
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total parameters: {}'.format(pytorch_total_params))
    loss = nn.CrossEntropyLoss().to(device)
    # Trainer = BaseTrainer(net, optimizer, scheduler, loss, train_data_loader, val_data_loader, True, config)
    Trainer = multiheadTrainer(net, optimizer, scheduler, loss, train_data_loader, val_data_loader, True, config)
    Trainer.run(config.epoch)


if __name__ == '__main__':
    main()
