import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tans_f
import numpy as np
import pandas as pd
import librosa
import os
from PIL import Image
import random

import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        if random.randrange(0, 1) > self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p={2})'.format(self.mean, self.std, self.p)

class sptgFeatureAugmentation(object):
    '''
    # customized spectrogram augmentation class
    :param tuple zone: Default value is (0.2, 0.8). Assign a zone for augmentation. By default, no any augmentation
         will be applied in first 20% and last 20% of whole audio.
    '''
    def __init__(self, freq_zone=(0.4, 0.5), time_zone=(0.1, 0.5), load_zone=(0, 1)):
        self.f_aug = nas.FrequencyMaskingAug(zone=freq_zone)
        self.t_aug = nas.TimeMaskingAug(zone=time_zone)
        self.load_aug = nas.LoudnessAug(zone=load_zone)
        
    def __call__(self, x):
        p = random.randint(0, 3)
        if p>1:
            x = self.f_aug.substitute(x)
        elif p>2:
            x = self.f_aug.substitute(x)
            x = self.t_aug.substitute(x)
        # x = self.load_aug.substitute(x)
        return x

class urbanSoundLoader(data.Dataset):
    def __init__(self, test_folder='1', is_train=True, spec_path='./data/urbansound_spectrogram'):
        self.img_path = spec_path
        self.file_list = os.listdir(spec_path)
        self.classID = [x.split('.png')[0].split('_')[1] for x in self.file_list]
        self.foldID = [x.split('.png')[0].split('_')[2] for x in self.file_list]
        self.path_list = []
        self.id_list = []
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            for i in range(len(self.foldID)):
                if self.foldID[i] != test_folder:
                    self.path_list.append(self.file_list[i])
                    self.id_list.append(self.classID[i])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            for i in range(len(self.foldID)):
                if self.foldID[i] == test_folder:
                    self.path_list.append(self.file_list[i])
                    self.id_list.append(self.classID[i])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.path_list[index])
        spc = Image.open(img_path)
        label = int(self.id_list[index])
        if self.transform:
            spc = self.transform(spc)
        return spc, label

class ravdessLoader(data.Dataset):
    def __init__(self, test_folder=[12, 24], is_train=True, spec_path='./data/ravdess_spectrogram2'):
        self.img_path = spec_path
        self.file_list = os.listdir(spec_path)
        self.classID = [int(x.split('-')[2]) - 1 for x in self.file_list]
        #--------------------------------------------------------------------------#
        # Loader speaker ID. Naming convention might be different depends on how you
        # generated the data.
        #--------------------------------------------------------------------------#
        # self.spkID = [int(x.split('_')[0].split('-')[-1].split('.png')[0]) for x in self.file_list]
        self.spkID = [int(x.split('-')[-1].split('.png')[0]) for x in self.file_list]
        self.path_list = []
        self.id_list = []

        self.spec_aug = sptgFeatureAugmentation()
        self.is_train = is_train

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                AddGaussianNoise(mean=0., std=1., p=0.5),
                transforms.ToTensor(),
            ])
            for i in range(len(self.file_list)):
                
                if self.spkID[i] not in test_folder:
                    self.path_list.append(self.file_list[i])
                    self.id_list.append(self.classID[i])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            for i in range(len(self.file_list)):
                if self.spkID[i] in test_folder:
                    self.path_list.append(self.file_list[i])
                    self.id_list.append(int(self.classID[i]))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.path_list[index])
        if self.is_train:
            spc = Image.open(img_path)
            spc = np.array(spc)
            spc = self.spec_aug(spc)
        spc = Image.fromarray(spc)
        label = self.id_list[index]
        if self.transform:
            spc = self.transform(spc)
        return spc, label

# ./data/ravdess_spectrogram2/03-01-07-02-02-02-12.png
# ./data/ravdess_spectrogram2/03-01-02-01-02-02-11.png