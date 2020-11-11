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
import cv2

import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        '''
        A Gaussian noise class that will add gaussian noise to spectrograms. This can be called by pytorch compose function.

        Params:

        : mean:
            Gaussian mean value.
        
        : std:
            Gaussian standard deviation
        
        : p:
            The probability to trigger this Gaussian function.
        '''
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
    def __init__(self, test_folder='1', is_train=True, spec_path='./urbansound_spectrogram'):
        '''
        Dataloader for urbansound dataset. Can only be used for original mobilenet and squeezenet.

        Params:

        : test_folder:
            foider id for testing. 
        
        : is_train:
            True if this loader is for training and False if it's for testing.

        : spec_path:
            The root path for spectrograms.
        '''
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
    def __init__(self, 
            test_folder=[12, 24], 
            is_train=True, 
            is_aug=False,
            input_type="all",
            spec_path='./data/2d_feature/spectrogram', 
            chr_path='./data/2d_feature/chromagram', 
            cqt_path='./data/2d_feature/cqt', 
            cens_path='./data/2d_feature/cens', 
            mfcc_path='./data/2d_feature/mfcc', 
            dmfcc_path='./data/2d_feature/dmfcc', 
            ddmfcc_path='./data/2d_feature/ddmfcc', 
            mutilhead=True,
        ):
        '''
        Dataloader for ravdess dataset. Can be used for original mobilenet, squeezenet and modified version of squeezenet.

        Params:

        : test_folder:
            testing folder ID. Default is [12, 24]
        
        : is_aug:
            Whether to use masking augmentation. Default is False as masking introduce degradation on Radvess dataset.
        
        : input type:
            Choice of different audio features, we have the following options:
            all: 
                Input is the combination of 7 audio features metioned in 
                https://2hatsecurity.atlassian.net/wiki/spaces/RDS/pages/702218542/Different+Audio+Features+and+Data+Aggregation+Test
            spectrogram:
                Input with only spectrogram
            chromagram:
                Input with only chromagram, CENs, and CQC
            mfcc:
                Input with only mfcc,, delta of mfcc, and second delta of mfcc
            scm:
                concatenation of spectrogram, chromagram, and mfcc
        
        : spec_path:
            spectrogram path.
        
        : chr_path:
            chromagram path.
        
        : cqt_path:
            cqt feature path

        : cens_path:
            cens feature path
        
        : mfcc path:
            mfcc feature path
        
        : dmfcc path:
            mfcc first order difference feature path
        
        : ddmfcc_path:
            mfcc second order difference feature path
        
        : multihead:
            If using multi-branch model, return the concatenation of all the input feature maps.
        '''
        self.is_multihead = mutilhead
        self.spc_path = spec_path
        self.chr_path = chr_path
        self.cqt_path = cqt_path
        self.cens_path = cens_path
        self.mfcc_path = mfcc_path
        self.dmfcc_path = dmfcc_path
        self.ddmfcc_path = ddmfcc_path

        self.file_list = os.listdir(spec_path)
        self.classID = [int(x.split('-')[2]) - 1 for x in self.file_list]
        self.spkID = [int(x.split('-')[-1].split('.png')[0]) for x in self.file_list]
        self.path_list = []
        self.id_list = []

        self.spec_aug = sptgFeatureAugmentation()
        self.is_train = is_train
        self.is_aug = is_aug
        self.input_type = input_type

        if self.is_train:
            print('is training')
            self.aug_opt_1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                AddGaussianNoise(mean=0., std=1., p=0.5),
                transforms.ToTensor()
            ])
            self.aug_opt_2 = transforms.Compose([
                transforms.RandomVerticalFlip(p=1),
                AddGaussianNoise(mean=0., std=1., p=0.5),
                transforms.ToTensor(),
            ])
            self.aug_opt_3 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                AddGaussianNoise(mean=0., std=1., p=0.5),
                transforms.ToTensor(),
            ])

            self.toTensor = transforms.Compose([
                transforms.ToTensor(),
            ])
            for i in range(len(self.file_list)):
                    
                if self.spkID[i] not in test_folder:
                        self.path_list.append(self.file_list[i])
                        self.id_list.append(self.classID[i])
        else:
            self.toTensor = transforms.Compose([transforms.ToTensor()])
            for i in range(len(self.file_list)):
                if self.spkID[i] in test_folder:
                    self.path_list.append(self.file_list[i])
                    self.id_list.append(int(self.classID[i]))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        label = self.id_list[index]

        # use only spectrogram
        if self.input_type == 'spectrogram':
            img_path = os.path.join(self.spc_path, self.path_list[index])
            spc = Image.open(img_path)
            if self.is_aug:
                img = np.array(spc)
                img = self.spec_aug(img)
                img = Image.fromarray(img)
            if self.toTensor:
                img = self.toTensor(img)

        # use only chromagram
        elif self.input_type == 'chromagram':
            img_path = os.path.join(self.chr_path, self.path_list[index])
            chroma = Image.open(img_path)
            if self.is_aug:
                img = np.array(chroma)
                img = self.spec_aug(img)
                img = Image.fromarray(img)
            if self.toTensor:
                img = self.toTensor(img)
        
        # use only mfcc
        elif self.input_type == 'mfcc':
            img_path = os.path.join(self.mfcc_path, self.path_list[index])
            mfcc = Image.open(img_path)
            if self.is_aug:
                img = np.array(mfcc)
                img = self.spec_aug(img)
                img = Image.fromarray(img)
            if self.toTensor:
                img = self.toTensor(img)

        # use spectrogram, chromagram, mfcc, cqt, cens, dmfcc, ddmfcc
        elif self.input_type == 'all':
            spc_path = os.path.join(self.spc_path, self.path_list[index])
            chr_path = os.path.join(self.chr_path, self.path_list[index])
            cqt_path = os.path.join(self.cqt_path, self.path_list[index])
            cens_path = os.path.join(self.cens_path, self.path_list[index])
            mfcc_path = os.path.join(self.mfcc_path, self.path_list[index])
            dmfcc_path = os.path.join(self.dmfcc_path, self.path_list[index])
            ddmfcc_path = os.path.join(self.ddmfcc_path, self.path_list[index])
            path_list = [spc_path, chr_path, cqt_path, cens_path, mfcc_path, dmfcc_path, ddmfcc_path]
            features = [Image.open(x) for x in path_list]

            # choose data augmentation methods
            if self.is_train:
                p = random.randint(0,3)
                if p == 0:
                    features = [self.aug_opt_1(x) for x in features]
                elif p == 1:
                    features = [self.aug_opt_2(x) for x in features]
                elif p == 2:
                    features = [self.aug_opt_3(x) for x in features]
                else:
                    features = [self.toTensor(x) for x in features]
            else:
                features = [self.toTensor(x) for x in features]

            if not self.is_multihead:
                img = torch.cat(
                    (
                        features[0], features[1], features[2], features[3], 
                        features[4], features[5], features[6]
                    ), dim=0)
            else:
                img = features

        # use spectrogram, chromagram, mfcc
        elif self.input_type == 'scm':
            spc_path = os.path.join(self.spc_path, self.path_list[index])
            chr_path = os.path.join(self.chr_path, self.path_list[index])
            mfcc_path = os.path.join(self.mfcc_path, self.path_list[index])
            path_list = [spc_path, chr_path, mfcc_path]

            features = [Image.open(x) for x in path_list]
            if self.is_train:
                p = random.randint(0,3)
                if p == 0:
                    features = [self.aug_opt_1(x) for x in features]
                elif p == 1:
                    features = [self.aug_opt_2(x) for x in features]
                elif p == 2:
                    features = [self.aug_opt_3(x) for x in features]
                else:
                    features = [self.toTensor(x) for x in features]
            else:
                features = [self.toTensor(x) for x in features]
            if not self.is_multihead:
                img = torch.cat(
                    (
                        features[0], features[1], features[2]
                    ), dim=0)
            else:
                img = features
        else:
            raise ValueError("{} is not a valid option for input data".format(self.input_type))

        return img, label