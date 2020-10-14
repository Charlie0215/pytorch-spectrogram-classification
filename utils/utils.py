import os
import numpy as np
import pandas as pd
import librosa
import cv2
import skimage
import skimage.io
import torch
from tqdm import tqdm
import shutil

import nlpaug.augmenter.audio as naa

def prepare_urbansound(csv='./data/UrbanSound8K/metadata/UrbanSound8K.csv', audio_root='./data/UrbanSound8K'):
    save_dir = '../data/urbansound_spectrogram'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = pd.read_csv(csv)
    valid_data = data[['slice_file_name', 'fold', 'classID', 'class']][data['end']-data['start'] >= 3]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + \
        valid_data['slice_file_name'].astype('str') 

    for row in tqdm(valid_data.itertuples()):
        y1, sr1 = librosa.load('{}/audio/'.format(audio_root) + row.path, duration=2.97)
        ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
        ps = np.log(ps + 1e-9)
        if ps.shape != (128, 128):
            continue
        img_path = row.path.split('.wav')[0].split('/')[1] + '_{}'.format(row.classID) + \
            '_{}'.format(row.fold) + '.png'
        print(img_path)
        save_path = os.path.join(save_dir, img_path)

        img = scale_minmax(ps, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy

        # save as PNG
        skimage.io.imsave(save_path, img)

def accuracy(Y_, Y):
    Y_ = Y_.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    count = Y.shape[0]
    true_value = 0
    Y__avg = torch.mean(Y_, 2)
    pred = Y__avg.max(1, keepdim=True)[1]
    pred = pred.eq(Y[:, 0].view_as(pred))
    # print(acc)
    for i in pred:
        if i == True:
            true_value += 1
    acc = true_value / count
    return acc

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def prepare_ravdess(audio_root):
    save_dir = '../data/ravdess_spectrogram5'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for folder in os.listdir(audio_root):
        folder_path = os.path.join(audio_root, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            y1, sr1 = librosa.load(file_path, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y1, n_mels=128, hop_length=512)
            ps = np.log(ps + 1e-9)
            if ps.shape != (128, 128):
                continue
            img_path = file.split('.wav')[0] + '.png'
            save_path = os.path.join(save_dir, img_path)
            img = scale_minmax(ps, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255-img 
            skimage.io.imsave(save_path, img)

def prepare_ravdess_aug(audio_root):
    save_dir = '../data/ravdess_spectrogram7'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for folder in os.listdir(audio_root):
        folder_path = os.path.join(audio_root, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            folder_id = int(file.split('-')[-1].split('.wav')[0])
            if folder_id not in [11, 12, 23, 24]:
                y1, sr1 = librosa.load(file_path, duration=3.15)
                for i in range(10):
                    cropaug = naa.CropAug(sampling_rate=sr1)
                    loadaug = naa.LoudnessAug()
                    maskaug = naa.MaskAug(sampling_rate=sr1, mask_with_noise=False)
                    y = maskaug.augment(loadaug.augment(cropaug.augment(y1)))
                    ps = librosa.feature.melspectrogram(y=y, n_mels=128, hop_length=512)
                    ps = np.log(ps + 1e-9)
                    if ps.shape != (128, 128):
                        continue
                    img_path = file.split('.wav')[0] + '_{}'.format(i) + '.png'
                    save_path = os.path.join(save_dir, img_path)

                    img = scale_minmax(ps, 0, 255).astype(np.uint8)
                    img = np.flip(img, axis=0)
                    img = 255-img 
                    print(save_path)
                    skimage.io.imsave(save_path, img)
            else:
                y1, sr1 = librosa.load(file_path, duration=2.97)
                ps = librosa.feature.melspectrogram(y=y1, n_mels=128, hop_length=512)
                ps = np.log(ps + 1e-9)
                if ps.shape != (128, 128):
                    continue
                img = scale_minmax(ps, 0, 255).astype(np.uint8)
                img = np.flip(img, axis=0)
                img = 255-img 
                img_path = file.split('.wav')[0] + '_0' + '.png'
                save_path = os.path.join(save_dir, img_path)
                print('###', save_path)
                skimage.io.imsave(save_path, img)

            



def Noisy_signal(signal, snr_low=15, snr_high=30, nb_augmented=2):
    
    # Signal length
    signal_len = len(signal)

    # Generate White noise
    noise = np.random.normal(size=(nb_augmented, signal_len))
    
    # Compute signal and noise power
    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len
    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len
    
    # Random SNR: Uniform [15, 30]
    snr = np.random.randint(snr_low, snr_high)
    
    # Compute K coeff for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))
    K = np.ones((signal_len, nb_augmented)) * K
    
    # Generate noisy signal
    return signal + K.T * noise

def poly_learning_decay(optimizer, iter, total_epoch, loader_length, base_lr=None):
    max_iteration = total_epoch * loader_length
    learning_rate = base_lr * (1 - iter/max_iteration)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='../data/UrbanSound8K/metadata/UrbanSound8K.csv')
    parser.add_argument('--audio_root', type=str, default='../data/ravdess')
    parser.add_argument('--dataset', type=str, default='ravdess')
    args = parser.parse_args()
    if args.dataset == 'ravdess':
        prepare_ravdess_aug(args.audio_root)
    elif args.dataset == 'urbansound':
        prepare_urbansound(audio_root=args.audio_root, csv=args.csv)
    else:
        raise NotImplementedError