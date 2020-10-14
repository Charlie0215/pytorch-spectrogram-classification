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

def prepare_urbansound(
            csv='./data/UrbanSound8K/metadata/UrbanSound8K.csv', 
            audio_root='./data/UrbanSound8K', 
            save_dir='../data/urbansound_spectrogram'
    ):
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

def prepare_ravdess(audio_root, save_dir='../data/ravdess_spectrogram5'):
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
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

def prepare_ravdess_aug(audio_root, save_dir='../data'):
    
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    else:
        os.makedirs(save_folder)
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

def prepare_ravdess_more_features(audio_root, save_dir='../data/'):
    
    features = ['spectrogram', 'chromagram', 'mfcc', 'cqt', 'cens', 'dmfcc', 'ddmfcc']
    save_folder = [os.path.join(save_dir, '2d_feature', fea) for fea in features]
    for f in save_folder:
        if os.path.isdir(f):
            shutil.rmtree(f)
            os.makedirs(f)
        else:
            os.makedirs(f)
    
    def image_normalization(img):
        img = np.log(img + 1e-9)
        img = scale_minmax(img, 0, 255).astype(np.uint8)
        return img
    def mfcc_image_normalization(img):
        img = scale_minmax(img, 0, 255).astype(np.uint8)
        return img

    for folder in os.listdir(audio_root):
        folder_path = os.path.join(audio_root, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img_path = file.split('.wav')[0] + '.png'
            y1, sr1 = librosa.load(file_path, duration=2.97)
            
            # Generate spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y1, n_mels=128, hop_length=512)
            if spectrogram.shape != (128, 128):
                continue
            spc = image_normalization(spectrogram)
            spc = np.flip(spc, axis=0)
            spc = 255-spc 
            spc_save_path = os.path.join(save_folder[0], img_path)
            skimage.io.imsave(spc_save_path, spc)
            # Generate chromagram
            chroma_stft = librosa.feature.chroma_stft(y=y1, sr=sr1, n_chroma=12, n_fft=2048)
            chroma_stft = cv2.resize(chroma_stft, (128, 128))
            chroma_stft = image_normalization(chroma_stft)
            chroma_save_path = os.path.join(save_folder[1], img_path)
            skimage.io.imsave(chroma_save_path, chroma_stft)  
            # Generate Constant-Q Chroma
            chroma_cq = librosa.feature.chroma_cqt(y=y1, sr=sr1)
            chroma_cq = cv2.resize(chroma_cq, (128, 128))
            chroma_cq = image_normalization(chroma_cq)
            chroma_cq_save_path = os.path.join(save_folder[3], img_path)
            skimage.io.imsave(chroma_cq_save_path, chroma_cq) 
            # Generate Chroma energy normalized statistics
            chroma_cens = librosa.feature.chroma_cens(y=y1, sr=sr1)
            chroma_cens = cv2.resize(chroma_cens, (128, 128))
            chroma_cens = image_normalization(chroma_cens)
            chroma_cens_save_path = os.path.join(save_folder[4], img_path)
            skimage.io.imsave(chroma_cens_save_path, chroma_cens)  
            # Generate mfcc
            mfcc = librosa.feature.mfcc(S=spectrogram, n_mfcc=128)
            mfcc = mfcc_image_normalization(mfcc)
            mfcc_save_path = os.path.join(save_folder[2], img_path)
            skimage.io.imsave(mfcc_save_path, mfcc)
            # Generate delta mfcc
            dmfcc = librosa.feature.delta(mfcc)
            dmfcc = mfcc_image_normalization(dmfcc)
            dmfcc_save_path = os.path.join(save_folder[5], img_path)
            skimage.io.imsave(dmfcc_save_path, dmfcc)
            # Generate second order delta
            ddmfcc = librosa.feature.delta(mfcc, order=2)
            ddmfcc = mfcc_image_normalization(ddmfcc)
            ddmfcc_save_path = os.path.join(save_folder[6], img_path)
            skimage.io.imsave(ddmfcc_save_path, ddmfcc)
            print(spc_save_path)

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
        prepare_ravdess_more_features(args.audio_root)
    elif args.dataset == 'urbansound':
        prepare_urbansound(audio_root=args.audio_root, csv=args.csv)
    else:
        raise NotImplementedError
