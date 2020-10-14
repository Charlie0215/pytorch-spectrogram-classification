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

def prepare_ravdess_adv(audio_root):
    save_dir = '../data/ravdess_spectrogram4'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for folder in os.listdir(audio_root):
        folder_path = os.path.join(audio_root, folder)
        for file in os.listdir(folder_path):
            folder_id = file.split('-')[-1].split('.wav')[0]
            file_path = os.path.join(folder_path, file)
            
            if int(folder_id) in [11, 12, 23, 24]:
                y1, sr1 = librosa.load(file_path, duration=2.97)
                ps = librosa.feature.melspectrogram(y=y1, n_mels=128, hop_length=512)#1126)
                ps = np.log(ps + 1e-9)                

                img = scale_minmax(ps, 0, 255).astype(np.uint8)
                img = np.flip(img, axis=0)
                img = 255-img 
                img_path = file.split('.wav')[0] + '_0' + '.png'
                print('#########', folder_id, img_path, img.shape)
                save_path = os.path.join(save_dir, img_path)
                skimage.io.imsave(save_path, img)
            else:
                y1, sr1 = librosa.load(file_path, sr=None)
                y1 = Noisy_signal(y1)
                ps = librosa.feature.melspectrogram(y=y1, n_mels=128, hop_length=256)#1126)
                ps = np.log(ps + 1e-9)
                
                img = scale_minmax(ps, 0, 255).astype(np.uint8)
                img = np.flip(img, axis=0)
                img = 255-img 
                for i in range(5):
                    if i == 0: continue
                    start_point = i * 128
                    if start_point < img.shape[1]:
                        if start_point + 128 < img.shape[1]:
                            save_image = img[:, start_point:start_point+128]
                            img_path = file.split('.wav')[0] + '_{}'.format(i) + '.png'
                            save_path = os.path.join(save_dir, img_path)
                            print(save_image.shape)
                        else:
                            pad_size = start_point + 128 - img.shape[1]
                            pad = np.zeros([128, pad_size])
                            save_image = img[:, start_point:img.shape[1]] 
                            save_image = np.concatenate((save_image, pad), axis=1)
                            print(save_image.shape[1]+pad_size+1, save_image.shape[1])
                            img_path = file.split('.wav')[0] + '_{}'.format(i) + '.png'
                            save_path = os.path.join(save_dir, img_path)
                            print(save_image.shape)
                        assert save_image.shape == (128, 128)
                        skimage.io.imsave(save_path, save_image)
                    else: continue
