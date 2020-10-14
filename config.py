import os
class urbansound_train_config():
    epoch = 300
    lr = 0.0001
    batch_size = 32
    pretrain = False
    if_augment = True
    weight_save_dir = './weight'
    if not os.path.exists(weight_save_dir):
        os.makedirs(weight_save_dir)
    if if_augment:
        weight_load_dir = './weight/best_aug_weight.pkl'
    else: weight_load_dir = './weight/best_weight.pkl'
    test_folder = '1'


class ravdess_train_config():
    epoch = 1000
    lr = 0.0001
    batch_size = 16
    pretrain = False
    if_augment = False
    weight_save_dir = './weight'
    if not os.path.exists(weight_save_dir):
        os.makedirs(weight_save_dir)
    if if_augment:
        weight_load_dir = './weight/best_aug_weight.pkl'
    else: weight_load_dir = './weight/best_weight.pkl'
    test_folder = '1'