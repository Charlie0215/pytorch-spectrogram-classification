import os


class urbansound_train_config:
    epoch = 300
    lr = 0.0001
    batch_size = 16
    pretrain = False
    if_augment = True
    weight_save_dir = "./weight"
    if not os.path.exists(weight_save_dir):
        os.makedirs(weight_save_dir)
    if if_augment:
        weight_loader_dir = "./weight/best_aug_weight.pkl"
    else:
        weight_loader_dir = "./weight/best_weight.pkl"
    test_folder = "1"


class ravdess_train_config:
    epoch = 1000
    lr = 0.00001
    batch_size = 8
    pretrain = False
    if_augment = False
    # dic to save weight
    weight_save_dir = "./weight"
    if not os.path.exists(weight_save_dir):
        os.makedirs(weight_save_dir)

    # the best pretrained weight location
    weight_loader_dir = "./weight/best_aug_weight.pkl"

    # the testing folder ids
    test_folder = [11, 12, 23, 24]
