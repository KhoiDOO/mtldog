import os
import cv2
import h5py # require
import numpy as np
import argparse
import random

from alive_progress import alive_it

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dd', type=str, default='/media/mountHDD3/data_storage/nyu_full/nyu_depth_v2_labeled.mat',
                        help='data raw path, which is downloaded at https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html')
    parser.add_argument('--sd', type=str, default='/media/mountHDD3/data_storage/nyu_full',
                        help='extracted data save path')
    parser.add_argument('--seed', type=int, help='seed number')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio used for test set')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.sd + f"/preprocess_ratio_{args.ratio}_{args.seed}"
    
    train_save_dir = save_dir + "/train"
    valid_save_dir = save_dir + "/valid"
    for dir in [save_dir, train_save_dir, valid_save_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    for par_dir in [train_save_dir, valid_save_dir]:
        for sub_dir in ['image', 'label', 'depth', 'instances']:
            dir_path = par_dir + f"/{sub_dir}"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        

    with h5py.File(args.dd, 'r') as fr:
        images = np.array(fr['images']).transpose(0, 3, 2, 1)
        labels = np.array(fr["labels"]).transpose([0, 2, 1])
        depths = np.array(fr["depths"]).transpose([0, 2, 1])
        instas = np.array(fr["instances"]).transpose([0, 2, 1])

        assert images.shape[:-1] == labels.shape == depths.shape == instas.shape

        idxs = np.arange(len(images))
        train_idxs = np.random.choice(idxs, size=int(len(images) * (1 - args.ratio)), replace=False)
        valid_idxs = np.setdiff1d(idxs, train_idxs)

        train_images = images[train_idxs]
        train_labels = labels[train_idxs]
        train_depths = depths[train_idxs]
        train_instas = instas[train_idxs]

        valid_images = images[valid_idxs]
        valid_labels = labels[valid_idxs]
        valid_depths = depths[valid_idxs]
        valid_instas = instas[valid_idxs]

        print("Train Preprocessing")
        for idx, (img, lbl, dep, ins) in alive_it(enumerate(zip(train_images, train_labels, train_depths, train_instas))):
            np.savez_compressed(train_save_dir + f"/image/{idx}", data = img)
            np.savez_compressed(train_save_dir + f"/label/{idx}", data = lbl)
            np.savez_compressed(train_save_dir + f"/depth/{idx}", data = dep)
            np.savez_compressed(train_save_dir + f"/instances/{idx}", data = ins)
        
        print("Valid Preprocessing")
        for idx, (img, lbl, dep, ins) in alive_it(enumerate(zip(valid_images, valid_labels, valid_depths, valid_instas))):
            np.savez_compressed(valid_save_dir + f"/image/{idx}", data = img)
            np.savez_compressed(valid_save_dir + f"/label/{idx}", data = lbl)
            np.savez_compressed(valid_save_dir + f"/depth/{idx}", data = dep)
            np.savez_compressed(valid_save_dir + f"/instances/{idx}", data = ins)