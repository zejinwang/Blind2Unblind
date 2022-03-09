import os
from glob import glob
import numpy as np
from scipy.io import loadmat, savemat
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm


data_dir = "/YourPath/SIDD_Medium_Raw/Data"
path_all_noisy = glob(os.path.join(data_dir, '**/*NOISY*.MAT'), recursive=True) # GT or NOISY
path_all_noisy = sorted(path_all_noisy)
print('Number of big images: {:d}'.format(len(path_all_noisy)))

save_folder = "./data/train/SIDD_Medium_Raw_noisy_sub512" # gt or noisy
if os.path.exists(save_folder):
    os.system("rm -r {}".format(save_folder))
os.makedirs(save_folder)   

crop_size = 512
step = 256

def pipline(ii):
    img_name, extension = os.path.splitext(os.path.basename(path_all_noisy[ii]))
    print(img_name)
    mat = h5py.File(path_all_noisy[ii])
    # im = mat['x'].value
    im = mat['x']
    h, w = im.shape
    # prepare to crop
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > 0:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > 0:
        w_space = np.append(w_space, w - crop_size)
    # crop
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = im[x:x + crop_size, y:y + crop_size]
            cropped_img = np.ascontiguousarray(cropped_img)
            save_path = os.path.join(save_folder, "{}_s{:0>3d}{}".format(img_name, index, extension.lower()))
            savemat(save_path, {"x": cropped_img})

Parallel(n_jobs=10)(delayed(pipline)(i) for i in tqdm(range(len(path_all_noisy))))