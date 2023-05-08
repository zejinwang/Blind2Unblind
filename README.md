# Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots
[Blind2Unblind](https://arxiv.org/abs/2203.06967)

## Citing Blind2Unblind
```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zejin and Liu, Jiazheng and Li, Guoqing and Han, Hua},
    title     = {Blind2Unblind: Self-Supervised Image Denoising With Visible Blind Spots},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2027-2036}
}
```
## Installation
The model is built in Python3.8.5, PyTorch 1.7.1 in Ubuntu 18.04 environment.

## Data Preparation

### 1. Prepare Training Dataset

- For processing ImageNet Validation, please run the command

  ```shell
  python ./dataset_tool.py
  ```

- For processing SIDD Medium Dataset in raw-RGB, please run the command

  ```shell
  python ./dataset_tool_raw.py
  ```

### 2. Prepare Validation Dataset

​	Please put your dataset under the path: **./Blind2Unblind/data/validation**.

## Pretrained Models
Download pre-trained models: [Google Drive](https://drive.google.com/drive/folders/1ruA6-SN1cyf30-GHS8w2YD1FG-0A-k7h?usp=sharing) 

The pre-trained models are placed in the folder: **./Blind2Unblind/pretrained_models**

```yaml
# # For synthetic denoising
# gauss25
./pretrained_models/g25_112f20_beta19.7.pth
# gauss5_50
./pretrained_models/g5-50_112rf20_beta19.4.pth
# poisson30
./pretrained_models/p30_112f20_beta19.1.pth
# poisson5_50
./pretrained_models/p5-50_112rf20_beta20.pth

# # For raw-RGB denoising
./pretrained_models/rawRGB_112rf20_beta19.4.pth

# # For fluorescence microscopy denoising
# Confocal_FISH
./pretrained_models/Confocal_FISH_112rf20_beta20.pth
# Confocal_MICE
./pretrained_models/Confocal_MICE_112rf20_beta19.7.pth
# TwoPhoton_MICE
./pretrained_models/TwoPhoton_MICE_112rf20_beta20.pth
```

## Train
* Train on synthetic dataset
```shell
python train_b2u.py --noisetype gauss25 --data_dir ./data/train/Imagenet_val --val_dirs ./data/validation --save_model_path ../experiments/results --log_name b2u_unet_gauss25_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```
* Train on SIDD raw-RGB Medium dataset
```shell
python train_sidd_b2u.py --data_dir ./data/train/SIDD_Medium_Raw_noisy_sub512 --val_dirs ./data/validation --save_model_path ../experiments/results --log_name b2u_unet_raw_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```
* Train on FMDD dataset
```shell
python train_fmdd_b2u.py --data_dir ./dataset/fmdd_sub/train --val_dirs ./dataset/fmdd_sub/validation --subfold Confocal_FISH --save_model_path ../experiments/fmdd --log_name Confocal_FISH_b2u_unet_fmdd_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```

## Test

* Test on **Kodak, BSD300 and Set14**

  * For noisetype: gauss25

    ```shell
    python test_b2u.py --noisetype gauss25 --checkpoint ./pretrained_models/g25_112f20_beta19.7.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g25_112rf20 --beta 19.7
    ```

  * For noisetype: gauss5_50

    ```shell
    python test_b2u.py --noisetype gauss5_50 --checkpoint ./pretrained_models/g5-50_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g5_50_112rf20 --beta 19.4
    ```

  * For noisetype: poisson30

    ```shell
    python test_b2u.py --noisetype poisson30 --checkpoint ./pretrained_models/p30_112f20_beta19.1.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_p30_112rf20 --beta 19.1
    ```

  * For noisetype: poisson5_50

    ```shell
    python test_b2u.py --noisetype poisson5_50 --checkpoint ./pretrained_models/p5-50_112rf20_beta20.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_p5_50_112rf20 --beta 20.0
    ```

* Test on **SIDD Validation** in raw-RGB space

```shell
python test_sidd_b2u.py --checkpoint ./pretrained_models/rawRGB_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name validation_b2u_unet_raw_112rf20 --beta 19.4
```

* Test on **SIDD Benchmark** in raw-RGB space

```shell
python benchmark_sidd_b2u.py --checkpoint ./pretrained_models/rawRGB_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name benchmark_b2u_unet_raw_112rf20 --beta 19.4
```

* Test on **FMDD Validation**

  *  For Confocal_FISH

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/Confocal_FISH_112rf20_beta20.pth --test_dirs ./dataset/fmdd_sub/validation --subfold Confocal_FISH --save_test_path ./test --log_name Confocal_FISH_b2u_unet_fmdd_112rf20 --beta 20.0
    ```

  *  For Confocal_MICE

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/Confocal_MICE_112rf20_beta19.7.pth --test_dirs ./dataset/fmdd_sub/validation --subfold Confocal_MICE --save_test_path ./test --log_name Confocal_MICE_b2u_unet_fmdd_112rf20 --beta 19.7
    ```

  *  For TwoPhoton_MICE

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/TwoPhoton_MICE_112rf20_beta20.pth --test_dirs ./dataset/fmdd_sub/validation --subfold TwoPhoton_MICE --save_test_path ./test --log_name TwoPhoton_MICE_b2u_unet_fmdd_112rf20 --beta 20.0
    ```
