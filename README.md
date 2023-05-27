# [CVPRW2023] "Leveraging Future Trajectory Prediction for Multi-Camera People Tracking"

[Track1: Multi-Camera People Tracking](https://www.aicitychallenge.org/2023-challenge-tracks/)

The official resitory for [7th NVIDIA AI City Challenge](https://www.aicitychallenge.org/)

## Pipeline Overview

<p align="center"><img src="assets/overview.png"/></p>

## Environment

We run on 2 NVIDIA A6000 GPUs.

- Linux or macOS
- Python 3.7+  (Python 3.8 in our envs)
- PyTorch 1.9+ (1.11.0 in our envs)
- CUDA 10.2+ (CUDA 11.3 in our envs)
- mmcv-full==1.7.1 ([MMCV](https://mmcv.readthedocs.io/en/latest/#installation))

## Installation

- Create an anaconda environment.

```bash 
sh setup.sh
```
## Train

## Inference

- Step #1. Single-Camera Tracking.

```bash 
sh run.sh
```

- Step #1. Multi-Camera Tracking (Association).
```bash 
sh run.sh
```


## Citation
```
@InProceedings{yuntae_2023_CVPR,
    author    = {Yuntae Jeon, Dai Quoc Tran, Minsoo Park and Seunghee Park},
    title     = {Leveraging Future Trajectory Prediction for Multi-Camera People Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
}
```
