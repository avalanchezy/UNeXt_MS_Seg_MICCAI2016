# UNeXt

Official Pytorch Code base for [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967), MICCAI 2022

[Paper](https://arxiv.org/abs/2203.04967) | [Project](https://jeya-maria-jose.github.io/UNext-web/)

## Introduction

UNet and its latest extensions like TransUNet have been the leading medical image segmentation methods in recent years. However, these networks cannot be effectively adopted for rapid image segmentation in point-of-care applications as they are parameter-heavy, computationally complex and slow to use.  To this end, we propose UNeXt which is a Convolutional multilayer perceptron (MLP) based network for image segmentation. We design UNeXt in an effective way with an early convolutional stage and a MLP stage in the latent stage. We propose a tokenized MLP block where we efficiently tokenize and project the convolutional features and use MLPs to model the representation. To further boost the performance, we propose shifting the channels of the inputs while feeding in to MLPs so as to focus on learning local dependencies. Using tokenized MLPs in latent space reduces the number of parameters and computational complexity while being able to result in a better representation to help segmentation. The network also consists of skip connections between various levels of encoder and decoder.   We test UNeXt on multiple medical image segmentation datasets and show that we reduce the number of parameters by 72x, decrease the computational complexity by 68x, and improve the inference speed by 10x while also obtaining better segmentation performance over the  state-of-the-art medical image segmentation architectures.

<p align="center">
  <img src="imgs/unext.png" width="800"/>
</p>


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
[git clone https://github.com/jeya-maria-jose/UNeXt-pytorch
](https://github.com/avalanchezy/UNeXt_MS_Seg_MICCAI2016.git)cd UNeXt-pytorch
```

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate unext
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```

## Datasets

MICCAI 2016 chanllenge - [Link]([https://challenge.isic-archive.com/data/](https://portal.fli-iam.irisa.fr/msseg-challenge/english-msseg-data/))

```bash
python kfold.py
```


## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 1):

```
data/
├── data0/
│   ├── train/
│   │   ├── images/
│   │   └── masks/0/
│   ├── val/
│   │   ├── images/
│   │   └── masks/0/
│   └── test/
│       ├── images/
│       └── masks/0/
├── data1/
│   ├── train/
│   │   ├── images/
│   │   └── masks/0/
│   ├── val/
│   │   ├── images/
│   │   └── masks/0/
│   └── test/
│       ├── images/
│       └── masks/0/
├── data2/
│   ├── train/
│   │   ├── images/
│   │   └── masks/0/
│   ├── val/
│   │   ├── images/
│   │   └── masks/0/
│   └── test/
│       ├── images/
│       └── masks/0/
├── data3/
│   ├── train/
│   │   ├── images/
│   │   └── masks/0/
│   ├── val/
│   │   ├── images/
│   │   └── masks/0/
│   └── test/
│       ├── images/
│       └── masks/0/
└── data4/
    ├── train/
    │   ├── images/
    │   └── masks/0/
    ├── val/
    │   ├── images/
    │   └── masks/0/
    └── test/
        ├── images/
        └── masks/0/
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
python train.py --dataset <dataset name> --arch UNext --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 500 --input_w 256 --input_h 320 --b 8
```
2. Evaluate.
```
python testk.py
python predictk.py
python reconstructk.py
```

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet), [Segformer](https://github.com/NVlabs/SegFormer), and [AS-MLP](https://github.com/svip-lab/AS-MLP). Naming credits to [Poojan](https://scholar.google.co.in/citations?user=9dhBHuAAAAAJ&hl=en).

### Citation:
```
@article{valanarasu2022unext,
  title={UNeXt: MLP-based Rapid Medical Image Segmentation Network},
  author={Valanarasu, Jeya Maria Jose and Patel, Vishal M},
  journal={arXiv preprint arXiv:2203.04967},
  year={2022}
}
```
