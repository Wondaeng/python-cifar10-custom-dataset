# Python Custom CIFAR10 Dataset

This repo contains two major steps:
1) Convert CIFAR10 archive to images into below structure:
```
.
└── cifar-10/
    ├── train/
    │   ├── airplane
    │   ├── automobile
    │   ├── bird
    │   └── ...
    └── test/
        ├── airplane
        ├── automobile
        ├── bird
        └── ...
```
2) Create custom Pytorch dataset using CIFAR10 images.
