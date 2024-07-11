# Python Custom CIFAR10 Dataset

## What is this repository about?
This repo contains python scripts for creating custom Pytorch dataset from scratch (without using torchvision.datasets.CIFAR10 class) for AI summer school 2024.
Students can learn how to preprocess raw dataset from the public online source and how to organize them nicely.

It mainly focuses on:

1) Converting CIFAR10 archive to images into below structure:
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
2) Creating custom Pytorch dataset using those saved images, rather than using pre-defined Pytorch CIFAR10 class:

```python
class CIFAR10(Dataset):
    def __init__(self, dataset_path):
        ...

    def __getitem__(self, index):
        ...
        return (image, label)


    def __len__(self):
        ...
        return data_length
```

If there is any error, please let us know. 

