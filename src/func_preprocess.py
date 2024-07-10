import os
import tarfile
import numpy as np

from torchvision.datasets.utils import check_integrity
from typing import Optional
from PIL import Image


def unpickle(file):
    """
    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def extract_archive(
    archive_path: str,
    extract_root: str = None,
    md5: Optional[str] = None
) -> None:
    """
    Reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
    """
    # check md5 checksum of the .tar.gz file
    if not check_integrity(archive_path, md5):
        raise RuntimeError("Dataset metadata file not found or corrupted.")

    # for who wonders meaning of "archive":
    # https://en.wikipedia.org/wiki/Archive_file
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_root)
        print(f"Extracted {archive_path} to {extract_root}.")


def unpack_data_batch(
    source_root: str,
    save_root: str,
    filenames: list,
    classes: list,
    train_or_test: str
):
    """Unpack a batch file or a list of batch files and save images to save root

        Args:
            source_root (str): Path containing batch file(s)
            save_root (str): Path to save image (i.e., root of 'train' & 'test' directory)
            filenames (list): List of batch filename(s)
            classes (list): List of classes
        
        Return:
            (None)
    """
    for batch in filenames:
        batch_path = os.path.join(source_root, batch)
        batch_unpickled = unpickle(batch_path)

        batch_data = batch_unpickled[b'data']

        # convert dim (10000, 3072) -> (10000, 3, 32, 32) 
        batch_data = np.vstack(batch_data).reshape(-1, 3, 32, 32)
        
        # convert dim to HWC: (10000, 32, 32, 3)
        batch_data = batch_data.transpose((0, 2, 3, 1)) 

        batch_labels = batch_unpickled[b'labels']
        batch_filenames = batch_unpickled[b'filenames']
        
        # Save images in the batch file 
        for idx, _ in enumerate(batch_data):
            img = Image.fromarray(batch_data[idx])
            target = batch_labels[idx]
            class_name = classes[target]
            filename = batch_filenames[idx].decode('utf8')  # convert byte to string
            save_path = os.path.join(save_root, f'{train_or_test}/{class_name}/{filename}')
            img.save(save_path)
            