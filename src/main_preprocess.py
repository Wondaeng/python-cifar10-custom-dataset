import os
from func_extract import unpickle, extract_archive, unpack_data_batch

"""
This script is written to show students:

1) How to extract .tar.gz compressed dataset
2) How checksum integrity is checked
3) How to process pickled data and save them as individual image

Assumptions:
- Python 3.5+ (for typing)
- File suffix is .tar.gz (e.g., CIFAR10)
"""

"""
There should be 5 train batch and 1 test batch.

Each batch file is a dictionary of following elements:

data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
"""

if __name__ == "__main__":
    """
    Step 1: Extract batch files from .tar.gz file 
    """
    archive_path = "../data/cifar-10/cifar-10-python.tar.gz"
    extract_root = "../data/cifar-10/"
    md5 = "c58f30108f718f92721af3b95e74349a"
    extract_archive(archive_path, extract_root=extract_root, md5=md5)

    """
    Step 2: Load and unpickle meta to parse class information
    """ 
    meta = unpickle("../data/cifar-10/cifar-10-batches-py/batches.meta")
    classes = meta[b"label_names"]  # list of classes: [b'airplane', b'automobile', ...]
    classes = [i.decode("utf-8") for i in classes]  # convert byte to string
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    """
    Step 3: Make directory for each class
    """ 
    train_dir = os.path.join(extract_root, 'train')
    test_dir = os.path.join(extract_root, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    for class_name in classes:
        os.mkdir(os.path.join(train_dir, class_name))
        os.mkdir(os.path.join(test_dir, class_name))

    """
    Step 4: Unpack train and test batch
    """
    source_root = os.path.join(extract_root, "cifar-10-batches-py")
    save_root = extract_root
    train_batches = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_batches =[
        "test_batch",
    ]

    print('Saving images from train batches ...')
    unpack_data_batch(source_root, save_root, train_batches, classes, "train")
    print('Saving images from test batches ...')
    unpack_data_batch(source_root, save_root, test_batches, classes, "test")
    