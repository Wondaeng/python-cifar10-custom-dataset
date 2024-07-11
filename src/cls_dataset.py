import os
import torch
import glob
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

def map_class_index(image_folder_path):
    """Get a dictionary mapping each class to (integer) index in alphabetical order.

    Args: 
        image_folder_path (str): path to 'train' OR 'test' folder
        e.g., ('../data/cifar-10/train/')
    
    Returns:
        (dict): mapping classes to indices, each class as a key 
    """
 
    # dataset: 'airplane', 'automobile', 'bird', ... (10 classes)
    classes = sorted(os.listdir(image_folder_path))
    class_to_index = dict()
    for idx, class_name in enumerate(classes):
        class_to_index[class_name] = idx
    return class_to_index


class CIFAR10(Dataset):
    def __init__(self, dataset_path, train_or_test='train'):
        """
        Args:
            dataset_path (str): path to image dataset (root) folder
            train_or_test (str): 'train' for training data, 'test' for testing data
        """
        image_folder_path = os.path.join(dataset_path, train_or_test)
        
        class_to_index = map_class_index(image_folder_path)
        
        self.image_list = []

        # EXPLANATION ABOUT LOGIC HERE
        # image_folder_path = '../data/cifar-10/test/'
        # in image_folder_path, 10 subfolders of 10 classes containing images
        # make a list of pair of image and label => [(image, label), ...]
        # 'image' is the full path (e.g., '../data/cifar-10/test/bird/pigeon_001.png')
        for class_name, class_index in class_to_index.items():
            class_folder_path = os.path.join(image_folder_path, class_name)
            for image in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image)
                self.image_list.append((image_path, class_index))
                
        self.data_len = len(self.image_list)

        if train_or_test == 'train':  
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            # transformation without randomness for evaluation
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        image_path, label = self.image_list[index]
        im_as_pil = Image.open(image_path)
        im_as_ten = self.transform(im_as_pil)
        return (im_as_ten, label)


    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    # sanity check:

    data_dir = '../data/cifar-10/'

    train_dataset = CIFAR10(dataset_path = data_dir, train_or_test='train')
    test_dataset = CIFAR10(dataset_path = data_dir, train_or_test='test')

    for i in test_dataset:
        print(i)