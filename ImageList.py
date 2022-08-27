import os
import os.path
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from collections import Counter
from sklearn.model_selection import train_test_split


def stratify_sampling(image_list, ratio=1.0):
    '''stratify sampling a subset from the input dataset with a given ratio
    Args: 
        image_list: [list], each element is a str, "image_file_path label"
        num_labels_per_class: [int], number of labeled sample per class, if -1, then return a labeled dataset
    Returns:
        sampled_list: [list], the same structure as `image_list`
    '''
    assert(ratio > 0. and ratio <= 1.0)
    if ratio == 1.:
        return image_list
    images = [val.strip().split()[0] for val in image_list]
    labels = [int(val.strip().split()[1]) for val in image_list]
    assert(len(images) == len(labels))
    # print('image size={}, label size={}'.format(len(images),len(labels)))
    num_classes = len(np.unique(labels))
    labeled_images, _, labeled_y, _ = train_test_split(images, labels,
                                                       train_size=ratio, stratify=labels, random_state=1)
    return [image_name + " " + str(image_label) for image_name, image_label in zip(labeled_images, labeled_y)]


def make_dataset(image_list, labels, xclude_label=None):
    if labels is not None:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la)
                                                 for la in val.split()[1:]])) for val in image_list]
        else:
            if xclude_label is not None:
                images = [(val.split()[0], int(val.split()[1]))
                          for val in image_list if int(val.split()[1]) != xclude_label]
            else:
                images = [(val.split()[0], int(val.split()[1]))
                          for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, xclude_label=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels, xclude_label)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
