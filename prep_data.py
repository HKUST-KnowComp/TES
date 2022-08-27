import gzip
import pickle as pkl
import errno
import os
import logging
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision.datasets.utils import check_integrity
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from scipy.io import loadmat
from ImageList import stratify_sampling, ImageList
from imagenet_downsampled import ImageNetDS


DATA_PATH = "/home/data/yzhangdx/dataset"
RANDOM_STATE = 1234


# create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_dataset(dataname, data_split='train', use_normalize=False, test_size=0.1, train_size=None, xclude_label=None):
    '''
    Args:
        dataname: str
        data_split: str, 'train', 'val' or 'test'
        use_normalize: boolean. normalize data to unit gaussian or not. 
        test_size: float, train/val ratio
        xclude_label: int, (x, y) pairs where y==xclude_label are excluded
        TODO: xclude_label for svhn, not implemented
    '''
    # if `train_size` is left as None, set `train_size` with `test_size`
    if train_size is None and test_size is not None:
        train_size = 1. - test_size
    elif test_size == 0. and train_size is not None and train_size < 1.:
        test_size = 1 - train_size  # dump
    assert(test_size + train_size <= 1.)
    transform_ops = [transforms.ToTensor(), ]
    if dataname == "mnist":
        transform_ops += [transforms.Lambda(lambda x: F.pad(x, (2, 2, 2, 2), 'constant', 0)),
                          transforms.Lambda(lambda x: x.expand(3, -1, -1))
                          ]
        if use_normalize:
            # transform_ops.append(transforms.Normalize((0.1307,), (0.3081,)))
            # transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5,
            # 0.5, 0.5))) # rescale to [-1,1]
            transform_ops.append(transforms.Normalize(
                (0.5, ), (0.5, )))  # rescale to [-1,1]
        logger.info('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            if xclude_label is None:
                return datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=False, download=True,
                                      transform=transforms.Compose(transform_ops))
            else:
                test_ds = datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=False, download=True,
                                      transform=transforms.Compose(transform_ops))
                sel_indices = torch.nonzero(test_ds.targets != xclude_label).view(-1)
                return torch.utils.data.Subset(test_ds, sel_indices)
        else:
            train_ds = datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=True, download=True,
                                      transform=transforms.Compose(transform_ops))
            if xclude_label is not None:
                sel_indices = torch.nonzero(train_ds.targets != xclude_label).view(-1)
                train_ds = torch.utils.data.Subset(train_ds, sel_indices)
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(
                skf.split(train_ds.data, train_ds.targets))[0]
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == "usps":
        usps = pkl.load(gzip.open(os.path.join(
            DATA_PATH, 'usps/usps_28x28.pkl'), "rb"), encoding="latin1")
        if data_split in ["train", "val"]:
            # 7438, 1, 28, 28
            images = torch.from_numpy(usps[0][0])
            # 7438x[0~9]
            labels = torch.from_numpy(usps[0][1]).long()
        else:  # 1860
            images = torch.from_numpy(usps[1][0])
            labels = torch.from_numpy(usps[1][1]).long()
        images = F.pad(images, (2, 2, 2, 2))
        images = images.expand(-1, 3, -1, -1)
        if use_normalize:
            # images = (images - 0.1608) / 0.2578
            # images = (images - 0.1231) / 0.2357
            images = (images - 0.5) / 0.5
        if xclude_label is not None:
            sel_indices = labels != xclude_label
            images = images[sel_indices]
            labels = labels[sel_indices]
        if data_split == "test":
            return torch.utils.data.TensorDataset(images, labels)
        else:
            if data_split == 'train' and train_size == 1.:
                return torch.utils.data.TensorDataset(images, labels)
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(labels)), train_size=int(train_size * len(labels)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(images, labels))[0]
            if data_split == 'train':
                return torch.utils.data.TensorDataset(images[train_index], labels[train_index])
            else:
                return torch.utils.data.TensorDataset(images[val_index], labels[val_index])
    elif dataname == "svhn":
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if data_split == "test":
            return datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split=data_split, download=True,
                                 transform=transforms.Compose(transform_ops))
        else:
            # always use the extra set
            train_ds = torch.utils.data.ConcatDataset([
                datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='train', download=True,
                              transform=transforms.Compose(transform_ops)),
                datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='extra', download=True,
                              transform=transforms.Compose(transform_ops)),
            ])
            # train_ds = datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='train', download=True,
            #                transform=transforms.Compose(transform_ops))
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(
                np.concatenate([ds.data for ds in train_ds.datasets], 0),
                np.concatenate([ds.labels for ds in train_ds.datasets], 0)
            ))[0]
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == "mnist_m":
        # resize from [28,28] to [32,32]
        transform_ops = [transforms.Resize(32),
                         transforms.ToTensor()]
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # rescale to [-1,1]
        if data_split == "test":
            if xclude_label is None:
                return MNISTM(os.path.join(DATA_PATH, 'mnist_m'), train=False, download=True,
                          transform=transforms.Compose(transform_ops))
            else:
                test_ds = MNISTM(os.path.join(DATA_PATH, 'mnist_m'), train=False, download=True,
                            transform=transforms.Compose(transform_ops))
                sel_indices = torch.nonzero(test_ds.test_labels != xclude_label).view(-1)
                return torch.utils.data.Subset(test_ds, sel_indices)
        else:
            train_ds = MNISTM(os.path.join(DATA_PATH, 'mnist_m'), train=True, download=True,
                              transform=transforms.Compose(transform_ops))
            if xclude_label is not None:
                sel_indices = torch.nonzero(train_ds.train_labels != xclude_label).view(-1)
                train_ds = torch.utils.data.Subset(train_ds, sel_indices)
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(
                skf.split(train_ds.train_data, train_ds.train_labels))[0]
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == "syn_digits":
        mat_data = loadmat(os.path.join(
            DATA_PATH, 'SynthDigits', 'synth_{}_32x32.mat'.format(data_split)))
        images = torch.from_numpy(mat_data['X'] / 255.).float()
        labels = torch.from_numpy(mat_data['y']).long().view(-1)
        images = images.permute(3, 2, 0, 1)
        if use_normalize:
            images = (images - 0.5) / 0.5
        if data_split == "test":
            return torch.utils.data.TensorDataset(images, labels)
        else:
            if data_split == 'train' and train_size == 1.:
                return torch.utils.data.TensorDataset(images, labels)
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(labels)), train_size=int(train_size * len(labels)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(images, labels))[0]
            if data_split == 'train':
                return torch.utils.data.TensorDataset(images[train_index], labels[train_index])
            else:
                return torch.utils.data.TensorDataset(images[val_index], labels[val_index])
    elif dataname.startswith("cifar"):
        CIFAR_DATASETS = {"cifar10": datasets.CIFAR10,
                          "cifar100": datasets.CIFAR100}
        tv_cifar = CIFAR_DATASETS[dataname]
        if data_split == "train":
            transform_ops = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             ]
        if use_normalize:
            # transform_ops.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                                           (0.2023, 0.1994, 0.2010)))
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)))
        logger.info('transforms for {} set: {}'.format(data_split, transform_ops))
        # Remap class indices so that the frog class (6) has an index of -1 as
        # it does not appear int the STL dataset
        cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
        if data_split == "test":
            test_ds = tv_cifar(os.path.join(DATA_PATH, 'cifar'), train=False, download=True,
                               transform=transforms.Compose(transform_ops))
            if dataname == 'cifar10':
                test_ds.targets = cls_mapping[test_ds.targets]
                valid_indices = np.all([test_ds.targets > -1, test_ds.targets != xclude_label], axis=0)
                test_ds.data, test_ds.targets = test_ds.data[
                    valid_indices], test_ds.targets[valid_indices]
                return test_ds
            else:
                return test_ds
        else:
            train_ds = tv_cifar(os.path.join(DATA_PATH, 'cifar'), train=True, download=True,
                                transform=transforms.Compose(transform_ops))
            if dataname == 'cifar10':
                train_ds.targets = cls_mapping[train_ds.targets]
                valid_indices = np.all([train_ds.targets > -1, train_ds.targets != xclude_label], axis=0)
                train_ds.data, train_ds.targets = train_ds.data[
                    valid_indices], train_ds.targets[valid_indices]
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(
                skf.split(train_ds.data, train_ds.targets))[0]
            # logger.info('training set of cifar: {} samples. '.format(len(train_ds)))
            # logger.info('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == 'stl10':
        '''cifar-10-stl-10 transfer task
           align two label spaces by removing "monkey" class in stl
        '''
        if data_split in ["train", 'val']:
            transform_ops = [transforms.Resize(32),
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             ]
        else:
            transform_ops = [transforms.Resize(32),
                             transforms.ToTensor(),
                             ]
        if use_normalize:
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)))
        logger.info('transforms for {} set: {}'.format(data_split, transform_ops))
        # remap class indices to match cifar-10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
        if data_split == "test":
            stl_test_ds = datasets.STL10(os.path.join(DATA_PATH, 'stl10'), split='test', download=True,
                                         transform=transforms.Compose(transform_ops))
            stl_test_ds.labels = cls_mapping[stl_test_ds.labels]
            valid_indices = np.all([stl_test_ds.labels > -1, stl_test_ds.labels != xclude_label], axis=0)
            stl_test_ds.data, stl_test_ds.labels = stl_test_ds.data[
                valid_indices], stl_test_ds.labels[valid_indices]
            return stl_test_ds
        else:
            train_ds = datasets.STL10(os.path.join(DATA_PATH, 'stl10'), split='train', download=True,
                                      transform=transforms.Compose(transform_ops))
            train_ds.labels = cls_mapping[train_ds.labels]
            valid_indices = np.all([train_ds.labels > -1, train_ds.labels != xclude_label], axis=0)
            train_ds.data, train_ds.labels = train_ds.data[
                valid_indices], train_ds.labels[valid_indices]
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(
                skf.split(train_ds.data, train_ds.targets))[0]
            # logger.info('training set of cifar: {} samples. '.format(len(train_ds)))
            # logger.info('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == 'imagenet32x32':
        '''imagenet 32x32 as source domain
        '''
        if data_split in ["train", 'val']:
            transform_ops = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             ]
        else:
            transform_ops = [transforms.ToTensor(),
                             ]
        if use_normalize:
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)))
        logger.info('transforms for {} set: {}'.format(data_split, transform_ops))
        if data_split == "test":
            return ImageNetDS(os.path.join(DATA_PATH, 'imagenet32x32'), 32, train=False,
                              transform=transforms.Compose(transform_ops))
        else:
            train_ds = ImageNetDS(os.path.join(DATA_PATH, 'imagenet32x32'), 32, train=True,
                                  transform=transforms.Compose(transform_ops))
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(
                test_size * len(train_ds)), train_size=int(train_size * len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(
                skf.split(train_ds.train_data, train_ds.train_labels))[0]
            # logger.info('training set of cifar: {} samples. '.format(len(train_ds)))
            # logger.info('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname in ['Art', 'Clipart', 'Product', 'Real_World']:
        transform_ops = [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # imagenet default normalization
        logger.info('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            return ImageList(open(os.path.join('./datasets/office-home/{}_test.txt'.format(dataname)), 'r').readlines(), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        if data_split == 'train':
            return ImageList(stratify_sampling(open('./datasets/office-home/{}_train.txt'.format(dataname), 'r').readlines(), ratio=train_size), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
    elif dataname in ['real', 'sketch', 'quickdraw', 'clipart']:
        transform_ops = [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # imagenet default normalization
        logger.info('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            return ImageList(open('./datasets/domain-net/{}_test.txt'.format(dataname), 'r').readlines(), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        elif data_split == 'attack_test':
            return ImageList(open('./datasets/domain-net/{}_attack_test.txt'.format(dataname), 'r').readlines(), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        elif data_split == 'train':
            return ImageList(stratify_sampling(open('./datasets/domain-net/{}_train.txt'.format(dataname), 'r').readlines(), ratio=train_size), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        else:
            raise ValueError('invalid split: {}'.format(data_split))
    elif dataname in ['amazon', 'webcam', 'dslr']:
        transform_ops = [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # imagenet default normalization
        logger.info('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            return ImageList(open('./datasets/office31/{}_test.txt'.format(dataname), 'r').readlines(), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        elif data_split == 'train':
            return ImageList(stratify_sampling(open('./datasets/office31/{}_train.txt'.format(dataname), 'r').readlines(), ratio=train_size), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        else:
            raise ValueError('invalid split: {}'.format(data_split))
    elif dataname == 'imagenet-val':
        transform_ops = [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                        ]
        if use_normalize:
            transform_ops.append(transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # imagenet default normalization
        logger.info('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            return ImageList(open(os.path.join('./datasets/imagenet-val/sample_1k.txt'), 'r').readlines(), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
        if data_split == 'train':
            return ImageList(stratify_sampling(open('./datasets/imagenet-val/sample_49k.txt', 'r').readlines(), ratio=train_size), 
                             xclude_label=xclude_label, transform=transforms.Compose(transform_ops))
    else:
        raise ValueError('no supported loader for dataset {}'.format(dataname))


"""Dataset setting and data loader for MNIST-M.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
CREDIT: https://github.com/corenel
"""
class MNISTM(torch.utils.data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            )
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        logger.info("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        logger.info("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root, train=True, download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root, train=False, download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        logger.info("Done!")

