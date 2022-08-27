import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm


class EarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
    """
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.meter = deque(maxlen=patience)

    def is_stop_training(self, score):
        stop_sign = False
        self.meter.append(score)
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                stop_sign = True
        # approximately equal
        elif np.abs(score - self.best_score) < 1e-9:
            if len(self.meter) == self.patience and np.abs(np.mean(self.meter) - score) < 1e-7:
                stop_sign = True
            else:
                self.best_score = score
                self.counter = 0
        # if score > best_score, update best_score and clear counter
        else:
            self.best_score = score
            self.counter = 0
        return stop_sign


class MarginLoss(nn.Module):

    def __init__(self, margin=1.0, target=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, logits, label):
        one_hot = torch.zeros_like(logits, dtype=torch.uint8)
        label = label.reshape(-1, 1)
        one_hot.scatter_(1, label, 1)
        diff = logits[one_hot.bool()] - torch.max(logits[~one_hot.bool()].view(len(logits), -1), dim=1)[0]
        if self.target:
            diff *= -1.
        margin = F.relu(diff + self.margin, True) - self.margin
        return margin.mean()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def one_hot_encoding(y, n_classes):
    '''convert torch.LongTensor into one-hot encodings'''
    # print('label size: {}'.format(y.size()))
    batch_size = y.size(0)
    ohe = torch.zeros(batch_size, n_classes).long().cuda()
    ohe.scatter_(1, y.view(batch_size, 1), 1)
    return ohe


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def extract_features(model, data_loader, device, feature_type='logits'):
    '''extract features of the penultimate layer of the model
    Args: 
        model: nn.Module
        data_loader
        feature_type: str, ["logits", "probs"]
    Return: 
        h_features: 2D Torch.FloatTensor, latent features
        labels: Torch.Tensor
        prediction_success: Torch.Tensor, 
        if predicted_label == ground_truth_label, prediction_success = 1
    '''
    h_features = []
    labels = []
    prediction_success = []

    for batch_idx, (data, label) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        label = label.to(device)
        labels.append(label.detach().cpu())

        h, logits = model(data, return_feat=True)
        if feature_type == 'logits':
            h_features.append(h.view(data.size(0), -1).detach().cpu())
        elif feature_type == 'probs':
            h_features.append(nn.Softmax(dim=1)(logits).detach().cpu())
        else:
            raise ValueError('not supported return feature type {}. '.format(feature_type))
        prediction_success.append((torch.argmax(logits, dim=1) == label).long().detach().cpu())

    labels = torch.cat(labels, axis=0)
    h_features = torch.cat(h_features, axis=0)
    prediction_success = torch.cat(prediction_success, axis=0)
    return h_features, labels, prediction_success


def get_centers(h, num_classes, labels):
    '''estimate centers of features
    Args: 
        h: torch.Tensor, latent features
        num_classes: int, 
    '''
    n_dim = h.size(1)
    centers = torch.zeros(num_classes, n_dim)
    for k in range(num_classes):
        indices = labels == k
        centers[k] = torch.mean(h[indices], axis=0)
    return centers


def distance_to_centroids(x, centroids):
    '''euclidean distance of a batch of samples to class centers
    Args:
        x: FloatTensor [batch_size, d]
        centroids: FloatTensor [K, d] where K is the number of classes
    Returns:
        dist: FloatTensor [batch_size, K]
    '''
    b, d = x.size()
    K, _ = centroids.size()
    dist = x.unsqueeze(1).expand(b, K, d) - centroids.unsqueeze(0).expand(b, K, d)
    return torch.norm(dist, dim=-1)


# transform op
def normalize_op(t, 
    mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    o = t.clone()
    o[:, 0, :, :] = (o[:, 0, :, :] - mean[0])/std[0]
    o[:, 1, :, :] = (o[:, 1, :, :] - mean[1])/std[1]
    o[:, 2, :, :] = (o[:, 2, :, :] - mean[2])/std[2]
    return o


def create_target_mapping(cm, attack_type):
    '''
    Args:
        cm: numpy.array, [n_classes, n_classes] shape
        attack_type: str, most/least common mistake prediction
    '''
    n_classes = cm.shape[0]
    mapping_tensor = -1 * torch.ones(n_classes).long()
    if attack_type == 'hard-targeted':
        MASK = 1e8
    else:
        MASK = -1e8
    for k in range(n_classes):
        # remove cm[k][k]
        l = cm[k].flatten().tolist()
        l[k] = MASK
        # return argmin if attack_type == 'hard-targeted'
        if attack_type == 'hard-targeted':
            mapping_tensor[k] = int(np.argmin(l))
        # return argmax if attack_type == 'easy-targeted'
        else:
            mapping_tensor[k] = int(np.argmax(l))
    return mapping_tensor.long()
