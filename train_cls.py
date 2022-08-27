'''train a network. '''
import os
import pickle as pkl
import argparse
import time
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import torchvision.utils as tv_utils
from tensorboardX import SummaryWriter

import networks
import denoise_networks
from prep_data import get_dataset
from utils import EarlyStopping, CenterLoss, normalize_op


# create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


CLIP_GRAD_NORM = 5.
NUM_IMAGE_SAMPLES = 8


def train(args, model, center_criterion, device, train_loader, optimizer, epoch, writer=None):
    model.train()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        h, output = model(data, return_feat=True)
        loss_center = center_criterion(h, target)
        loss = F.cross_entropy(output, target) + \
            args.center_loss_weight * loss_center
        if torch.sum(torch.isnan(loss)) > 0:
            raise ValueError('nan loss')
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, loss_center: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_center.item()))
            if writer is not None:
                writer_step = batch_idx + \
                    int((epoch - 1) * len(train_loader.dataset) / args.batch_size)
                writer.add_scalar('loss', loss.item(), writer_step)
                writer.add_image('train_samples', tv_utils.make_grid(
                    data[:NUM_IMAGE_SAMPLES], normalize=True), writer_step)
    logger.info('train accuracy: {:.4f}'.format(100. * correct / total))
    return float(correct) / total


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            y_true.append(target.cpu())
            y_pred.append(pred.view(-1).cpu())

    test_loss /= len(test_loader.dataset)

    accuracy = correct / len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch classification example')
    parser.add_argument('--dataset', type=str, help='dataset',
                        choices=['mnist', 'usps', 'svhn', 'mnist_m', 'syn_digits',
                                 'imagenet32x32',
                                 'cifar10', 'stl10',
                                 'Art', 'Clipart', 'Product', 'Real_World',
                                 'real', 'sketch', 'quickdraw', 'clipart', 
                                 'amazon', 'webcam', 'dslr', 
                                 ])
    parser.add_argument('--arch', type=str, help='network architecture')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='sampling ratio of validation data')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='sampling ratio of training data')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-6,
                        help='weight_decay (default: 1e-6)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--center_loss_weight', type=float, default=0.,
                        help='weight of center loss (default: 0.)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str,
                        help='path to save ckpt and log. ')
    parser.add_argument('--resume', type=str,
                        help='resume training from ckpt path')
    parser.add_argument('--ckpt_path', type=str, help='init model from ckpt. ')
    parser.add_argument('--exclude_vars', type=str,
                        help='prefix of variables not restored form ckpt, seperated with commas; valid if ckpt_file is not None')
    parser.add_argument('--imagenet_pretrain', action='store_true',
                        help='use pretrained imagenet model (only works for VGG and ResNet)')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    writer = SummaryWriter(args.output_path)

    use_normalize = False
    model_normalize_op = None
    if args.dataset == "cifar100":
        n_classes = 100
    elif args.dataset == 'imagenet32x32':
        n_classes = 1000
        args.batch_size = 256
    elif args.dataset in ["cifar10", "stl10"]:
        # use_normalize = True
        n_classes = 9
    elif args.dataset in ["usps", "mnist", "svhn", 'mnist_m', 'syn_digits']:
        n_classes = 10
    elif args.dataset in ['Art', 'Clipart', 'Product', 'Real_World']:
        if args.arch.startswith('densenet') or args.arch == 'denoise_resnet152':
            args.batch_size = 32
        elif args.arch == 'denoise_resnext101':
            args.batch_size = 16
        else:
            args.batch_size = 64
        args.test_batch_size = 4
        n_classes = 65
        model_normalize_op = normalize_op
    elif args.dataset in ['real', 'sketch', 'quickdraw', 'clipart']:
        if args.arch.startswith('densenet') or args.arch == 'denoise_resnet152':
            args.batch_size = 32
        elif args.arch == 'denoise_resnext101':
            args.batch_size = 16
        else:
            args.batch_size = 64
        args.test_batch_size = 4
        n_classes = 345
        model_normalize_op = normalize_op
    elif args.dataset in ['amazon', 'webcam', 'dslr', ]:
        if args.arch.startswith('densenet') or args.arch == 'denoise_resnet152':
            args.batch_size = 32
        elif args.arch == 'denoise_resnext101':
            args.batch_size = 16
        else:
            args.batch_size = 64
        args.test_batch_size = 4
        n_classes = 31
        model_normalize_op = normalize_op
    else:
        raise ValueError('invalid dataset option: {}'.format(args.dataset))

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    assert(args.val_ratio >= 0. and args.val_ratio < 1.)
    assert(args.train_ratio > 0. and args.train_ratio <= 1.)
    train_ds = get_dataset(args.dataset, 'train', use_normalize=use_normalize,
                           test_size=args.val_ratio, train_size=args.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        get_dataset(args.dataset, 'test', use_normalize=use_normalize,
                    test_size=args.val_ratio, train_size=args.train_ratio),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.val_ratio == 0.0:
        val_loader = test_loader
    else:
        val_ds = get_dataset(args.dataset, 'val', use_normalize=use_normalize,
                             test_size=args.val_ratio, train_size=args.train_ratio)
        val_loader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.arch == "dtn":
        model = networks.DTN().to(device)
    elif args.arch == 'wrn':
        model = networks.WideResNet(
            depth=28, num_classes=n_classes, widen_factor=10, dropRate=0.0).to(device)
    elif args.arch.startswith('VGG'):
        model = networks.VGGFc(
            args.arch, use_bottleneck=False, new_cls=True, 
            class_num=n_classes, imagenet_pretrain=args.imagenet_pretrain, normalize_op=model_normalize_op).to(device)
    elif args.arch.startswith('ResNet'):
        model = networks.ResNetFc(
            args.arch, use_bottleneck=False, new_cls=True, 
            class_num=n_classes, imagenet_pretrain=args.imagenet_pretrain, normalize_op=model_normalize_op).to(device)
    elif args.arch.startswith('densenet'):
        model = networks.DenseNetFc(
            args.arch, use_bottleneck=False, new_cls=True, 
            class_num=n_classes, imagenet_pretrain=args.imagenet_pretrain, normalize_op=model_normalize_op).to(device)
    elif args.arch == 'mobilenet':
        model = networks.MobileNetFc(use_bottleneck=False, new_cls=True, 
            class_num=n_classes, imagenet_pretrain=args.imagenet_pretrain, normalize_op=model_normalize_op).to(device)
    elif args.arch.startswith('denoise'):
        model = denoise_networks.DenoiseResNet(
            args.arch, use_bottleneck=False, new_cls=True, 
            class_num=n_classes, imagenet_pretrain=args.imagenet_pretrain, normalize_op=model_normalize_op).to(device)
    else:
        raise ValueError('invalid network architecture {}'.format(args.arch))

    center_criterion = CenterLoss(n_classes, model.get_output_num(), use_cuda)

    if args.ckpt_path is not None:
        logger.info(
            'initialize model parameters from {}'.format(args.ckpt_path))
        model.restore_from_ckpt(torch.load(args.ckpt_path, map_location='cpu'),
                                exclude_vars=args.exclude_vars.split(',') if args.exclude_vars is not None else [])
        logger.info('accuracy on test set before fine-tuning')
        test(args, model, device, test_loader)

    if args.resume is not None:
        assert(os.path.isfile(args.resume))
        print('resume training from {}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if use_cuda:
        # model = torch.nn.DataParallel(model)
        if args.arch.startswith('denoise'):
            torch.backends.cudnn.enabled = False
        else:
            torch.backends.cudnn.enabled = True
            cudnn.benchmark = True

    if args.dataset.startswith("cifar") or args.dataset in ['stl10']:
        if args.ckpt_path is not None:
            lr_decay_step = 50
            lr_decay_rate = 0.5
        else:
            lr_decay_step = 100
            lr_decay_rate = 0.1
        PATIENCE = 20
        optimizer = optim.SGD(model.get_parameters(
            args.lr) + [{"params": center_criterion.parameters(), "lr": args.lr}], momentum=args.momentum, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    elif args.dataset in ['Art', 'Clipart', 'Product', 'Real_World', 
                          'real', 'sketch', 'quickdraw', 'clipart', 
                          'amazon', 'webcam', 'dslr']:
        if args.imagenet_pretrain or args.ckpt_path is not None:
            lr_decay_step = 10
        else:
            lr_decay_step = 25
        lr_decay_rate = 0.5
        if args.dataset in ['Art', 'Clipart', 'Product', 'Real_World']:
            PATIENCE = 20
        else:
            PATIENCE = 10
        optimizer = optim.SGD(model.get_parameters(
            args.lr) + [{"params": center_criterion.parameters(), "lr": args.lr}], momentum=args.momentum, weight_decay=args.wd)
        scheduler = StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    elif args.dataset in ["mnist", "usps", "svhn", "mnist_m", "syn_digits"]:
        lr_decay_step = 25
        lr_decay_rate = 0.5
        if args.dataset == 'svhn':
            PATIENCE = 10
        else:
            PATIENCE = 25
        optimizer = optim.SGD(model.get_parameters(
            args.lr) + [{"params": center_criterion.parameters(), "lr": args.lr}], momentum=0.5, weight_decay=args.wd)
        scheduler = StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    elif args.dataset == 'imagenet32x32':
        PATIENCE = 10
        lr_decay_step = 10
        lr_decay_rate = 0.2

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

        optimizer = torch.optim.SGD(
            model.get_parameters(args.lr) + [{"params": center_criterion.parameters(), "lr": args.lr}], momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda step: cosine_annealing(
        #         step,
        #         args.epochs * len(train_loader),
        #         1,  # since lr_lambda computes multiplicative factor
        #         1e-6 / args.lr))
    else:
        raise ValueError("invalid dataset option: {}".format(args.dataset))

    early_stop_engine = EarlyStopping(PATIENCE)

    logger.info("args:{}".format(args))

    # start training.
    best_accuracy = 0.
    save_path = os.path.join(args.output_path, "model.pt")
    time_stats = []
    train_accuracy = 0.
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_accuracy = train(args, model, center_criterion, device,
              train_loader, optimizer, epoch, writer)
        training_time = time.time() - start_time
        logger.info('epoch: {} training time: {:.2f}'.format(
            epoch, training_time))
        time_stats.append(training_time)

        val_accuracy, _ = test(args, model, device, val_loader)
        scheduler.step()

        writer.add_scalar("val_accuracy", val_accuracy, epoch)
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

        if epoch % 20 == 0:
            logger.info('accuracy on test set at epoch {}'.format(epoch))
            test(args, model, device, test_loader)

        if early_stop_engine.is_stop_training(val_accuracy):
            logger.info("no improvement after {}, stop training at epoch {}\n".format(
                PATIENCE, epoch))
            break

    # print('finish training {} epochs'.format(args.epochs))
    mean_training_time = np.mean(np.array(time_stats))
    logger.info('Average training_time: {}'.format(mean_training_time))
    logger.info(
        'load ckpt with best validation accuracy from {}'.format(save_path))
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    test_accuracy, _ = test(args, model, device, test_loader)

    writer.add_scalar("test_accuracy", test_accuracy, args.epochs)
    with open(os.path.join(args.output_path, 'accuracy.pkl'), 'wb') as pkl_file:
        pkl.dump({'train': train_accuracy, 'test': test_accuracy,
                  'training_time': mean_training_time}, pkl_file)


if __name__ == '__main__':
    main()
