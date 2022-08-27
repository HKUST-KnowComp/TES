import argparse
import random
import os
import json
import logging
import time
import pickle as pkl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.utils as tv_utils
from tensorboardX import SummaryWriter

from prep_data import get_dataset
from utils import MarginLoss, EarlyStopping, normalize_op, create_target_mapping
import netG
import networks
import denoise_networks
from train_cls import test


# create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# consts
BBOX_MIN, BBOX_MAX = 0., 1.


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train an adversarial generator')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'svhn', 'usps', 'mnist_m', 
                                 'cifar10', 'stl10', 
                                 'Art', 'Clipart', 'Product', 'Real_World', 
                                 'real', 'sketch', 'quickdraw', 
                                 'amazon', 'webcam', 'dslr', 
                                 'imagenet-val',
                                 ],
                        help='dataset')
    parser.add_argument('--arch', type=str, default='dtn', 
                        help='net arch of the target of attack')
    parser.add_argument('--ckpt_path', type=str,
                        help='path to load weights of the target of attack. ')
    # AG params
    parser.add_argument('--ngf', type=int, default=64,
                        help='number of filters in AG (default: 64)')
    parser.add_argument('--g_arch', type=str, choices=[
                        'basic', 'res', 'conv'], default='basic', help='net arch of the adversarial generator')
    parser.add_argument('--g_loss', type=str, choices=[
                        'margin', 'relativistic'], default='margin', help='loss of the adversarial generator')
    parser.add_argument('--margin_thresh', type=float, default=200.,
                        help='threshold of the margin loss, only effective if g_loss==margin. '
                             'The higher, the more transferable adversarial examples (default: 200)')
    # optim params
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='sampling ratio of training data')
    # attack params
    parser.add_argument('--attack_type', type=str, default='untargeted', 
                        choices=['untargeted', 'random-targeted', 'easy-targeted', 'hard-targeted', 'class-targeted'], 
                        help='attack type')
    parser.add_argument('--target_label', type=int, 
                        help='label of class-targeted attacks')
    parser.add_argument('--eps', type=int, default=32,
                        help='epsilon bound of attack')
    parser.add_argument('--output_path', type=str,
                        default='/home/data/yzhangdx/workspace/AGT',
                        help='path to save results')
    args = parser.parse_args()

    # set env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = True
    torch.manual_seed(1)
    random.seed(1)
    log_interval = 50
    patience = 10
    num_image_samples = 8
    early_stop_engine = EarlyStopping(patience)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path)

    # load dataset
    args.batch_size = 128
    args.test_batch_size = 4
    eps = args.eps / 255.
    use_normalize = False
    if args.dataset in ["cifar10", "stl10"]:
        n_classes = 9
    elif args.dataset in ["usps", "mnist", "svhn", "mnist_m"]:
        n_classes = 10
    elif args.dataset in ['Art', 'Clipart', 'Product', 'Real_World', ]:
        args.batch_size = 64
        n_classes = 65
    elif args.dataset in ['real', 'sketch', 'quickdraw']:
        args.batch_size = 64
        n_classes = 345
    elif args.dataset in ['amazon', 'webcam', 'dslr', ]:
        args.batch_size = 64
        n_classes = 31
    elif args.dataset == 'imagenet-val':
        args.batch_size = 64
        n_classes = 1000
        early_stop_engine.patience = 25
    else:
        raise ValueError('invalid dataset option: {}'.format(args.dataset))

    # nets are the model to attack
    if args.arch == 'dtn':
        target_model = networks.DTN().to(device)
    elif args.arch == 'wrn':
        target_model = networks.WideResNet(
            depth=28, num_classes=n_classes, widen_factor=10, dropRate=0.0).to(device)
    elif args.arch.startswith('VGG'):
        target_model = networks.VGGFc(
            args.arch, use_bottleneck=False, new_cls=False if args.dataset == 'imagenet-val' else True, 
            class_num=n_classes, imagenet_pretrain=True, normalize_op=normalize_op).to(device)
    elif args.arch.startswith('ResNet'):
        target_model = networks.ResNetFc(
            args.arch, use_bottleneck=False, new_cls=False if args.dataset == 'imagenet-val' else True, 
            class_num=n_classes, imagenet_pretrain=True, normalize_op=normalize_op).to(device)
    elif args.arch.startswith('densenet'):
        args.batch_size = 32
        target_model = networks.DenseNetFc(
            args.arch, use_bottleneck=False, new_cls=False if args.dataset == 'imagenet-val' else True, 
            class_num=n_classes, imagenet_pretrain=True, normalize_op=normalize_op).to(device)
    elif args.arch == 'mobilenet':
        target_model = networks.MobileNetFc(use_bottleneck=False, new_cls=False if args.dataset == 'imagenet-val' else True, 
            class_num=n_classes, imagenet_pretrain=True, normalize_op=normalize_op).to(device)
    elif args.arch.startswith('denoise'):
        target_model = denoise_networks.DenoiseResNet(
            args.arch, use_bottleneck=False, new_cls=False if args.dataset == 'imagenet-val' else True, 
            class_num=n_classes, imagenet_pretrain=True, normalize_op=normalize_op).to(device)
        cudnn.enabled = False
        args.batch_size = 16
    else:
        raise ValueError('invalid network arch: {}'.format(args.arch))
    if args.dataset != 'imagenet-val':
        target_model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    target_model.eval()

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    if args.attack_type == 'class-targeted':
        if args.target_label is None:
            args.target_label = random.choice(list(range(n_classes)))
            logger.info('randomly pick target_label: {}'.format(args.target_label))
    train_ds = get_dataset(
        args.dataset, 'train', use_normalize=use_normalize, 
        test_size=1. - args.train_ratio, train_size=args.train_ratio, 
        xclude_label=args.target_label)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        get_dataset(args.dataset, 'test', use_normalize=use_normalize,
                    test_size=1., train_size=None, xclude_label=args.target_label),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.g_arch == 'basic':
        model = netG.BasicAG(ngf=args.ngf).to(device)
    elif args.g_arch == 'res':
        model = netG.ResG(ngf=args.ngf).to(device)
    elif args.g_arch == 'conv':
        model = netG.ConvAG(ngf=args.ngf).to(device)
    else:
        raise ValueError('invalid generator arch: {}'.format(args.g_arch))

    if args.g_loss == 'margin':
        optimizer_G = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,
                                weight_decay=0, nesterov=True)
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.epochs // 10,
                                                gamma=0.5)
    else:
        optimizer_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        scheduler_G = None

    if args.g_loss == 'margin':
        margin_criterion = MarginLoss(margin=args.margin_thresh, 
            target=False if args.attack_type == 'untargeted' else True)

    # create target mapping for targeted attacks
    logger.info('clean test accuracy: ')
    clean_accuracy, cm = test(None, target_model, device, test_loader)
    
    if args.attack_type == 'untargeted':
        target_mapping = None
    elif args.attack_type == 'random-targeted':
        if args.dataset in ['Art', 'Clipart', 'Product', 'Real_World', ]:
            target_mapping = torch.load('datasets/office-home_target.pt').to(device)
        else:
            raise ValueError('not support random targeted attacks for {}'.format(args.dataset))
    elif args.attack_type == 'class-targeted':
        logger.info('fix target to {}'.format(args.target_label))
        target_mapping = torch.ones(n_classes) * args.target_label
        target_mapping = target_mapping.long().to(device)
    else:
        cm_filename = './datasets/{}_{}_{}.pkl'.format(args.dataset, args.arch, args.attack_type)
        if os.path.exists(cm_filename):
            # load
            with open(cm_filename, 'rb') as pkl_file:
                target_mapping = pkl.load(pkl_file)['target_mapping'].to(device)
        else:
            target_mapping = create_target_mapping(cm, args.attack_type)
            with open(cm_filename, 'wb') as pkl_file:
                pkl.dump({'cm': cm, 'target_mapping': target_mapping}, pkl_file)
            target_mapping = target_mapping.to(device)

    logger.info('configs:\n{}'.format(args))

    def train(epoch):
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.to(device)
            label = label.to(device)
            if args.attack_type == 'class-targeted':
                assert(label.eq(args.target_label).sum() == 0)

            if args.attack_type != 'untargeted':
                targeted_attack_label = target_mapping[label]
                assert(targeted_attack_label.eq(label).sum() == 0)
                label = targeted_attack_label

            optimizer_G.zero_grad()

            noise = model(nat)
            
            # Projection
            if args.g_loss == 'margin':
                # use addition projection if use margin loss
                adv = noise * eps + nat
            else:
                # use clip projection otherwise
                adv = torch.min(torch.max(noise, nat - eps), nat + eps)
            
            adv = torch.clamp(adv, BBOX_MIN, BBOX_MAX)
            logits_adv = target_model(adv)

            if args.g_loss == 'margin':
                loss_g = margin_criterion(logits_adv, label)
            elif args.g_loss == 'relativistic':
                logits_clean = target_model(nat)
                if args.attack_type == 'untargeted':
                    loss_g = -1. * F.cross_entropy(logits_adv - logits_clean, label)
                else:
                    loss_g = F.cross_entropy(logits_adv, label)
            else:
                raise ValueError('not implemented {} generator loss'.format(args.g_loss))
            
            loss_g.backward()
            optimizer_G.step()

            if (batch_idx+1) % log_interval == 0:
                logger.info("batch {}, loss_g {:.3f}".format(
                    batch_idx + 1, loss_g))
                writer.add_scalar('train_loss', loss_g.item(), int(
                    epoch * len(train_loader.dataset) / args.batch_size) + batch_idx)

    def test(epoch):
        model.eval()
        success = 0
        fool_rate = 0

        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.to(device)
            label = label.to(device)

            clean_out = target_model(nat.detach())
            to_attack_indices = clean_out.argmax(dim=-1) == label

            noise = model(nat)
            # Projection
            if args.g_loss == 'margin':
                # use addition projection if use margin loss
                adv = noise * eps + nat
            else:
                # use clip projection otherwise
                adv = torch.min(torch.max(noise, nat - eps), nat + eps)
            adv = torch.clamp(adv, BBOX_MIN, BBOX_MAX)

            logits_adv = target_model(adv)
            
            final_match = logits_adv.argmax(dim=-1) == label
            # calculate adversarial accuracy on correctly predicted correct samples
            success += torch.sum(final_match[to_attack_indices]).item()

            # fool rate
            if args.attack_type == 'untargeted':
                fool_match = logits_adv.argmax(dim=-1) != label
            else:
                fool_match = logits_adv.argmax(dim=-1) == target_mapping[label]
            fool_rate += torch.sum(fool_match[to_attack_indices]).item()

            # sample clean/adv/noise images
            if batch_idx % log_interval == 0:
                writer.add_image('clean', tv_utils.make_grid(
                    nat[:num_image_samples], normalize=True), batch_idx + int(epoch * len(test_loader.dataset) / args.test_batch_size))
                writer.add_image('adv', tv_utils.make_grid(
                    adv[:num_image_samples], normalize=True), batch_idx + int(epoch * len(test_loader.dataset) / args.test_batch_size))
                writer.add_image('noise', tv_utils.make_grid(
                    noise[:num_image_samples]), batch_idx + int(epoch * len(test_loader.dataset) / args.test_batch_size))

        adversarial_accuracy = success / len(test_loader.dataset) * 100.
        fool_rate = fool_rate / len(test_loader.dataset) * 100.
        return adversarial_accuracy, fool_rate

    best_success = 0.
    time_stats = []
    for epoch in range(args.epochs):
        logger.info('start epoch {} training. '.format(epoch))
        start_time = time.time()
        train(epoch)
        epoch_train_time = time.time() - start_time
        logger.info('training time: {:.2f}'.format(epoch_train_time))
        time_stats.append(epoch_train_time)
        torch.cuda.empty_cache()
        if scheduler_G is not None:
            scheduler_G.step()

        with torch.no_grad():
            adversarial_accuracy, fool_rate = test(epoch)
        writer.add_scalar('adversarial_accuracy', adversarial_accuracy, epoch)
        writer.add_scalar('fool_rate', fool_rate, epoch)

        if fool_rate >= best_success:
            best_success = fool_rate
            torch.save(model.state_dict(), os.path.join(
                output_path, "G_weights.pt"))
        logger.info('epoch {}: adversarial accuracy={:.2f}, fool rate={:.2f}, best_attack_success={:.2f}'.format(
            epoch, adversarial_accuracy, fool_rate, best_success))

        if early_stop_engine.is_stop_training(fool_rate):
            logger.info('out of patience {}, stop training. \nbest success {:.2f}'.format(
                patience, best_success))
            break

    with open(os.path.join(output_path, 'metrics.pkl'), 'wb') as pkl_file:
        pkl.dump({'fool_rate': best_success, 
                  'adversarial_accuracy': adversarial_accuracy, 
                  'train_time': np.mean(time_stats)}, pkl_file)

    logger.info('finish training adversarial generator. ')
