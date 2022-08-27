'''search in the embedding space of cross-domain perturbation with guided evolutionary strategies'''

import argparse
import os
import copy
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tv_utils
from tensorboardX import SummaryWriter

import netG
import networks
from prep_data import get_dataset
from utils import normalize_op
from TREMBA.QueryAgent import QueryAgent


# create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


parser = argparse.ArgumentParser(
    description='Transferred Evolutionary Strategies')
parser.add_argument('--dataset', type=str, help='target domain dataset',
                    choices=['cifar10', 'stl10', 'mnist_m', 
                             'Art', 'Clipart', 'Product', 'Real_World', 
                             'real', 'sketch', 'quickdraw', 
                             'amazon', 'webcam', 'dslr', 
                             ])
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--arch_a', type=str, default='VGG16', 
                    help='model arch of the surrogate model')
parser.add_argument('--arch_b', type=str, default='VGG16', 
                    help='model arch of the target of attack')
parser.add_argument('--ckpt_a', type=str,
                    help='path to load weights of the source model to produce surrogate gradients. ')
parser.add_argument('--ckpt_b', type=str,
                    help='path to load weights of the target of attack. ')
parser.add_argument('--imagenet_pretrain', action='store_true',
                    help='use pretrained imagenet model as model A')
parser.add_argument('--imagenet_target_label', type=int, 
                    help='the target label in imagenet label space. (only for class-targeted attacks)')
# adversarial generator params
parser.add_argument('--ngf', type=int, default=64,
                    help='Number of filters in the adversarial generator')
parser.add_argument('--g_arch', type=str, default='basic', choices=['basic', 'res', 'conv'], 
                    help='net arch of the adversarial generator')
parser.add_argument('--g_loss', type=str, choices=['margin', 'relativistic'], default='relativistic', 
                    help='load generator weights. ')
parser.add_argument('--g_path', type=str,
                    help='load generator weights. ')
# attack params
parser.add_argument('--attack_type', type=str, default='untargeted', 
                    choices=['untargeted', 'class-targeted'], 
                    help='attack type')
parser.add_argument('--target_label', type=int, 
                    help='label of class-targeted attacks')
# TREMBA params
parser.add_argument('--alpha', type=float, default=0.01,
                    help='weight of normal distribution. ')
parser.add_argument('--p', type=int, default=20,
                    help='number of samples of NES')
parser.add_argument('--max_iterations', type=int, default=100,
                    help='maximum number of iterations allowed of NES')
args = parser.parse_args()
logger.info('args: {}'.format(args))

# Normalize (0-1)
eps = args.eps / 255

# cosine similarity function
cosine_func = nn.CosineSimilarity(dim=1)

# const
NUM_IMAGE_SAMPLES = 8
BBOX_MIN, BBOX_MAX = 0., 1.

# config of attack_untarget.json
config = {
    "margin": 5.0,
    "target": False if args.attack_type == 'untargeted' else True,
    "sample_size": args.p,
    "num_iters": args.max_iterations,
    "lr": 5.0,
    "lr_min": 0.1,
    "lr_decay": 2.0,
    "momentum": 0.0,
    "plateau_length": 20,
    "plateau_overhead": 0.3,

    "sigma": 1.0,
    "alpha": args.alpha,
    "beta": 2.,
    "k": 28,
}

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# output_path
output_path = os.path.join(os.path.split(args.ckpt_b)[0], 'TES', args.attack_type, 
                           datetime.strftime(datetime.now(), "%d%b%y-%H%M%S"))
logger.info('save results to {}'.format(output_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)
writer = SummaryWriter(output_path)


# load adversarial generator
if args.g_arch == 'basic':
    model_G = netG.BasicAG(ngf=args.ngf).to(device)
elif args.g_arch == 'res':
    model_G = netG.ResG(ngf=args.ngf).to(device)
elif args.g_arch == 'conv':
    model_G = netG.ConvAG(ngf=args.ngf).to(device)
else:
    raise ValueError('invalid generator arch: {}'.format(args.g_arch))
model_G.load_state_dict(torch.load(args.g_path, map_location='cpu'))
model_G = model_G.to(device)
model_G.eval()

# load the target of attack
if args.dataset in ["cifar10", "stl10"]:
    n_classes = 9
elif args.dataset in ["usps", "mnist", "svhn", "mnist_m"]:
    n_classes = 10
elif args.dataset in ['Art', 'Clipart', 'Product', 'Real_World', ]:
    n_classes = 65
elif args.dataset in ['real', 'sketch', 'quickdraw']:
    n_classes = 345
elif args.dataset in ['amazon', 'webcam', 'dslr', ]:
    n_classes = 31
else:
    raise ValueError('invalid dataset option: {}'.format(args.dataset))

if args.arch_b == 'dtn':
    model_t = networks.DTN().to(device)
elif args.arch_b == 'wrn':
    model_t = networks.WideResNet(
        depth=28, num_classes=n_classes, widen_factor=10, dropRate=0.0).to(device)
elif args.arch_b.startswith('VGG'):
    model_t = networks.VGGFc(
        args.arch_b, use_bottleneck=False, new_cls=True, class_num=n_classes, normalize_op=normalize_op).to(device)
elif args.arch_b.startswith('ResNet'):
    model_t = networks.ResNetFc(
        args.arch_b, use_bottleneck=False, new_cls=True, class_num=n_classes, normalize_op=normalize_op).to(device)
else:
    raise ValueError('invalid network architecture {}'.format(args.arch_b))
model_t.load_state_dict(torch.load(args.ckpt_b, map_location='cpu'))
model_t.eval()

# load source model
n_classes_a = n_classes
if args.imagenet_pretrain:
    n_classes_a = 1000

if args.arch_a == 'dtn':
    model_a = networks.DTN().to(device)
elif args.arch_a == 'wrn':
    model_a = networks.WideResNet(
        depth=28, num_classes=n_classes, widen_factor=10, dropRate=0.0).to(device)
elif args.arch_a.startswith('VGG'):
    model_a = networks.VGGFc(
        args.arch_a, use_bottleneck=False, new_cls=False if args.imagenet_pretrain else True, 
        class_num=n_classes_a, imagenet_pretrain=args.imagenet_pretrain, normalize_op=normalize_op).to(device)
elif args.arch_a.startswith('ResNet'):
    model_a = networks.ResNetFc(
        args.arch_a, use_bottleneck=False, new_cls=False if args.imagenet_pretrain else True, 
        class_num=n_classes_a, imagenet_pretrain=args.imagenet_pretrain, normalize_op=normalize_op).to(device)
elif args.arch_a.startswith('densenet'):
    model_a = networks.DenseNetFc(
        args.arch_a, use_bottleneck=False, new_cls=False if args.imagenet_pretrain else True, 
        class_num=n_classes_a, imagenet_pretrain=args.imagenet_pretrain, normalize_op=normalize_op).to(device)
else:
    raise ValueError('invalid network architecture {}'.format(args.arch_a))
if not args.imagenet_pretrain:
    model_a.load_state_dict(torch.load(args.ckpt_a, map_location='cpu'))
model_a.eval()

# Setup-Data
use_normalize = False
kwargs = {'num_workers': 2, 'pin_memory': True}
if args.attack_type == 'untargeted':
    args.target_label = None
elif args.attack_type == 'class-targeted':
    logger.info('fix target to {}'.format(args.target_label))
    if args.target_label is None:
        args.target_label = random.choice(list(range(n_classes)))
else:
    raise ValueError('invalid attack_type: {}'.format(args.attack_type))

test_split = 'test'
if args.dataset in ['real', 'sketch', 'quickdraw']:
    test_split = 'attack_test'
test_loader = torch.utils.data.DataLoader(
    get_dataset(args.dataset, test_split, use_normalize=use_normalize,
                test_size=1., train_size=0., xclude_label=args.target_label),
    batch_size=1, shuffle=False, **kwargs)
test_size = len(test_loader.dataset)
logger.info('in {}: # test={}'.format(args.dataset, test_size))

# query attack target
fa = QueryAgent(model_a, batch_size=64, margin=config['margin'],
                nlabels=n_classes_a, target=config['target'])
fb = QueryAgent(model_t, batch_size=config[
                'sample_size'], margin=config['margin'], nlabels=n_classes, target=config['target'])

# Evaluation
adv_acc = 0
clean_acc = 0
fool_rate = 0
test_size = len(test_loader.dataset)
with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        int_label = label.cpu().item()
        if args.attack_type == 'class-targeted':
            int_label = args.target_label

        clean_out = model_t(img.detach())
        to_attack_indices = clean_out.argmax(dim=-1) == label
        clean_acc += torch.sum(to_attack_indices).item()

        # attack if correctly predicted
        if to_attack_indices.sum().item() > 0:
            # search in the encoder space
            latent = model_G.encode(img)
            enc_out_size = latent.size()
            latent = latent.squeeze().view(-1)
            n_dim = len(latent)
            lr = config['lr']
            last_loss = []

            for t in range(config['num_iters']):
                # get surrogate gradients by NES using model A
                latent.requires_grad = True
                latent.retain_grad()
                with torch.enable_grad():
                    adv = model_G.decode(latent.view(*enc_out_size))

                    # Projection
                    if args.g_loss == 'margin':
                        # use addition projection if use margin loss
                        adv = adv * eps + img
                    else:
                        # use clip projection otherwise
                        adv = torch.min(torch.max(adv, img - eps), img + eps)
                    adv = torch.clamp(adv, BBOX_MIN, BBOX_MAX)

                    fa.model.zero_grad()
                    if args.imagenet_pretrain:
                        _, loss_A = fa(adv, args.imagenet_target_label)
                    else:
                        _, loss_A = fa(adv, int_label)
                    loss_A.backward(retain_graph=True)
                    gz = latent.grad.data

                    # true gradient
                    fb.model.zero_grad()
                    _, loss_B = fb(adv, int_label)
                    fb.current_counts -= 1
                    loss_B.backward()
                    gb = latent.grad.data

                logit, loss = fb(adv, int_label)
                attack_success = torch.argmax(logit, dim=1) != label
                last_loss.append(loss.item())

                if fb.current_counts > 50000:
                    break

                if bool(attack_success.item()):
                    fool_rate += len(img)
                    break

                # guided es along single dimension
                U, _ = torch.qr(gz.view(n_dim, 1).detach().cpu())
                U = torch.reshape(U, (n_dim, 1))
                a = (config['sigma'] ** 2) * config['alpha']
                b = config['sigma'] * np.sqrt((1 - config['alpha']))
                mvn = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                    torch.zeros(n_dim), cov_diag=a * torch.ones(n_dim),
                    cov_factor=b * U)
                # shape: [p/2, n_dim]
                mvn_sample = mvn.sample(
                    [config['sample_size'] // 2]).to(device)
                noise = torch.cat([mvn_sample, -1 * mvn_sample], dim=0)
                latents = latent.repeat(config['sample_size'], 1) + noise
                perturbations = model_G.decode(latents.view(
                    config['sample_size'], enc_out_size[1], enc_out_size[2], enc_out_size[3], ))
                if args.g_loss == 'margin':
                    # use addition projection if use margin loss
                    perturbations = perturbations * eps + img
                else:
                    # use clip projection otherwise
                    perturbations = torch.min(
                        torch.max(perturbations, img - eps), img + eps)
                perturbations = torch.clamp(perturbations, BBOX_MIN, BBOX_MAX)
                _, losses = fb(perturbations, int_label)

                # logger.info('losses size: {}, noise size: {}'.format(losses.size(), mvn_sample.size()))
                c = config['beta'] / (2 * config['sigma'] ** 2)
                grad = c * \
                    torch.mean(losses.view(-1, 1).expand_as(noise)
                               * noise, dim=0).view(-1)
                latent = latent - lr * grad

                last_loss = last_loss[-config['plateau_length']:]
                if (last_loss[-1] > last_loss[0] + config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1] < 0.6) and len(last_loss) == config['plateau_length']:
                    if lr > config['lr_min']:
                        lr = max(lr / config['lr_decay'], config['lr_min'])
                    last_loss = []

                if t % 10 == 0:
                    cosine_distance = cosine_func(
                        gz.view(-1, n_dim), gb.view(-1, n_dim))
                    # rho = torch.matmul(gb.view(-1, n_dim),
                    #                    U.to(device)).item() / torch.norm(gb)
                    rho = torch.Tensor([0])
                    logger.info('iteration {}\tlosses: {:.3f}, cosine: {:.3f}, rho: {:.3f}'.format(
                        t, loss.item(), cosine_distance.item(), rho.item()))

            if not bool(attack_success.item()):
                adv_acc += len(img)

            logger.info('At Batch:{}\tsuccess:{}\tl_inf:{:.4f}\teval_count:{}\taverage_count: {} '.format(
                i, attack_success.item(), (img - adv).max() * 255, fb.current_counts, fb.get_average()))
            fb.new_counter()

            writer.add_scalar('eval_count', fb.current_counts, i)
            if i % 10 == 0:
                writer.add_image('clean', img.squeeze().cpu(), i)
                writer.add_image('adv', adv.squeeze().cpu(), i)


clean_acc = clean_acc / test_size
adv_acc = adv_acc / test_size
fool_rate = fool_rate / test_size

logger.info('config: {}'.format(config))
logger.info('Clean:{:.3%}\t Adversarial :{:.3%}\t Fooling Rate:{:.3%}\tAverage query count: {:.4f}'.format(
    clean_acc, adv_acc, fool_rate, fb.get_average()))

with open(os.path.join(output_path, 'log.txt'), 'w') as log_file:
    print('args: {}\n'.format(args), file=log_file)
    print('attack config: {}\n'.format(config), file=log_file)
    print('clean_acc: {:.3f}\tadv_acc: {:.3f}\tfool_rate: {:.3f}\tAverage query count: {:.3f}'.format(
        clean_acc, adv_acc, fool_rate, fb.get_average()), file=log_file)

np.save(os.path.join(output_path, 'eval_count.npy'), np.array(fb.counts))
