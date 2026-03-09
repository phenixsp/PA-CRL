import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.vq_resnet_pacrl import VQResNetPaCRL
from vq_pacrl import VQPaCRL

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Implementation of PA-CRL')
parser.add_argument('--data', metavar='DIR',
                    default="",
                    help='path to dataset')
parser.add_argument('-dataset-name', default='custom',
                    help='dataset name', choices=['stl10', 'cifar10', 'custom'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate 0.0003', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=64, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=20, type=int,
                    help='Log every n steps default: 100')

parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
parser.add_argument('--T', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--n_embed', default=1024, type=int)
parser.add_argument('--embed_dim', default=None)

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--phase', default='train', type=str)

parser.add_argument('--resume', default='',
                    type=str, metavar='PATH')
parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--mask_th', default=-1, type=float, help='mask threshold, -1 for otsu')
parser.add_argument('--disable_ckpt', action='store_true', help='disable torch.utils.checkpoint.checkpoint')
# parser.set_defaults(disable_ckpt=True)
parser.add_argument('--use_random_masking', action='store_true')
parser.add_argument('--masking_rate', default=0.6, type=float)
parser.add_argument('--patch_size', default=4, type=int)

parser.add_argument("--log_dir", default="./logs")


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = VQResNetPaCRL(args)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = VQPaCRL(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()