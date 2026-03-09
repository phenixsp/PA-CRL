#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its rights reserved.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import models.resnet as resnet_models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch PA-CRL linear probing evaluation')
parser.add_argument('--data',
                    help='path to dataset',
                    default="",
                    )
parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet50',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default="",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training')
parser.add_argument('--num_classes', default=4, type=int)

# additional configs:
parser.add_argument('--pretrained',
                    default="",
                    type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--ckpt_dir',
                    default="",
                    type=str)
parser.add_argument('--log_file', default="linear_probing.log", type=str, help='filename for text logging')

best_acc1 = 0

def save_evaluation_results(results, file_path):
    with open(file_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Linear Probing Evaluation Results\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall Top-1 Accuracy: {results['overall_accuracy']:.4f}\n")
        # f.write(f"Overall Top-3 Accuracy: {results['overall_top3_accuracy']:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")

        for i in range(len(results['precision_per_class'])):
            f.write(f"{i:<10} {results['precision_per_class'][i]:<12.4f} "
                    f"{results['recall_per_class'][i]:<12.4f} "
                    f"{results['f1_per_class'][i]:<12.4f} "
                    f"{results['support_per_class'][i]:<10}\n")
        f.write("\n")

        f.write("Average Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Type':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Macro':<12} {results['macro_precision']:<12.4f} "
                f"{results['macro_recall']:<12.4f} {results['macro_f1']:<12.4f}\n")
        f.write(f"{'Micro':<12} {results['micro_precision']:<12.4f} "
                f"{results['micro_recall']:<12.4f} {results['micro_f1']:<12.4f}\n")
        f.write(f"{'Weighted':<12} {results['weighted_precision']:<12.4f} "
                f"{results['weighted_recall']:<12.4f} {results['weighted_f1']:<12.4f}\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        cm = np.array(results['confusion_matrix'])
        n_classes = cm.shape[0]

        header = "True \\ Pred" + "".join([f"{i:>8}" for i in range(n_classes)]) + "\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for i in range(n_classes):
            row = f"Class {i:<5}" + "".join([f"{cm[i][j]:>8}" for j in range(n_classes)]) + "\n"
            f.write(row)

        f.write("\n")

        f.write("Detailed Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(results['classification_report'])


def compute_detailed_metrics(all_predictions, all_labels, num_classes):
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    cm = confusion_matrix(labels, predictions)
    class_report = classification_report(labels, predictions, digits=4, zero_division=0)
    overall_accuracy = (predictions == labels).mean()

    results = {
        'overall_accuracy': overall_accuracy,
        # 'overall_top3_accuracy': overall_top3_accuracy,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }

    return results


def main():
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.ckpt_dir, args.log_file)),
        ]
    )
    logging.info(f"Start training with args: {args}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch in ['resnet50w2']:
        model, _ = resnet_models.resnet50x2()
        model.fc = nn.Linear(4096, args.num_classes)
    elif args.arch in ['resnet50w4']:
        model, _ = resnet_models.resnet50x4()
        model.fc = nn.Linear(8192, args.num_classes)
    else:
        model = models.__dict__[args.arch]()
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                new_k = k.replace('backbone.', '')
                state_dict[new_k] = state_dict[k]
                del state_dict[k]
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        acc1, detailed_metrics = validate(val_loader, model, criterion, args)
        epoch_eval_path = os.path.join(args.ckpt_dir, f"detailed_evaluation_val.txt")
        save_evaluation_results(detailed_metrics, epoch_eval_path)
        logging.info(f"Evaluation results saved to: {epoch_eval_path}")
        return

    best_detailed_metrics = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, detailed_metrics = validate(val_loader, model, criterion, args)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            epoch_eval_path = os.path.join(args.ckpt_dir, f"detailed_evaluation_epoch{epoch}.txt")
            save_evaluation_results(detailed_metrics, epoch_eval_path)
            logging.info(f"Epoch {epoch} evaluation results saved to: {epoch_eval_path}")

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            best_detailed_metrics = detailed_metrics

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, ckpt_dir=args.ckpt_dir)

            if is_best and best_detailed_metrics is not None:
                best_eval_path = os.path.join(args.ckpt_dir, "best_model_detailed_evaluation.txt")
                save_evaluation_results(best_detailed_metrics, best_eval_path)
                logging.info(f"Best model evaluation results saved to: {best_eval_path}")

            if epoch == args.start_epoch:
                sanity_check(args, model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top3],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top3.update(acc3[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            logging.info(progress.get_log_message(i))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top3],
        prefix='Test: ')

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        detailed_metrics = compute_detailed_metrics(all_predictions, all_labels, args.num_classes)

        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return top1.avg, detailed_metrics


def save_checkpoint(state, is_best, ckpt_dir, filename='checkpoint.pth.tar'):
    torch.save(state, ckpt_dir + filename)
    if is_best:
        shutil.copyfile(ckpt_dir + filename, ckpt_dir + 'model_best.pth.tar')


def sanity_check(args, state_dict, pretrained_weights):
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if 'fc.weight' in k or 'fc.bias' in k:
            continue
        k_pre = 'backbone.' + k
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def get_log_message(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [meter.__str__() for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()