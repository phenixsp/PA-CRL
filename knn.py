import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from datetime import datetime

from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import models.resnet as resnet_models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--data',
                    help='path to dataset',
                    default="")
parser.add_argument('-a', '--arch', metavar='ARCH', help='model architecture')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32) 2')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
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


def main(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
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
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
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
    # elif args.arch in ['resnet50_w2x']:
    #     model = resnet50_w2x()
    #     model.fc = nn.Linear(model.fc[0].in_features, args.num_classes)
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

            # rename moco pre-trained keys
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

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    validate(train_loader, val_loader, model, args, phase="train")
    validate(train_loader, val_loader, model, args, phase="val")

def model_forward(args, model, x):
    if args.arch in ['resnet50w2', 'resnet50w4']:
        x = model.padding(x)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
    else:
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x


def validate(train_loader, val_loader, model, args, phase="train"):
    data = None
    label = None
    # switch to evaluate mode
    model.eval()
    loader = train_loader if phase == "train" else val_loader

    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(loader)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model_forward(args, model, images).squeeze()

            data = output if data is None else torch.cat([data, output], dim=0)
            label = target if label is None else torch.cat([label, target], dim=0)

    data = data.cpu().numpy()
    label = label.cpu().numpy()
    if phase == "train":
        np.save(f"{args.ckpt_dir}/train_data.npy", data)
        np.save(f"{args.ckpt_dir}/train_label.npy", label)
    else:
        np.save(f"{args.ckpt_dir}/data.npy", data)
        np.save(f"{args.ckpt_dir}/label.npy", label)
    return data, label


def save_detailed_classification_report(y_test, y_pred, target_names, save_path=None):
    avg_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, digits=4)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='micro', zero_division=0
    )
    conf_matrix = confusion_matrix(y_test, y_pred)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assert save_path is not None
    if save_path is None:
        save_path = f"classification_report_{timestamp}.txt"
    else:
        save_path = f"{save_path}/classification_report_{timestamp}.txt"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Detailed Classification Evaluation Report\n")
        f.write("=" * 70 + "\n\n")

        # Overall Accuracy
        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Accuracy: {avg_accuracy:.5f}\n\n")

        # Detailed metrics for each class
        f.write("Per-Class Detailed Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write("-" * 80 + "\n")

        for class_name in target_names:
            metrics = class_report[class_name]
            f.write(f"{class_name:<12} {metrics['precision']:<12.5f} {metrics['recall']:<12.5f} "
                    f"{metrics['f1-score']:<12.5f} {metrics['support']:<12}\n")
        f.write("\n")

        # Macro Average
        f.write("Macro Average:\n")
        f.write("-" * 40 + "\n")
        macro_metrics = class_report['macro avg']
        f.write(f"Precision: {macro_metrics['precision']:.5f}\n")
        f.write(f"Recall: {macro_metrics['recall']:.5f}\n")
        f.write(f"F1-Score: {macro_metrics['f1-score']:.5f}\n")
        f.write(f"Support: {macro_metrics['support']}\n\n")

        # Micro Average
        f.write("Micro Average:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision: {precision_micro:.5f}\n")
        f.write(f"Recall: {recall_micro:.5f}\n")
        f.write(f"F1-Score: {f1_micro:.5f}\n")
        f.write(f"Support: {len(y_test)}\n\n")

        # Weighted Average
        f.write("Weighted Average:\n")
        f.write("-" * 40 + "\n")
        weighted_metrics = class_report['weighted avg']
        f.write(f"Precision: {weighted_metrics['precision']:.5f}\n")
        f.write(f"Recall: {weighted_metrics['recall']:.5f}\n")
        f.write(f"F1-Score: {weighted_metrics['f1-score']:.5f}\n")
        f.write(f"Support: {weighted_metrics['support']}\n\n")

        # Confusion Matrix
        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        f.write("Rows: True Class, Columns: Predicted Class\n\n")

        # Write column headers
        header = " " * 8 + " ".join(f"{name:>8}" for name in target_names)
        f.write(header + "\n")
        f.write("-" * (8 + 9 * len(target_names)) + "\n")

        # Write confusion matrix content
        for i, row in enumerate(conf_matrix):
            row_name = f"{target_names[i]:<8}" if i < len(target_names) else f"Class {i}:<8"
            row_values = " ".join(f"{value:>8}" for value in row)
            f.write(f"{row_name}{row_values}\n")

        f.write("\n")

        # F1-Score calculation explanation
        f.write("Metric Calculation Explanation:\n")
        f.write("-" * 40 + "\n")
        f.write("Precision = TP / (TP + FP)\n")
        f.write("Recall = TP / (TP + FN)\n")
        f.write("F1-Score = 2 × (Precision × Recall) / (Precision + Recall)\n")
        f.write("Macro Average: Arithmetic mean of per-class metrics\n")
        f.write("Micro Average: Calculated by aggregating TP/FP/FN across all classes\n")
        f.write("Weighted Average: Weighted by the number of samples in each class\n")

        # Dataset statistics
        f.write("\nDataset Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {len(y_test)}\n")
        f.write(f"Number of Classes: {len(target_names)}\n")
        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Detailed classification report saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    X_train = np.load(f"{args.ckpt_dir}/train_data.npy")
    y_train = np.load(f"{args.ckpt_dir}/train_label.npy")

    X_test = np.load(f"{args.ckpt_dir}/data.npy")
    y_test = np.load(f"{args.ckpt_dir}/label.npy")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    target_names = ["Benign", "Grade 3", "Grade 4", "Grade 5"]

    avg_accuracy = accuracy_score(y_test, y_pred)
    avg_recall = recall_score(y_test, y_pred, average='macro')
    avg_f1 = f1_score(y_test, y_pred, average='macro')

    report_path = save_detailed_classification_report(y_test, y_pred, target_names, save_path=args.ckpt_dir)

