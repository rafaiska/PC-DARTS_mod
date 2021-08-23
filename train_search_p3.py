import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset

import my_utils
import utils
from architect import Architect
from genotypes import PRIMITIVES
from model_search import Network


class OpPerformanceOracle:
    def __init__(self):
        self.weights = {}

    def reset_weights(self):
        self.weights.clear()
        for p in PRIMITIVES:
            self.weights[p] = None

    def set_weight(self, operation, weight):
        self.weights[operation] = weight

    def _weights_are_valid(self):
        for weight in self.weights.values():
            if not weight:
                return False
        return True

    def get_operation_rate(self, network_cells_alphas):
        if not self._weights_are_valid():
            raise RuntimeError('Undefined weights')
        total_weight = sum(self.weights.values())
        weighted_alphas = []
        for alpha in network_cells_alphas:
            if len(self.weights) != len(alpha):
                raise RuntimeError('Incorrect dimension for Alpha')
            a_softmax = F.softmax(alpha)
            for i in range(alpha.dim()):
                weighted_alphas.append(self.weights[PRIMITIVES[i]] * a_softmax[i])
        return sum(weighted_alphas) / (total_weight * len(network_cells_alphas))


oracle = OpPerformanceOracle()
oracle.reset_weights()
oracle.set_weight('none', 100)
oracle.set_weight('max_pool_3x3', 100)
oracle.set_weight('avg_pool_3x3', 100)
oracle.set_weight('skip_connect', 100)
oracle.set_weight('sep_conv_3x3', 1)
oracle.set_weight('sep_conv_5x5', 100)
oracle.set_weight('dil_conv_3x3', 100)
oracle.set_weight('dil_conv_5x5', 100)


class CustomLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, current_alpha=None):
        super(CustomLoss, self).__init__(weight, size_average)
        self.current_network_cells_alphas = current_alpha

    def update_current_network_cells_alphas(self, network_cells_alphas):
        self.current_network_cells_alphas = network_cells_alphas

    def forward(self, input, target):
        a = super(CustomLoss, self).forward(input, target)
        op_rate = oracle.get_operation_rate(self.current_network_cells_alphas)
        return a * op_rate


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--custom_loss', action='store_true', default=False, help='use custom loss')
parser.add_argument('--resume_checkpoint', action='store_true', default=False, help='resume training from checkpoint')
args = parser.parse_args()

if not args.resume_checkpoint:
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    checkpoint_epoch = None
else:
    args.save, checkpoint_epoch = my_utils.get_checkpoint_info()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.custom_loss:
        criterion = CustomLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    if args.resume_checkpoint:
        logging.info('Loading checkpoint {}...'.format(args.save))
        my_utils.load_checkpoint_model(model, args.save)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    scheduler.step()
    starting_epoch = 0 if not checkpoint_epoch else checkpoint_epoch
    logging.info('Starting EXP from epoch {}'.format(starting_epoch))
    for epoch in range(starting_epoch, args.epochs):
        # scheduler.step()
        # lr = scheduler.get_lr()[0]
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        logging.info('ALPHAS NORMAL: {}'.format(F.softmax(model.alphas_normal, dim=-1)))
        print(F.softmax(model.alphas_reduce, dim=-1))
        logging.info('ALPHAS REDUCE: {}'.format(F.softmax(model.alphas_normal, dim=-1)))
        print(F.softmax(model.betas_normal[2:5], dim=-1))
        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)

        scheduler.step()  # Para consertar warning "UserWarning: Detected call of `lr_scheduler.step()` before
        # `optimizer.step()`"

        # validation
        if args.epochs - epoch <= 1:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        if args.custom_loss:
            criterion.update_current_network_cells_alphas(torch.cat((model.alphas_normal, model.alphas_reduce), dim=0))
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        # input_search, target_search = next(iter(valid_queue))
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        if epoch >= 15:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
