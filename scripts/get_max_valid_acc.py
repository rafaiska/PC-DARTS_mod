#!/usr/bin/env python
import os
import sys
import tarfile


def get_best_acc(fp):
    best = 0.0
    for line in fp:
        if 'valid_acc' in line:
            acc = float(line.split()[-1])
            best = acc if acc > best else best
    return best


def main():
    exp_dir = sys.argv[1]
    filenames = filter(lambda fn: 'eval' in fn, os.listdir(exp_dir))
    for fn in filenames:
        exp_id = fn[:-7]
        tf = tarfile.open('{}/{}'.format(exp_dir, fn))
        tf.extractall('/tmp/{}'.format(exp_id))
        with open('/tmp/{}/log.txt'.format(fn[:-7]), 'r') as fp:
            best_acc = get_best_acc(fp)
            print('{} best acc: {}'.format(exp_id, best_acc))


if __name__ == '__main__':
    main()
