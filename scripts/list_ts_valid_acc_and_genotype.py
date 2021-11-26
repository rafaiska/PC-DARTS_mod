#!/usr/bin/env python
import os
import sys
import tarfile


def get_data(fp):
    best = 0.0
    geno = None
    for line in fp:
        if 'valid_acc' in line:
            acc = float(line.split()[-1])
            best = acc if acc > best else best
        elif 'genotype = ' in line:
            geno = ' '.join(line.split()[4:])
    return best, geno


def main():
    exp_dir = sys.argv[1]
    filenames = list(filter(lambda fn: 'search' in fn, os.listdir(exp_dir)))
    filenames.sort()
    for fn in filenames:
        exp_id = fn[:-7]
        tf = tarfile.open('{}/{}'.format(exp_dir, fn))
        tf.extractall('/tmp/{}'.format(exp_id))
        with open('/tmp/{}/log.txt'.format(fn[:-7]), 'r') as fp:
            valid_acc, genotype = get_data(fp)
            print('{}, {}, {}'.format(exp_id, valid_acc, genotype))


if __name__ == '__main__':
    main()