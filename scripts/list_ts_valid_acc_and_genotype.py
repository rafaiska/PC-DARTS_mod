#!/usr/bin/env python
import tarfile

from scripts.arch_data import ArchDataCollection

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def get_best_va_and_geno(fp):
    best = 0.0
    geno = None
    for line in fp:
        if 'valid_acc' in line:
            acc = float(line.split()[-1])
            best = acc if acc > best else best
        elif 'genotype = ' in line:
            geno = ' '.join(line.split()[4:])
    return best, geno


def extract_from_train_search(train_search_id):
    ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, train_search_id))
    ts_f.extractall('/tmp/{}'.format(train_search_id))
    with open('/tmp/{}/log.txt'.format(train_search_id), 'r') as fp:
        valid_acc, genotype = get_best_va_and_geno(fp)
    return valid_acc, genotype


def extract_from_train(train_id):
    if not train_id:
        return None
    ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, train_id))
    ts_f.extractall('/tmp/{}'.format(train_id))
    with open('/tmp/{}/log.txt'.format(train_id), 'r') as fp:
        valid_acc, genotype = get_best_va_and_geno(fp)
    return valid_acc


def main():
    arch_data_collection = ArchDataCollection()
    arch_data_collection.load()
    archs = arch_data_collection.archs
    for a_id in archs:
        ts_id = archs[a_id].train_search_id
        t_id = archs[a_id].best_train_id
        archs[a_id].super_model_acc, archs[a_id].genotype_txt = extract_from_train_search(ts_id)
        archs[a_id].model_acc = extract_from_train(t_id)
        arch_data_collection.save()


if __name__ == '__main__':
    main()
