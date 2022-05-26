#!/usr/bin/env python
import tarfile

from scripts.arch_data import ArchDataCollection
from scripts.arch_data import CLossV

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'
FINAL_CLOSS_V = CLossV.D_LOSS_V5


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
    best_closs_arch_acc = 0.0
    for a_id in archs:
        was_modified = False
        ts_id = archs[a_id].train_search_id
        t_id = archs[a_id].best_train_id
        if not archs[a_id].super_model_acc or not archs[a_id].genotype_txt:
            archs[a_id].super_model_acc, archs[a_id].genotype_txt = extract_from_train_search(ts_id)
            was_modified = True
        if not archs[a_id].model_acc:
            archs[a_id].model_acc = extract_from_train(t_id)
            was_modified = True
        print(a_id, archs[a_id].super_model_acc, archs[a_id].model_acc, archs[a_id].genotype_txt)
        # if archs[a_id].model_acc:
        #     print(','.join([
        #         a_id, str(archs[a_id].model_acc), str(archs[a_id].macs_count)]))
        if was_modified:
            arch_data_collection.save()
        if archs[a_id].closs_v == FINAL_CLOSS_V and archs[a_id].model_acc and archs[a_id].model_acc > best_closs_arch_acc:
            best_closs_arch_acc = archs[a_id].model_acc
    print('Best Acc:', best_closs_arch_acc)


if __name__ == '__main__':
    main()
