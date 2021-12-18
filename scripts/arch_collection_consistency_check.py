import tarfile
from unittest import TestCase

from scripts.arch_data import ArchDataCollection
import genotypes
from genotypes import Genotype

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def check_train_search(fp, arch_id):
    print('Checking arch id match for {}'.format(arch_id))
    case = TestCase()
    for line in fp:
        if 'arch=' in line:
            start = line.find('arch=') + 6
            end = line.find('\',', line.find('arch='))
            file_arch_id = line[start:end]
            case.assertEqual(file_arch_id, arch_id)


def extract_and_check(train_id, arch_id):
    if not train_id:
        return None
    try:
        ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, train_id))
    except tarfile.ReadError:
        print('Corrupted tarfile: {}/{}.tar.gz'.format(EXP_DIR, train_id))
        return
    ts_f.extractall('/tmp/{}'.format(train_id))
    with open('/tmp/{}/log.txt'.format(train_id), 'r') as fp:
        check_train_search(fp, arch_id)


def check_genotypes_equal(genotype_from_col, genotype_from_src):
    case = TestCase()
    for n in genotype_from_src.normal:
        case.assertIn(n, genotype_from_col.normal)
    for n in genotype_from_src.reduce:
        case.assertIn(n, genotype_from_col.reduce)


def check_genotype(a_id, genotype_txt):
    assert type(genotypes.M1) == Genotype
    print('checking {} genotype'.format(a_id))
    genotype_from_src = eval('genotypes.{}'.format(a_id))
    genotype_from_col = eval(genotype_txt)
    check_genotypes_equal(genotype_from_col, genotype_from_src)


def main():
    arch_data_collection = ArchDataCollection()
    arch_data_collection.load()
    archs = arch_data_collection.archs
    for a_id in archs:
        extract_and_check(archs[a_id].best_train_id, a_id)
        check_genotype(a_id, archs[a_id].genotype_txt)


if __name__ == '__main__':
    main()
