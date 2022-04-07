import tarfile

from scripts.arch_data import ArchDataCollection, CLossV

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def check_if_script_bogus_original(fp):
    """
    Bogus line example:
    args = Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=256, c_loss_w=1e-07,
        custom_loss=False, cutout=False, cutout_length=16, data='../data', drop_path_prob=0.3, epochs=50, gpu=0,
        grad_clip=5, init_channels=16, layers=8, learning_rate=0.1, learning_rate_min=0.0, load_warmup=False,
        model_path='saved_models', momentum=0.9, report_freq=50, resume_checkpoint=False,
        save='search-EXP-20220319-185244', seed=2, set='cifar10', train_portion=0.5, unrolled=False,
        weight_decay=0.0003)
    :param fp: file pointer
    :return: bool
    """
    for line in fp:
        if 'Namespace' in line:
            if 'custom_loss=' in line:
                # This is a modified train_search.py script
                return True
            else:
                return False
    raise RuntimeError('Namespace not found in log file')


def is_bogus_original(arch):
    ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, arch.train_search_id))
    ts_f.extractall('/tmp/{}'.format(arch.train_search_id))
    with open('/tmp/{}/log.txt'.format(arch.train_search_id), 'r') as fp:
        is_bogus = check_if_script_bogus_original(fp)
    return is_bogus


def find_and_mark_bogus_originals(arch_collection):
    original_archs = list(filter(lambda i: i.closs_v == CLossV.ORIGINAL, arch_collection.archs.values()))
    for a in original_archs:
        if is_bogus_original(a):
            print('{} is bogus'.format(a.arch_id))
            a.closs_v = CLossV.BOGUS_ORIGINAL
            arch_collection.save()


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    find_and_mark_bogus_originals(arch_collection)


if __name__ == '__main__':
    main()
