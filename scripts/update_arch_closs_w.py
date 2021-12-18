import tarfile

from scripts.arch_data import ArchDataCollection

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def parse_namespace(line):
    closs_w = None
    for s in line.split():
        if 'c_loss_w=' in s:
            closs_w = float(s[s.find('=') + 1:s.find(',')])
    return closs_w


def parse_file(fp):
    closs_w = None
    for line in fp:
        if 'args = Namespace(' in line:
            closs_w = parse_namespace(line)
    return closs_w


def extract_from_train_search(train_search_id):
    ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, train_search_id))
    ts_f.extractall('/tmp/{}'.format(train_search_id))
    with open('/tmp/{}/log.txt'.format(train_search_id), 'r') as fp:
        closs_w = parse_file(fp)
    return closs_w


def update_arch_closs_w():
    arch_c = ArchDataCollection()
    arch_c.load()
    archs = arch_c.archs
    for a_id in archs:
        if not archs[a_id].closs_w:
            archs[a_id].closs_w = extract_from_train_search(archs[a_id].train_search_id)
        print(a_id, archs[a_id].closs_w)
    arch_c.save()
    w_instance_qt = {}
    for a in arch_c.archs.values():
        if hasattr(a, 'closs_w'):
            if a.closs_w not in w_instance_qt:
                w_instance_qt[a.closs_w] = 0
            w_instance_qt[a.closs_w] += 1
    print(w_instance_qt)


if __name__ == '__main__':
    update_arch_closs_w()
