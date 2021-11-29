import tarfile

import matplotlib.pyplot as plt

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'
ARCH_EXP_IDS = {'M2': {'train_search': 'search-EXP-20210904-015723', 'train': 'eval-EXP-20211001-182408'},
                'M3': {'train_search': 'search-EXP-20210909-083612', 'train': 'eval-EXP-20210929-092446'},
                'M5': {'train_search': 'search-EXP-20210914-132854', 'train': 'eval-EXP-20210930-001834'},
                'M6': {'train_search': 'search-EXP-20210915-193742', 'train': 'eval-EXP-20210930-012047'},
                'M7': {'train_search': 'search-EXP-20210916-134929', 'train': 'eval-EXP-20210930-144310'},
                'M8': {'train_search': 'search-EXP-20210923-085601', 'train': 'eval-EXP-20211018-175456'},
                'M9': {'train_search': 'search-EXP-20210930-113256', 'train': 'eval-EXP-20211019-000907'},
                'M10': {'train_search': 'search-EXP-20211007-153233', 'train': 'eval-EXP-20211008-102216'},
                'M11': {'train_search': 'search-EXP-20211008-090653', 'train': 'eval-EXP-20211019-013549'},
                'M12': {'train_search': 'search-EXP-20211025-164301', 'train': 'eval-EXP-20211028-153142'}}


def plot_acc_vs_macs(plot_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlim(80, 90)
    # ax.set_ylim(90, 100)
    plt.xlabel('train_search.py accuracy')
    plt.ylabel('train.py accuracy')
    ax.scatter([plot_data[a_id]['train_search'] for a_id in plot_data],
               [plot_data[a_id]['train'] for a_id in plot_data], color='red', marker='o')
    for a_id in plot_data:
        x = plot_data[a_id]['train_search']
        y = plot_data[a_id]['train']
        plt.text(x=x, y=y, s=a_id)
    plt.savefig('smodel_acc_vs_model_acc.svg')


def get_best_acc(fp):
    best = 0.0
    for line in fp:
        if 'valid_acc' in line:
            acc = float(line.split()[-1])
            best = acc if acc > best else best
    return best


def fetch_acc_data(exp_data):
    for a_id in exp_data:
        for exp_type in exp_data[a_id]:
            exp_id = exp_data[a_id][exp_type]
            tf = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, exp_id))
            tf.extractall('/tmp/{}'.format(exp_id))
            with open('/tmp/{}/log.txt'.format(exp_id), 'r') as fp:
                exp_data[a_id][exp_type] = get_best_acc(fp)


if __name__ == '__main__':
    fetch_acc_data(ARCH_EXP_IDS)
    plot_acc_vs_macs(ARCH_EXP_IDS)
