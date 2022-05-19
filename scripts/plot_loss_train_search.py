import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def configure_plot(a_data, b_data, titles):
    fig = plt.figure(figsize=(16, 8))
    # Creating subplot/axes
    a_ax = fig.add_subplot(211)
    b_ax = fig.add_subplot(212)

    # Setting axes/plot title
    title_a, title_b = titles
    a_ax.set_title(title_a)
    b_ax.set_title(title_b)

    # Setting X-axis and Y-axis limits
    #     ax.set_xlim([1, len(data['dil_conv_3x3'])])
    #     ax.set_ylim([0.1, 0.25])
    a_ax.autoscale()
    b_ax.autoscale()

    # Setting X-axis and Y-axis labels
    a_ax.set_ylabel('Loss value')
    a_ax.set_xlabel('Training step')
    b_ax.set_ylabel('Loss value')
    b_ax.set_xlabel('Training step')

    # Plot
    ce_loss_a, custom_loss_a = a_data
    a_ax.plot([x for x in range(len(ce_loss_a))], ce_loss_a, label='$L_{CE}$')
    a_ax.plot([x for x in range(len(custom_loss_a))], custom_loss_a, label='$N_{MACS} \\times \\omega$')

    ce_loss_b, custom_loss_b = b_data
    b_ax.plot([x for x in range(len(ce_loss_b))], ce_loss_b, label='$L_{CE}$')
    b_ax.plot([x for x in range(len(custom_loss_b))], custom_loss_b, label='$L_{custom} \\times w$')

    a_ax.legend()
    b_ax.legend()

    plt.subplots_adjust(hspace=0.3)

    return fig, a_ax, b_ax


def extract_experiment_data_to_tmp(exp_id):
    tarfile_path = '{}/{}.tar.gz'.format(EXP_DIR, exp_id)
    ts_f = tarfile.open(tarfile_path)
    target_dir = '/tmp/{}'.format(exp_id)
    ts_f.extractall(target_dir)
    return target_dir


def build_from_exp_data(exp_path):
    ce_loss_history = []
    custom_loss_history = []
    file_pt = open(exp_path + '/log.txt', 'r')
    while True:
        line = file_pt.readline()
        if not line:
            break
        if ' LOSS = ' in line:
            splitted = line.split()
            ce_loss = float(splitted[4])
            ce_loss_history.append(ce_loss)
            custom_loss = float(splitted[6])
            custom_loss_history.append(custom_loss)
    file_pt.close()
    return ce_loss_history, custom_loss_history


def main():
    collection = ArchDataCollection()
    collection.load()
    arch_ids = sys.argv[1:]
    if len(arch_ids) % 2 != 0:
        raise ValueError('Number of archs must be even')
    for a_id in arch_ids:
        if a_id not in collection.archs:
            raise ValueError('Invalid arch id: {}'.format(a_id))
    for i in range(0, len(arch_ids), 2):
        ts_id_a = collection.archs[arch_ids[i]].train_search_id
        ts_id_b = collection.archs[arch_ids[i + 1]].train_search_id
        a_dir = extract_experiment_data_to_tmp(ts_id_a)
        b_dir = extract_experiment_data_to_tmp(ts_id_b)
        a_data = build_from_exp_data(a_dir)
        b_data = build_from_exp_data(b_dir)
        titles = ['Customized Loss Function Components: {}'.format(t) for t in (
            'Non-Differentiable', 'Differentiable')]
        configure_plot(a_data, b_data, titles)
        plt.savefig('{}{}-training-loss.pdf'.format(*arch_ids[i:i + 2]), bbox_inches='tight')


if __name__ == '__main__':
    main()
