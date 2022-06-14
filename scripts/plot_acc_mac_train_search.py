import os
import shutil
import tarfile

import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection, CLossV

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'
TEXT_WIDTH = 6.32283486112
FIGSIZE = (TEXT_WIDTH, TEXT_WIDTH / 2.0)
FIGSIZE_SQUARE = (TEXT_WIDTH, TEXT_WIDTH)
LEGEND_FONT_SIZE = 9
VERSION_NAMES = {CLossV.D_LOSS_V1: 'Loss-v1', CLossV.D_LOSS_V2: 'Loss-v2', CLossV.D_LOSS_V3: 'Loss-v3',
                 CLossV.D_LOSS_V5: 'Loss-v4'}


def configure_plot(accs, macs, title):
    fig = plt.figure(figsize=FIGSIZE)
    # Creating subplot/axes
    acc_ax = fig.add_subplot(111)
    macs_ax = acc_ax.twinx()

    # Setting axes/plot title
    acc_ax.set_title(title)

    # Setting X-axis and Y-axis limits
    #     ax.set_xlim([1, len(data['dil_conv_3x3'])])
    #     ax.set_ylim([0.1, 0.25])
    acc_ax.autoscale()
    macs_ax.autoscale()

    # Setting X-axis and Y-axis labels
    acc_ax.set_ylabel('Acurácia Top1')
    macs_ax.set_ylabel('# MACS')
    acc_ax.set_xlabel('Tempo de Treinamento')

    acc_ax.plot([x for x in range(len(accs))], accs, label='Acc', color='blue')
    macs_ax.plot([x for x in range(len(macs))], macs, label='# MACS', color='red')
    legend = fig.legend(loc='upper right')
    plt.draw()

    bb = legend.get_bbox_to_anchor().transformed(acc_ax.transAxes.inverted())

    # Change to location of the legend.
    x_offset = -1.15
    bb.x0 += x_offset
    bb.x1 += x_offset
    legend.set_bbox_to_anchor(bb, transform=acc_ax.transAxes)
    return fig


def extract_experiment_data_to_tmp(exp_id):
    tarfile_path = '{}/{}.tar.gz'.format(EXP_DIR, exp_id)
    tarfile_obj = tarfile.open(tarfile_path)
    shutil.rmtree('/tmp/{}'.format(exp_id), ignore_errors=True)
    os.mkdir('/tmp/{}'.format(exp_id))
    tarfile_obj.extractall('/tmp/{}/'.format(exp_id))


def parse_train_line(line):
    splitted = line.split()
    try:
        accuracy = float(splitted[5])
        mac = float(splitted[7])
    except ValueError as e:
        print('Value error parsing line: {}'.format(line))
        raise e
    except IndexError as e:
        print('Unexpected line size ({}): {}'.format(len(splitted), line))
        raise e
    return mac, accuracy


def parse_valid_line(line):
    splitted = line.split()
    try:
        accuracy = float(splitted[5])
    except ValueError as e:
        print('Value error parsing line: {}'.format(line))
        raise e
    except IndexError as e:
        print('Unexpected line size ({}): {}'.format(len(splitted), line))
        raise e
    return accuracy


def build_from_exp_data(exp_path):
    macs = []
    train_acc_top1 = []
    file_pt = open(exp_path + '/log.txt', 'r')
    while True:
        line = file_pt.readline()
        if not line:
            break
        if ' train ' in line and 'arch' not in line:
            mac, _ = parse_train_line(line)
            macs.append(mac)
        if ' valid ' in line:
            accuracy = parse_valid_line(line)
            train_acc_top1.append(accuracy)
    file_pt.close()
    if len(macs) != len(train_acc_top1):
        raise RuntimeError('Different size for macs and accs')
    return train_acc_top1, macs


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': LEGEND_FONT_SIZE})
    # for arch_id, arch in {'M71': arch_collection.archs['M71']}.items():
    for arch_id in ['M39', 'M44', 'M49']:
        arch = arch_collection.archs[arch_id]
        if arch.closs_v not in VERSION_NAMES:
            continue
        exp_id = arch.train_search_id
        extract_experiment_data_to_tmp(exp_id)
        try:
            accs, macs = build_from_exp_data('/tmp/{}'.format(exp_id))
        except Exception as e:
            print('Arch from {} ({}, {}) not plottable'.format(exp_id, arch.arch_id, arch.closs_v))
            print('Error: {}'.format(e))
            continue
        fig = configure_plot(accs, macs,
                             'Dados de Treinamento para {} ({})'.format(arch.arch_id, VERSION_NAMES[arch.closs_v]))
        plt.savefig('{}({})-training-data.pdf'.format(arch.arch_id, arch.closs_v), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
