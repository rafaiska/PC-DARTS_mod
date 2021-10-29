import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def configure_plot(accs, macs, title):
    fig = plt.figure(figsize=(16, 8))
    # Creating subplot/axes
    ax = fig.add_subplot(111)

    # Setting axes/plot title
    ax.set_title(title)

    # Setting X-axis and Y-axis limits
    #     ax.set_xlim([1, len(data['dil_conv_3x3'])])
    #     ax.set_ylim([0.1, 0.25])
    ax.autoscale()

    # Setting X-axis and Y-axis labels
    ax.set_ylabel('MACS (10e6), Top1 Acc')
    ax.set_xlabel('Training step')

    plt.plot([x for x in range(len(accs))], accs, label='Acc')
    plt.plot([x for x in range(len(macs))], [m / 10e6 for m in macs], label='MACS (10e6)')
    ax.legend()

    return fig, ax


def extract_experiment_data_to_tmp(exp_id):
    tarfile_path = '{}/{}.tar.gz'.format(EXP_DIR, exp_id)
    tarfile_obj = tarfile.open(tarfile_path)
    shutil.rmtree('/tmp/{}'.format(exp_id), ignore_errors=True)
    os.mkdir('/tmp/{}'.format(exp_id))
    tarfile_obj.extractall('/tmp/{}/'.format(exp_id))


def build_from_exp_data(exp_path):
    macs = []
    train_acc_top1 = []
    file_pt = open(exp_path + '/log.txt', 'r')
    while True:
        line = file_pt.readline()
        if not line:
            break
        if ' train ' in line:
            splitted = line.split()
            accuracy = float(splitted[5])
            train_acc_top1.append(accuracy)
            mac = float(splitted[7])
            macs.append(mac)
    file_pt.close()
    return train_acc_top1, macs


def main():
    exp_id = sys.argv[1]
    extract_experiment_data_to_tmp(exp_id)
    accs, macs = build_from_exp_data('/tmp/{}'.format(exp_id))
    configure_plot(accs, macs, 'Training data for {}'.format(exp_id))
    plt.savefig('{}-training-data.png'.format(exp_id))


if __name__ == '__main__':
    main()
