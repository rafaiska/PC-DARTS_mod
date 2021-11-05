import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def configure_plot(exp_id, fig, accs, macs, ax):
    ax.plot([x / 2 for x in range(len(accs))], accs, label='{} Acc'.format(exp_id))
    ax.plot([x / 2 for x in range(len(macs))], [m / 10e6 for m in macs], label='{} MACS'.format(exp_id))
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


def plot_subgraph(exp_ids, fig, ax):
    for exp_id in exp_ids:
        accs, macs = build_from_exp_data('/tmp/{}'.format(exp_id))
        _, ax = configure_plot(exp_id, fig, accs, macs, ax)


def is_custom_loss_on(exp_id):
    exp_path = '/tmp/{}'.format(exp_id)
    with open(exp_path + '/log.txt', 'r') as fp:
        for line in fp:
            if "args = Namespace" in line:
                return "custom_loss=True" in line
    raise RuntimeError("Invalid EXP log file")


def configure_axes(closs_ax, wocloss_ax):
    closs_ax.set_title('Training data using custom loss')
    wocloss_ax.set_title('Training data without custom loss')
    for ax in [closs_ax, wocloss_ax]:
        ax.autoscale()
        ax.set_ylabel('MACS (10e6), Top1 Acc')
        ax.set_xlabel('Training epoch')


def main():
    fig = plt.figure(figsize=(16, 8))
    closs_ax = fig.add_subplot(211)
    wocloss_ax = fig.add_subplot(212)
    configure_axes(closs_ax, wocloss_ax)
    exp_ids = sys.argv[1:]
    for e_id in exp_ids:
        extract_experiment_data_to_tmp(e_id)
    closs_exp_ids = list(filter(lambda x: is_custom_loss_on(x), exp_ids))
    wocloss_exp_ids = list(filter(lambda x: not is_custom_loss_on(x), exp_ids))
    plot_subgraph(closs_exp_ids, fig, closs_ax)
    plot_subgraph(wocloss_exp_ids, fig, wocloss_ax)

    plt.subplots_adjust(hspace=0.3)
    plt.savefig('mac-acc-training-data.svg')


if __name__ == '__main__':
    main()
