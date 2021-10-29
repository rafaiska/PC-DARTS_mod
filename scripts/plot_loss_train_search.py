import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def configure_plot(ce_loss_history, custom_loss_history, title):
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
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Training step')

    plt.plot([x for x in range(len(ce_loss_history))], ce_loss_history, label='Cross Entropy Loss')
    plt.plot([x for x in range(len(custom_loss_history))], custom_loss_history, label='Custom Loss')
    ax.legend()

    return fig, ax


def extract_experiment_data_to_tmp(exp_id):
    tarfile_path = '{}/{}.tar.gz'.format(EXP_DIR, exp_id)
    tarfile_obj = tarfile.open(tarfile_path)
    shutil.rmtree('/tmp/{}'.format(exp_id), ignore_errors=True)
    os.mkdir('/tmp/{}'.format(exp_id))
    tarfile_obj.extractall('/tmp/{}/'.format(exp_id))


def build_from_exp_data(exp_path):
    ce_loss_history = []
    custom_loss_history = []
    file_pt = open(exp_path + '/log.txt', 'r')
    while True:
        line = file_pt.readline()
        if not line:
            break
        if ' LOSS = ' in line:
            line = file_pt.readline()
            ce_loss = float(line.split()[0])
            ce_loss_history.append(ce_loss)
            for _ in range(2):
                file_pt.readline()
            line = file_pt.readline()
            custom_loss = float(line.split()[0])
            custom_loss_history.append(custom_loss)
    file_pt.close()
    return ce_loss_history, custom_loss_history


def main():
    exp_id = sys.argv[1]
    extract_experiment_data_to_tmp(exp_id)
    ce_loss_history, custom_loss_history = build_from_exp_data('/tmp/{}'.format(exp_id))
    configure_plot(ce_loss_history, custom_loss_history, 'Loss values for {}'.format(exp_id))
    plt.savefig('{}-training-loss.png'.format(exp_id))


if __name__ == '__main__':
    main()
