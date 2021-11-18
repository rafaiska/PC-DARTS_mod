import os
import shutil
import tarfile

import matplotlib.pyplot as plt

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'
EXP_WO_CLOSS = ['search-EXP-20211105-143218', 'search-EXP-20211106-152514']
EXP_CLOSS_XKW2 = ['search-EXP-20211106-152513', 'search-EXP-20211105-120555']
EXP_CLOSS_XKW10 = ['search-EXP-20211107-092623', 'search-EXP-20211107-092453']
EXP_CLOSS_KW_W2 = ['search-EXP-20211107-233004', 'search-EXP-20211107-233603']
EXP_CLOSS_LOG1MK = ['search-EXP-20211109-101420', 'search-EXP-20211109-102212']
EXP_CLOSS_LOG1MK_ARCH_WARMUP = ['search-EXP-20211112-091217', 'search-EXP-20211112-065726']
EXP_CLOSS_OPORACLE_W10 = ['search-EXP-20211111-091754', 'search-EXP-20211111-093439']
EXP_CLOSS_OPORACLE_W2 = ['search-EXP-20211111-193703', 'search-EXP-20211111-195217']
EXP_CLOSS_DIFF_V1 = ['search-EXP-20211117-234131', 'search-EXP-20211117-234203']


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
            mac = float(splitted[7])
            macs.append(mac)
        elif ' valid ' in line:
            splitted = line.split()
            accuracy = float(splitted[5])
            train_acc_top1.append(accuracy)

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


def configure_axes(title, closs_ax, wocloss_ax):
    closs_ax.set_title('Training data using {}'.format(title))
    wocloss_ax.set_title('Training data without custom loss')
    for ax in [closs_ax, wocloss_ax]:
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 100)
        ax.set_ylabel('MACS (10e6), Top1 Acc')
        ax.set_xlabel('Training epoch')


def plot_case(function_title, exp_ids):
    fig = plt.figure(figsize=(16, 8))
    closs_ax = fig.add_subplot(211)
    wocloss_ax = fig.add_subplot(212)
    configure_axes(function_title, closs_ax, wocloss_ax)
    for e_id in exp_ids:
        extract_experiment_data_to_tmp(e_id)
    closs_exp_ids = list(filter(lambda x: is_custom_loss_on(x), exp_ids))
    wocloss_exp_ids = list(filter(lambda x: not is_custom_loss_on(x), exp_ids))
    plot_subgraph(closs_exp_ids, fig, closs_ax)
    plot_subgraph(wocloss_exp_ids, fig, wocloss_ax)

    plt.subplots_adjust(hspace=0.3)
    plt.savefig('Training_with_{}.svg'.format('_'.join(function_title.split())))


def main():
    plot_case('x + x*k*w Loss Function with w=2', [*EXP_CLOSS_XKW2, *EXP_WO_CLOSS])
    plot_case('x + x*k*w Loss Function with w=10', [*EXP_CLOSS_XKW10, *EXP_WO_CLOSS])
    plot_case('x + k*w Loss Function with w=2', [*EXP_CLOSS_KW_W2, *EXP_WO_CLOSS])
    plot_case('x + min(-log(1-k) * w, max_cl) Loss Function with w=2', [*EXP_CLOSS_LOG1MK, *EXP_WO_CLOSS])
    plot_case('x + min(-log(1-k) * w, max_cl) Loss Function with w=2 and arch warmup',
              [*EXP_CLOSS_LOG1MK_ARCH_WARMUP, *EXP_WO_CLOSS])
    plot_case('MACs based Op. Oracle with w=10', [*EXP_CLOSS_OPORACLE_W10, *EXP_WO_CLOSS])
    plot_case('MACs based Op. Oracle with w=2', [*EXP_CLOSS_OPORACLE_W2, *EXP_WO_CLOSS])
    plot_case('Differentiable Loss Function V1 with w=2div10e6', [*EXP_CLOSS_DIFF_V1, *EXP_WO_CLOSS])


if __name__ == '__main__':
    main()
