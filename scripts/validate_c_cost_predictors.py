import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from scripts.arch_data import ArchDataCollection

TEXT_WIDTH = 6.32283486112
FIGSIZE = (TEXT_WIDTH, TEXT_WIDTH / 2.0)
DOT_SIZE = 8
LEGEND_FONT_SIZE = 9
DOT_FONT_SIZE = 3


def plot_curve(fit, fit_type, ax, x_range, color):
    x_v = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    if fit_type == 'log':
        y = [fit[1] + fit[0] * np.log(x) for x in x_v]
    elif fit_type == 'linear':
        y = [fit[1] + fit[0] * x for x in x_v]
    else:
        raise RuntimeError('Invalid fit type')
    ax.plot(x_v, y, color=color)


def plot_lin_regression(x, y, ax, color='red'):
    fit = linregress(x, y)
    print('Lin fit: {} + {} * x, r_squared = {}'.format(fit[1], fit[0], fit[2]))
    plot_curve(fit, 'linear', ax, (min(x), max(x)), color)


def plot_fp_vs_perf(arch_collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plt.xlabel('Tempo de 100 inferências (µs)')
    plt.ylabel('Número de Operações de Ponto Flutuante')
    archs = list(filter(lambda a: hasattr(a, 'fp_op_count') and a.fp_op_count, arch_collection.archs.values()))
    ax.scatter([a.fp_op_count for a in archs], [a.time_for_100_inf for a in archs], color='red', marker='o', s=DOT_SIZE)
    plot_lin_regression([a.fp_op_count for a in archs], [a.time_for_100_inf for a in archs], ax, color='blue')
    # for a in archs:
    #     x = a.fp_op_count
    #     y = a.time_for_100_inf
    #     plt.text(x=x, y=y, s=a.arch_id, fontsize=DOT_FONT_SIZE)
    plt.savefig('arch_fp_op_vs_perf.pdf', bbox_inches='tight')


def plot_mac_vs_perf(arch_collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plt.xlabel('Tempo de 100 inferências (µs)')
    plt.ylabel('# MACs Usando thop')
    archs = list(filter(lambda a: hasattr(a, 'macs_count') and a.macs_count, arch_collection.archs.values()))
    ax.scatter([a.macs_count for a in archs], [a.time_for_100_inf for a in archs], color='blue', marker='o', s=DOT_SIZE)
    plot_lin_regression([a.macs_count for a in archs], [a.time_for_100_inf for a in archs], ax)
    # for a in archs:
    #     x = a.macs_count
    #     y = a.time_for_100_inf
    #     plt.text(x=x, y=y, s=a.arch_id, fontsize=DOT_FONT_SIZE)
    plt.savefig('arch_mac_vs_perf.pdf', bbox_inches='tight')


def count_archs(collection):
    arch_list = list(filter(lambda a: hasattr(a, 'macs_count') and hasattr(a, 'fp_op_count') and a.macs_count,
                            collection.archs.values()))
    print('Number of architectures:', len(arch_list))


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': LEGEND_FONT_SIZE})
    count_archs(arch_collection)
    plot_fp_vs_perf(arch_collection)
    plot_mac_vs_perf(arch_collection)


if __name__ == '__main__':
    main()
