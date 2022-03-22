import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from scripts.arch_data import ArchDataCollection

FIGSIZE = (12, 5)


def plot_curve(fit, fit_type, ax, x_range):
    x_v = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    if fit_type == 'log':
        y = [fit[1] + fit[0] * np.log(x) for x in x_v]
    elif fit_type == 'linear':
        y = [fit[1] + fit[0] * x for x in x_v]
    else:
        raise RuntimeError('Invalid fit type')
    ax.plot(x_v, y, color='red')


def plot_lin_regression(x, y, ax):
    fit = linregress(x, y)
    print('Lin fit: {} + {} * x, r_squared = {}'.format(fit[1], fit[0], fit[2]))
    plot_curve(fit, 'linear', ax, (min(x), max(x)))


def plot_fp_vs_perf(arch_collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('# floating point ops')
    archs = list(filter(lambda a: hasattr(a, 'fpop_count') and a.fpop_count, arch_collection.archs.values()))
    ax.scatter([a.fpop_count for a in archs], [a.time_for_100_inf for a in archs], color='red', marker='o')
    plot_lin_regression([a.fpop_count for a in archs], [a.time_for_100_inf for a in archs], ax)
    for a in archs:
        x = a.fpop_count
        y = a.time_for_100_inf
        plt.text(x=x, y=y, s=a.arch_id)
    plt.savefig('arch_fp_op_vs_perf.pdf')


def plot_mac_vs_perf(arch_collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('# MACs using thop')
    archs = list(filter(lambda a: hasattr(a, 'macs_count') and a.macs_count, arch_collection.archs.values()))
    ax.scatter([a.macs_count for a in archs], [a.time_for_100_inf for a in archs], color='blue', marker='o')
    plot_lin_regression([a.macs_count for a in archs], [a.time_for_100_inf for a in archs], ax)
    for a in archs:
        x = a.macs_count
        y = a.time_for_100_inf
        plt.text(x=x, y=y, s=a.arch_id, fontsize=6)
    plt.savefig('arch_mac_vs_perf.pdf', bbox_inches='tight')


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    # plot_fp_vs_perf(arch_collection)
    plot_mac_vs_perf(arch_collection)


if __name__ == '__main__':
    main()
