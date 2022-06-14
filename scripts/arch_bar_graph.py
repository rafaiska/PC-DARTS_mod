import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scripts.arch_data import ArchDataCollection, CLossV

VERSION_TO_COLORS = {CLossV.ORIGINAL: 'blue',
                     CLossV.BOGUS_ORIGINAL: 'orange',
                     CLossV.LEGACY: 'yellow',
                     CLossV.D_LOSS_V1: 'purple',
                     CLossV.D_LOSS_V2: 'cyan',
                     CLossV.D_LOSS_V3: 'green',
                     CLossV.D_LOSS_V4: 'red',
                     CLossV.D_LOSS_V5: 'red'}
TEXT_WIDTH = 6.32283486112
FIGSIZE = (TEXT_WIDTH, TEXT_WIDTH / 2.0)
FIGSIZE_WIDE = (TEXT_WIDTH * 4, TEXT_WIDTH)
LEGEND_FONT_SIZE = 9


def plot_arch_macs(plot_data):
    fig = plt.figure(figsize=FIGSIZE_WIDE)
    ax = fig.add_subplot(111)
    archs = plot_data.archs.values()
    archs = list(filter(lambda a: a.macs_count is not None, archs))
    x = [a.arch_id for a in archs]
    y = [a.macs_count for a in archs]
    colors = [VERSION_TO_COLORS[a.closs_v] for a in archs]
    plt.xlabel('Arch ID')
    plt.ylabel('MACS using thop')
    ax.bar(x, y, color=colors, label=[a.closs_v for a in archs])
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    custom_lines = [Line2D([0], [0], color=c, lw=5) for c in VERSION_TO_COLORS.values()]
    ax.legend(custom_lines, [k.name for k in VERSION_TO_COLORS.keys()])
    fig.tight_layout()
    plt.savefig('arch_bars.pdf')


def plot_original_exp_macs_avg(collection, ax):
    average_macs = np.average([a.macs_count for a in collection.archs.values() if a.closs_v == CLossV.ORIGINAL])
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y = [average_macs for _ in range(len(x))]
    ax.plot(x, y, linestyle='dashed', color=VERSION_TO_COLORS[CLossV.ORIGINAL], linewidth=4)


def convert_repeated_w_to_macs_mean(x, y):
    def _group(i_):
        macs_values = [y[i_]]
        while i_ < len(x) - 1 and x[i_] == x[i_ + 1]:
            i_ += 1
            macs_values.append(y[i_])
        if i_ == len(x) - 1 and y[i_ - 1] == y[i_]:
            macs_values.append(y[i_])
        return np.average(macs_values), len(macs_values)

    x_out = []
    y_out = []
    i = 0
    while i < len(x):
        w = x[i]
        if i < len(x) - 1 and x[i] == x[i + 1]:
            macs, i_increment = _group(i)
            i += i_increment
        else:
            macs = y[i]
            i += 1
        x_out.append(w)
        y_out.append(macs)
    return x_out, y_out


def plot_macs_comparison_closs_original(collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    archs = collection.archs.values()
    archs = list(filter(lambda a: a.macs_count is not None, archs))
    closs_archs = sorted([a for a in archs if a.closs_v in [CLossV.D_LOSS_V4, CLossV.D_LOSS_V5]],
                         key=lambda a: a.closs_w)
    x = [a.closs_w for a in closs_archs]
    y = [a.macs_count for a in closs_archs]
    x, y = convert_repeated_w_to_macs_mean(x, y)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Peso $w$ do Componente Customizado')
    plt.ylabel('Média de $N_{MACS}$')
    print(x)
    print(y)
    ax.bar([str(x_) for x_ in x], y)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    fig.tight_layout()
    plot_original_exp_macs_avg(collection, ax)
    plt.savefig('arch_macs_vs_average.pdf', bbox_inches='tight')


def plot_macs_comparison_closs_v3_v4_original(collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    archs = list(filter(lambda a: a.closs_v in (CLossV.D_LOSS_V3, CLossV.D_LOSS_V4), collection.archs.values()))
    archs = list(filter(lambda a: a.closs_w <= 2e-8, archs))
    archs = sorted(archs, key=lambda a: a.closs_w)
    colors = [VERSION_TO_COLORS[a.closs_v] for a in archs]
    x = ['{}\n{}'.format(a.arch_id, a.closs_w) for a in archs]
    y = [a.macs_count for a in archs]
    plt.xlabel('Peso $w$ do Componente Customizado')
    plt.ylabel('Média de $N_{MACS}$')
    ax.bar(x, y, color=colors, label=[a.closs_v for a in archs])
    plt.setp(ax.get_xticklabels(), fontsize=4)
    fig.tight_layout()
    plot_original_exp_macs_avg(collection, ax)
    custom_lines = [Line2D([0], [0], color=c, lw=5) for c in [VERSION_TO_COLORS[CLossV.D_LOSS_V3],
                                                              VERSION_TO_COLORS[CLossV.D_LOSS_V4]]]
    ax.legend(custom_lines, [k.name for k in [CLossV.D_LOSS_V3, CLossV.D_LOSS_V4]])
    plt.savefig('arch_macs_vs_average_v3v4.pdf')


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': LEGEND_FONT_SIZE})
    plot_arch_macs(arch_collection)
    plot_macs_comparison_closs_original(arch_collection)
    plot_macs_comparison_closs_v3_v4_original(arch_collection)


if __name__ == '__main__':
    main()
