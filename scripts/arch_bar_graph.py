import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.arch_data import ArchDataCollection, CLossV

VERSION_TO_COLORS = {CLossV.ORIGINAL: 'red',
                     CLossV.BOGUS_ORIGINAL: 'orange',
                     CLossV.LEGACY: 'yellow',
                     CLossV.D_LOSS_V1: 'green',
                     CLossV.D_LOSS_V2: 'cyan',
                     CLossV.D_LOSS_V3: 'blue',
                     CLossV.D_LOSS_V4: 'purple',
                     CLossV.D_LOSS_V5: 'black'}


def plot_arch_macs(plot_data):
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(111)
    archs = plot_data.archs.values()
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


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plot_arch_macs(arch_collection)


if __name__ == '__main__':
    main()
