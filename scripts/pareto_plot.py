import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection, CLossV


def plot_acc_vs_macs(collection):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Model Accuracy (from train.py)')
    plt.ylabel('# MACS')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V3, CLossV.D_LOSS_V4), 'Original': (CLossV.ORIGINAL,)}.items():
        plot_data = list(filter(lambda a: a.model_acc is not None and a.closs_v in clv_group, collection.archs.values()))
        ax.scatter([a.model_acc for a in plot_data], [a.macs_count for a in plot_data], label=g_name)
        for a in plot_data:
            x = a.model_acc
            y = a.macs_count
            s = a.arch_id
            s += ', w={}'.format(a.closs_w) if hasattr(a, "closs_w") and a.closs_w else ''
            plt.text(x=x, y=y, s=s, fontsize=5)
    ax.legend()
    plt.savefig('pareto.svg')


if __name__ == '__main__':
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plot_acc_vs_macs(arch_collection)
