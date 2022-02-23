import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection, CLossV


def is_in_pareto_f(a1, plot_data):
    for a2 in plot_data:
        if a1 is a2:
            continue
        if a2.model_acc > a1.model_acc and a2.macs_count < a1.macs_count:
            return False
    return True


def add_extremities(frontier_elements):
    class DummyArch:
        def __init__(self, model_acc, macs_count):
            self.model_acc = model_acc
            self.macs_count = macs_count

    first = DummyArch(93, frontier_elements[0].macs_count)
    last = DummyArch(frontier_elements[-1].model_acc, 8e8)
    frontier_elements.insert(0, first)
    frontier_elements.append(last)


def draw_pareto_frontier(plot_data, ax):
    frontier_elements = []
    for a in plot_data:
        if is_in_pareto_f(a, plot_data):
            frontier_elements.append(a)
    frontier_elements = sorted(frontier_elements, key=lambda arch: arch.macs_count)
    add_extremities(frontier_elements)
    ax.plot(
        [a.model_acc for a in frontier_elements], [a.macs_count for a in frontier_elements],
        linestyle='dashed')


def plot_acc_vs_macs(collection):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Model Accuracy (from train.py)')
    plt.ylabel('# MACS')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V3, CLossV.D_LOSS_V4),
                              'Original': (CLossV.ORIGINAL,)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.macs_count is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.model_acc for a in plot_data], [a.macs_count for a in plot_data], label=g_name)
        draw_pareto_frontier(plot_data, ax)
        for a in plot_data:
            x = a.model_acc
            y = a.macs_count
            s = a.arch_id
            s += ', w={}'.format(a.closs_w) if hasattr(a, "closs_w") and a.closs_w else ''
            plt.text(x=x, y=y, s=s, fontsize=3)
    ax.legend()
    plt.savefig('pareto.svg')


def plot_acc_vs_w(collection):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Custom Loss Weight \"w\"')
    plt.ylabel('Model Accuracy (from train.py)')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V3, CLossV.D_LOSS_V4)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.closs_w for a in plot_data], [a.model_acc for a in plot_data], label=g_name)
        for a in plot_data:
            x = a.closs_w
            y = a.model_acc
            s = a.arch_id
            plt.text(x=x, y=y, s=s, fontsize=6)
    ax.legend()
    plt.savefig('acc_vs_w.svg')


def plot_macs_vs_w(collection):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Custom Loss Weight \"w\"')
    plt.ylabel('# MACS')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V3, CLossV.D_LOSS_V4)}.items():
        plot_data = list(
            filter(lambda a: a.macs_count is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.closs_w for a in plot_data], [a.macs_count for a in plot_data], label=g_name)
        for a in plot_data:
            x = a.closs_w
            y = a.macs_count
            s = a.arch_id
            plt.text(x=x, y=y, s=s, fontsize=6)
    ax.legend()
    plt.savefig('macs_vs_w.svg')


if __name__ == '__main__':
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plot_acc_vs_macs(arch_collection)
    plot_acc_vs_w(arch_collection)
    plot_macs_vs_w(arch_collection)
