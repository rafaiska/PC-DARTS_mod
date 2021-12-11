import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection


def plot_acc_vs_macs(collection):
    plot_data_1 = {}
    plot_data_2 = {}
    for a_id in collection.archs:
        n = int(a_id[1:])
        if collection.archs[a_id].model_acc is not None:
            if n < 45:
                plot_data_1[a_id] = (collection.archs[a_id].model_acc, collection.archs[a_id].macs_count)
            else:
                plot_data_2[a_id] = (collection.archs[a_id].model_acc, collection.archs[a_id].macs_count)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Model Accuracy (from train.py)')
    plt.ylabel('# MACS')
    ax.scatter([plot_data_1[a][0] for a in plot_data_1], [plot_data_1[a][1] for a in plot_data_1], color='darkgreen',
               marker='^')
    ax.scatter([plot_data_2[a][0] for a in plot_data_2], [plot_data_2[a][1] for a in plot_data_2], color='red',
               marker='o')
    for plot_data in [plot_data_1, plot_data_2]:
        for a in plot_data:
            x = plot_data[a][0]
            y = plot_data[a][1]
            plt.text(x=x, y=y, s=a)
    plt.savefig('pareto.svg')


if __name__ == '__main__':
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plot_acc_vs_macs(arch_collection)