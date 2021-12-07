import matplotlib
import matplotlib.pyplot as plt

from scripts.arch_data import ArchDataCollection


def get_avg_macs(arch_collection, range_min, range_max):
    archs = arch_collection.archs
    macs_list = [archs['M{}'.format(i)].macs_count for i in range(range_min, 4)]
    macs_list.extend([archs['M{}'.format(i)].macs_count for i in range(5, 18)])
    macs_list.extend([archs['M{}'.format(i)].macs_count for i in range(19, range_max)])
    return sum(macs_list) / len(macs_list)


def get_avg_macs_before_diff_loss(plot_data):
    return get_avg_macs(plot_data, 2, 37)


def plot_arch_data(arch_collection):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    avg_macs = get_avg_macs_before_diff_loss(arch_collection)
    plt.xlabel('Custom Loss w')
    plt.ylabel('# MACS using thop')
    archs = arch_collection.archs
    plot_data_t = [(a.arch_id, a.git_hash, a.closs_w, a.macs_count) for a in
                   filter(lambda a: hasattr(a, 'closs_w'), archs.values())]
    hashes = set([a[1] for a in plot_data_t])
    for h in hashes:
        ax.plot([a[2] for a in filter(lambda a: a[1] == h, plot_data_t)],
                [a[3] for a in filter(lambda a: a[1] == h, plot_data_t)], label=h)
    ax.plot([plot_data_t[0][2], plot_data_t[-1][2]], [avg_macs, avg_macs], "k--")
    plt.xscale('log')
    ax.legend()
    for i in range(len(plot_data_t)):
        x = plot_data_t[i][2]
        y = plot_data_t[i][3]
        plt.text(x=x, y=y, s=plot_data_t[i][0])
    plt.savefig('arch_macs_vs_closs_w.png')


def update_closs_w(archs):
    for i in range(47, 53):
        archs['M{}'.format(i)].git_hash = 0x294a81589ceb092e908ab9bc1453711e759b4ea1
    archs['M47'].closs_w = 5e-8
    archs['M48'].closs_w = 2e-8
    archs['M49'].closs_w = 1e-8
    archs['M50'].closs_w = 5e-9
    archs['M51'].closs_w = 2e-9
    archs['M52'].closs_w = 1e-9
    for i in range(56, 62):
        archs['M{}'.format(i)].git_hash = 0xc1740a8e22034c6818d2ca5a80d6c3ae90284d53
    archs['M56'].closs_w = 5e-8
    archs['M57'].closs_w = 2e-8
    archs['M58'].closs_w = 1e-8
    archs['M59'].closs_w = 5e-9
    archs['M60'].closs_w = 2e-9
    archs['M61'].closs_w = 1e-9


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    archs = arch_collection.archs
    update_closs_w(archs)
    arch_collection.save()
    plot_arch_data(arch_collection)


if __name__ == '__main__':
    main()
