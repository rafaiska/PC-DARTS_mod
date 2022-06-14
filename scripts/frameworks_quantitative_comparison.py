import numpy as np

from scripts.arch_data import ArchDataCollection, CLossV

ACC_DELTA = 0.5


def get_mean(collection, attribute, closs_v, acc_range=None):
    archs = collection.select((closs_v,)).values()
    if acc_range:
        archs = list(filter(lambda x: acc_range[0] <= x.model_acc <= acc_range[1], archs))
    attr_list = [a.__getattribute__(attribute) for a in archs]
    return np.average(attr_list), np.std(attr_list)


def print_avg(avg_tuple):
    print('{:e} +- {:e}'.format(*avg_tuple))


def main():
    collection = ArchDataCollection()
    collection.load()
    pc_darts_mean_accuracy = get_mean(collection, 'model_acc', CLossV.ORIGINAL)
    pc_darts_mean_macs = get_mean(collection, 'macs_count', CLossV.ORIGINAL)
    mopc_darts_mean_macs = get_mean(collection, 'macs_count', CLossV.D_LOSS_V5,
                                    (pc_darts_mean_accuracy[0] - ACC_DELTA, pc_darts_mean_accuracy[0]))
    print_avg(pc_darts_mean_accuracy)
    print_avg(pc_darts_mean_macs)
    print_avg(mopc_darts_mean_macs)
    print(1 - mopc_darts_mean_macs[0] / pc_darts_mean_macs[0])


if __name__ == '__main__':
    main()
