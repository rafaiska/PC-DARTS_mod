import matplotlib.pyplot as plt
import numpy
import thop
import torch

import genotypes
from model import NetworkCIFAR
from op_oracle import FPOpCounter


def get_avg_macs(plot_data, range_min, range_max):
    macs_list = [plot_data['M{}'.format(i)] for i in range(range_min, 4)]
    macs_list.extend([plot_data['M{}'.format(i)] for i in range(5, 18)])
    macs_list.extend([plot_data['M{}'.format(i)] for i in range(19, range_max)])
    return sum(macs_list) / len(macs_list)


def plot_arch_macs(plot_data):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    avg_macs = get_avg_macs(plot_data, 1, 36)
    plt.xlabel('Arch')
    plt.ylabel('# MACS using thop')
    plot_data_t = [k for k in plot_data.items()]
    ax.bar([pd[0] for pd in plot_data_t], [pd[1] for pd in plot_data_t])
    ax.plot([plot_data_t[0][0], plot_data_t[-1][0]], [avg_macs, avg_macs], "k--")
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    plt.savefig('arch_macs.png')


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    return macs


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    log = {}
    for arch_id in [1, 2, 3, *list(range(5, 18)), *list(range(19, 56))]:
        arch_id_str = 'M{}'.format(arch_id)
        print('Profiling arch {}'.format(arch_id_str))
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        network.drop_path_prob = 0.0
        fc = FPOpCounter()
        fc.setup(32, 32, 20, 36)
        fc.genotype = eval('genotypes.{}'.format(arch_id_str))
        # fp_ops = fc.count_network_fp_ops()
        macs = get_macs(network)
        log[arch_id_str] = macs
        torch.cuda.empty_cache()
        print(macs)
    plot_arch_macs(log)


if __name__ == '__main__':
    main()
