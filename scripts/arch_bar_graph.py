import pickle

import matplotlib.pyplot as plt
import numpy as np
import thop
import torch

import genotypes
from model import NetworkCIFAR
from op_oracle import FPOpCounter


def plot_arch_macs(plot_data):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    plot_data_t = [k for k in plot_data.items()]
    labels = [pd[0] for pd in plot_data_t]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.xlabel('Arch')
    plt.ylabel('MACS * 3e-2 using thop, Inf. time (us)')
    ax.bar(x - width / 2, [pd[1][0] * 3e-2 for pd in plot_data_t], width, label='MACS')
    ax.bar(x + width / 2, [pd[1][1] for pd in plot_data_t], width, label='time')
    print(labels)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    # plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
    fig.tight_layout()
    plt.savefig('arch_bars.png')


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    return macs


def get_inference_time(network):
    network.drop_path_prob = 0.0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(torch.randn(1, 3, 32, 32).cuda())
    return prof.total_average().cuda_time_total


def load_pickled_data():
    try:
        with open('arch_data.pickle', 'rb') as fp:
            d = pickle.load(fp)
    except FileNotFoundError:
        return {}
    print('Loaded arch_data.pickle')
    return d


def save_pickled_data(data_dict):
    with open('arch_data.pickle', 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    log = load_pickled_data()
    for arch_id in [*list(range(45, 54))]:
        arch_id_str = 'M{}'.format(arch_id)
        if arch_id_str not in log:
            print('Profiling arch {}'.format(arch_id_str))
            network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
            network.drop_path_prob = 0.0
            fc = FPOpCounter()
            fc.setup(32, 32, 20, 36)
            fc.genotype = eval('genotypes.{}'.format(arch_id_str))
            # fp_ops = fc.count_network_fp_ops()
            macs = get_macs(network)
            it = get_inference_time(network)
            log[arch_id_str] = (macs, it)
            torch.cuda.empty_cache()
        else:
            print('{} was already profiled'.format(arch_id_str))
    save_pickled_data(log)
    plot_arch_macs(log)


if __name__ == '__main__':
    main()
