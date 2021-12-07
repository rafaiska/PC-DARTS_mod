import pickle

import matplotlib.pyplot as plt
import numpy as np
import thop
import torch

import genotypes
from model import NetworkCIFAR
from scripts.arch_data import ArchDataCollection


def plot_arch_macs(plot_data, plot_time=True):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    archs = plot_data.archs
    plot_data_t = [(a, archs[a].macs_count, archs[a].time_for_100_inf) for a in archs]
    labels = [pd[0] for pd in plot_data_t]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.xlabel('Arch')
    plt.ylabel('MACS * 3e-2 using thop{}'.format(', Inf. time (us)' if plot_time else ''))
    ax.bar(x - width / 2, [pd[1] * 3e-2 for pd in plot_data_t], width, label='MACS')
    if plot_time:
        ax.bar(x + width / 2, [pd[2] for pd in plot_data_t], width, label='time')
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
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


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    arch_collection = ArchDataCollection()
    arch_collection.load()
    for arch_id in arch_collection.archs:
        if arch_collection.archs[arch_id].time_for_100_inf and arch_collection.archs[arch_id].macs_count:
            continue
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id))).cuda()
        network.drop_path_prob = 0.0
        if not arch_collection.archs[arch_id].time_for_100_inf:
            print('Profiling arch {}'.format(arch_id))
            arch_collection.archs[arch_id].time_for_100_inf = get_inference_time(network)
            torch.cuda.empty_cache()
            arch_collection.save()
        if not arch_collection.archs[arch_id].macs_count:
            print('Calculating MACS for arch {}'.format(arch_id))
            arch_collection.archs[arch_id].macs_count = get_macs(network)
            arch_collection.save()
    plot_arch_macs(arch_collection, plot_time=False)


if __name__ == '__main__':
    main()
