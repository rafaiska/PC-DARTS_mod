import matplotlib.pyplot as plt
import thop
import torch

import genotypes
from operations import OPS

N_INPUTS = [torch.randn(1, 3, 2, 2).cuda(),
            torch.randn(1, 3, 4, 4).cuda(),
            torch.randn(1, 3, 8, 8).cuda(),
            torch.randn(1, 3, 16, 16).cuda(),
            torch.randn(1, 3, 32, 32).cuda(),
            torch.randn(1, 3, 64, 64).cuda(),
            torch.randn(1, 3, 128, 128).cuda(),
            ]


def profile_arch(network, n_input):
    network.drop_path_prob = 0.0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(n_input)
    return prof.total_average().cpu_time_total, prof.total_average().cuda_time_total


def get_macs(network, n_input):
    macs, _ = thop.profile(network, inputs=(n_input,), verbose=False)
    return macs


def plot_op_data(op_data, title, index, ylabel):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.autoscale()

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Input')

    for op in op_data:
        plt.plot([str(x.shape) for x in N_INPUTS], [o[index] for o in op_data[op]], label=op)
    ax.legend()
    plt.savefig('{}.png'.format('_'.join(title.split())))


def plot_time_vs_macs_for_each_op(op_data):
    for op in op_data:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        title = 'Time VS MACS for {} operator'.format(op)
        ax.set_title(title)
        ax.autoscale()

        ax.set_ylabel('Time for 100 inferences (us)')
        ax.set_xlabel('MACS')
        plt.scatter([o[2] for o in op_data[op]], [o[1] for o in op_data[op]], label=op)
        for i in range(len(N_INPUTS)):
            x = op_data[op][i][2]
            y = op_data[op][i][1]
            plt.text(x=x, y=y, s=str(N_INPUTS[i].shape))
        ax.legend()
        plt.savefig('{}.png'.format('_'.join(title.split())))


def main():
    op_data = {}
    for op_name in genotypes.PRIMITIVES:
        op_constructor = OPS[op_name]
        op_data[op_name] = []
        for i in N_INPUTS:
            op = op_constructor(i.shape[1], 1, True).cuda()
            cpu_time, cuda_time = profile_arch(op, i)
            macs = get_macs(op, i)
            op_data[op_name].append((cpu_time, cuda_time, macs))
    plot_op_data(op_data, 'MACS per PC-DARTS operator', 2, 'MACS')
    plot_op_data(op_data, 'CUDA time (us) per PC-DARTS operator', 1, 'CUDA time (us)')
    plot_op_data(op_data, 'CPU time (us) per PC-DARTS operator', 0, 'CPU time (us)')
    plot_time_vs_macs_for_each_op(op_data)


if __name__ == '__main__':
    main()
