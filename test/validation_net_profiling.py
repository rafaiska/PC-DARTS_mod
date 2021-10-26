import thop
import torch

import genotypes
from model import NetworkCIFAR
from op_oracle import FPOpCounter


def profile_arch(network):
    network.drop_path_prob = 0.0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(torch.randn(1, 3, 32, 32).cuda())
    return prof.total_average().cpu_time_total, prof.total_average().cuda_time_total


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    return macs


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    log = {}
    for arch_id in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
        arch_id_str = 'M{}'.format(arch_id)
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        cpu_time, cuda_time = profile_arch(network)
        fc = FPOpCounter()
        fc.setup(32, 32, 20, 36)
        fc.genotype = eval('genotypes.{}'.format(arch_id_str))
        fp_ops = fc.count_network_fp_ops()
        macs = get_macs(network)
        log[arch_id_str] = (cpu_time, cuda_time, fp_ops, macs)
    with open('arch_data.csv', 'w') as fp:
        for a in log:
            fp.write('{}, {}, {}, {}, {}\n'.format(a, log[a][0], log[a][1], log[a][2], log[a][3]))


if __name__ == '__main__':
    main()
