import torch

import genotypes
from model import NetworkCIFAR


def profile_arch(arch_id, log):
    network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id))).cuda()
    network.drop_path_prob = 0.0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(torch.randn(1, 3, 32, 32).cuda())
    log[arch_id] = (prof.total_average().cpu_time_total, prof.total_average().cuda_time_total)


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    log = {}
    for arch_id in [1, 2, 3, 5, 6, 7, 8, 9]:
        profile_arch('M{}'.format(arch_id), log)
    for a in log:
        print('{}: {}'.format(a, log[a]))


if __name__ == '__main__':
    main()
