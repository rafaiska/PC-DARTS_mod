import thop
import torch

import genotypes
from model import NetworkCIFAR
from scripts.arch_data import ArchDataCollection


def profile_arch(network):
    network.drop_path_prob = 0.0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(torch.randn(1, 3, 32, 32).cuda())
    return prof.total_average().cpu_time_total, prof.total_average().cuda_time_total


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    return macs


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    collection = ArchDataCollection()
    collection.load()
    for arch_id_str, arch in collection.archs.items():
        print('Profiling arch {}'.format(arch_id_str))
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        cpu_time, cuda_time = profile_arch(network)
        macs = get_macs(network)
        arch.macs_count = macs
        arch.time_for_100_inf = cuda_time
        collection.save()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
