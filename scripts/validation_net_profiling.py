import thop
import torch
import gc

import genotypes
from model import NetworkCIFAR
from scripts.arch_data import ArchDataCollection


def profile_arch(network):
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
        print('Checking arch {}...'.format(arch_id_str))
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        network.drop_path_prob = 0.0
        if not hasattr(arch, 'macs_count') or not arch.macs_count:
            print('\tCounting MACs for arch {}'.format(arch_id_str))
            macs = get_macs(network)
            arch.macs_count = macs
        if not hasattr(arch, 'time_for_100_inf') or not arch.time_for_100_inf:
            cpu_time, cuda_time = profile_arch(network)
            print('\tProfiling arch {}'.format(arch_id_str))
            arch.time_for_100_inf = cuda_time
        print('\tSaving collection')
        collection.save()
        torch.cuda.empty_cache()
        del network
        gc.collect()


if __name__ == '__main__':
    main()
