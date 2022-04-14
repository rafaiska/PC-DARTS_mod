import thop
import torch
import gc

import genotypes
from model import NetworkCIFAR
from op_oracle import FPOpCounter
from scripts.arch_data import ArchDataCollection

MAX_NUMBER_OF_ARCHS_PROCESSED = 4  # So that the computer doesn't freeze due to swap memory usage


def profile_arch(network):
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(100):
            network(torch.randn(1, 3, 32, 32).cuda())
    return prof.total_average().cpu_time_total, prof.total_average().cuda_time_total


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    return macs


def get_fpops(arch_id):
    counter = FPOpCounter()
    counter.setup(32, 32, 20, 36)
    counter.genotype = eval('genotypes.{}'.format(arch_id))
    return counter.count_network_fp_ops()


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    collection = ArchDataCollection()
    collection.load()
    counter = 0
    for arch_id_str, arch in collection.archs.items():
        if counter >= MAX_NUMBER_OF_ARCHS_PROCESSED:
            break
        print('Checking arch {}...'.format(arch_id_str))
        assessed = False
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        network.drop_path_prob = 0.0
        if not hasattr(arch, 'macs_count') or not arch.macs_count:
            print('\tCounting MACs for arch {}'.format(arch_id_str))
            arch.macs_count = get_macs(network)
            assessed = True
        if not hasattr(arch, 'time_for_100_inf') or not arch.time_for_100_inf:
            cpu_time, cuda_time = profile_arch(network)
            print('\tProfiling arch {}'.format(arch_id_str))
            arch.time_for_100_inf = cuda_time
            assessed = True
        if not hasattr(arch, 'fp_op_count'):
            print('\tCounting FPOPs for arch {}'.format(arch_id_str))
            arch.fp_op_count = get_fpops(arch_id_str)
            assessed = True
        if assessed:
            counter += 1
            print('\tSaving collection')
            collection.save()
        torch.cuda.empty_cache()
        del network
        gc.collect()


if __name__ == '__main__':
    main()
