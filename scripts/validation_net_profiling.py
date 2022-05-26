import gc
import os

import torch
import thop

import genotypes
from model import NetworkCIFAR
from op_oracle import FPOpCounter
from scripts.arch_data import ArchDataCollection

MIN_FREE_MEM_THRESHOLD = 6000000  # So that the computer doesn't freeze due to swap memory usage


class InsufficientSysMem(RuntimeError):
    pass


def profile_arch(network):
    cpu_times = []
    cuda_times = []
    for i in range(5):
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for _ in range(20):
                network(torch.randn(1, 3, 32, 32).cuda())
        cpu_times.append(prof.total_average().cpu_time_total)
        cuda_times.append(prof.total_average().cuda_time_total)
        del prof
        print('\t\t{}%'.format((i + 1) * 20))
        check_free_sys_memory()
    return sum(cpu_times), sum(cuda_times)


def get_macs(network):
    macs, _ = thop.profile(network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
    return macs


def get_fpops(arch_id):
    counter = FPOpCounter()
    counter.setup(32, 32, 20, 36)
    counter.genotype = eval('genotypes.{}'.format(arch_id))
    return counter.count_network_fp_ops()


def check_free_sys_memory():
    if get_free_sys_memory() <= MIN_FREE_MEM_THRESHOLD:
        raise InsufficientSysMem


def get_free_sys_memory():
    return int(os.popen('free -t').readlines()[1].split()[-1])


def create_network_for_measures(arch):
    network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch.arch_id))).cuda()
    network.drop_path_prob = 0.0
    return network


def assess_and_set_macs(arch):
    if not hasattr(arch, 'macs_count') or not arch.macs_count:
        network = create_network_for_measures(arch)
        print('\tCounting MACs for arch {}'.format(arch.arch_id))
        arch.macs_count = get_macs(network)
        del network
        return True
    return False


def assess_and_set_inf_time(arch):
    if not hasattr(arch, 'time_for_100_inf') or not arch.time_for_100_inf:
        network = create_network_for_measures(arch)
        print('\tProfiling arch {}'.format(arch.arch_id))
        cpu_time, cuda_time = profile_arch(network)
        arch.time_for_100_inf = cuda_time
        del network
        return True
    return False


def assess_and_set_fp_op(arch):
    if not hasattr(arch, 'fp_op_count') or not arch.fp_op_count:
        print('\tCounting FPOPs for arch {}'.format(arch.arch_id))
        arch.fp_op_count = get_fpops(arch.arch_id)
        return True
    return False


def main():
    assert type(genotypes.M1) == genotypes.Genotype
    assert torch.cuda.is_available()
    collection = ArchDataCollection()
    collection.load()
    for arch in collection.archs.values():
        check_free_sys_memory()
        print('Checking arch {}...'.format(arch.arch_id))
        collection_modified = assess_and_set_macs(arch)
        collection_modified = assess_and_set_inf_time(arch) or collection_modified
        collection_modified = assess_and_set_fp_op(arch) or collection_modified
        if collection_modified:
            print('\tSaving collection')
            collection.save()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
