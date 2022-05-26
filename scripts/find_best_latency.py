from scripts.arch_data import ArchDataCollection, CLossV
from math import inf


def main():
    collection = ArchDataCollection()
    collection.load()
    best_latency = inf
    worst_latency = 0
    for a in collection.select((CLossV.D_LOSS_V5,)).values():
        if a.time_for_100_inf:
            latency = a.time_for_100_inf / 100 / 1e3
            if latency < best_latency:
                best_latency = latency
            if latency > worst_latency:
                worst_latency = latency
    print(best_latency, worst_latency)


if __name__ == '__main__':
    main()
