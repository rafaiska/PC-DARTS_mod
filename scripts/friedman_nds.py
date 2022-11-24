from scripts.arch_data import ArchDataCollection, CLossV


def calculate_friedman_nds(plot_data):
    frontiers = get_set_of_ranked_pfrontiers(plot_data)
    ranks_rij = calculate_ranks(frontiers, plot_data)
    avg_ranks = calculate_avg_ranks(ranks_rij, plot_data)
    q_statistics = calculate_q_statistics(avg_ranks, plot_data)
    correction = calculate_correction(frontiers)
    return q_statistics / correction


def get_set_of_ranked_pfrontiers(plot_data):
    sets = []
    remaining = set(plot_data)
    while len(remaining) > 0:
        new_set = set()
        for a in remaining:
            if is_on_pareto_f(a, remaining):
                new_set.add(a)
        remaining -= new_set
        sets.append(new_set)
    return sets


def is_on_pareto_f(a1, plot_data):
    for a2 in plot_data:
        if a1 is a2:
            continue
        if a2.model_acc > a1.model_acc and a2.macs_count < a1.macs_count:
            return False
    return True


def calculate_ranks(frontiers, plot_data):
    r_ij = []
    for i in plot_data:
        pg = methods_better_than(i, plot_data, frontiers)
        pe = methods_equal(i, plot_data, frontiers)
        r_ij.append(pg + (pe ** 2 + pe) / 4)
    return r_ij


def methods_better_than(i, plot_data, frontiers):
    i_frontier = find_frontier_index(frontiers, i)
    methods_better = 0
    for a in plot_data:
        if a is i:
            continue
        for f in range(i_frontier):
            if a in frontiers[f]:
                methods_better += 1
                break
    return methods_better


def methods_equal(i, plot_data, frontiers):
    i_frontier = find_frontier_index(frontiers, i)
    return len(list(filter(lambda x: x is not i and x in frontiers[i_frontier], plot_data)))


def find_frontier_index(frontiers, i):
    for f in range(len(frontiers)):
        if i in frontiers[f]:
            return f
    return None


def calculate_avg_ranks(ranks_rij, plot_data):
    return sum(ranks_rij) / len(ranks_rij)


def calculate_q_statistics(avg_ranks, plot_data):
    D = len(plot_data)
    K = 1
    s = sum([(r - (K + 1)/2) ** 2 for r in [avg_ranks]])
    return (12 * D * s) / (K * (K + 1))


def calculate_correction(frontiers):
    s = sum([len(f) ** 3 - len(f) for f in frontiers])
    N = 2
    K = 2
    return 1 - s / (N * (K ** 3 - K))


def main(arch_col):
    plot_data_pcdarts = list(arch_col.select([CLossV.ORIGINAL]).values())
    plot_data_mopcdarts = list(arch_col.select([CLossV.D_LOSS_V5]).values())

    print(calculate_friedman_nds(plot_data_pcdarts))
    print(calculate_friedman_nds(plot_data_mopcdarts))

    all_data = []
    all_data.extend(plot_data_mopcdarts)
    all_data.extend(plot_data_pcdarts)
    print(calculate_friedman_nds(all_data))


if __name__ == '__main__':
    arch_collection = ArchDataCollection()
    arch_collection.load()
    main(arch_collection)
