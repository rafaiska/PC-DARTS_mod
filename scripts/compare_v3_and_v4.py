from scripts.arch_data import ArchDataCollection, CLossV

V_NAME = {CLossV.D_LOSS_V3: 'Loss-v3', CLossV.D_LOSS_V4: 'Loss-v4'}


def compare_versions(arch_collection):
    archs_by_w = {}
    v3_archs = list(filter(lambda a: a.closs_v == CLossV.D_LOSS_V3, arch_collection.archs.values()))
    v4_archs = list(filter(lambda a: a.closs_v == CLossV.D_LOSS_V4, arch_collection.archs.values()))
    for a_1 in v3_archs:
        for a_2 in v4_archs:
            if a_1.closs_w == a_2.closs_w:
                if a_1.closs_w not in archs_by_w:
                    archs_by_w[a_1.closs_w] = [a_1, a_2]
                else:
                    archs_by_w[a_1.closs_w].append(a_2)
    return archs_by_w


def print_csv(archs_by_w):
    for w in archs_by_w:
        archs = archs_by_w[w]
        for a in archs:
            print(' & '.join([a.arch_id, V_NAME[a.closs_v], str(a.closs_w), str(a.macs_count), str(a.model_acc)]) +
                  ' \\\\ \\hline')


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    archs_by_w = compare_versions(arch_collection)
    print_csv(archs_by_w)


if __name__ == '__main__':
    main()
