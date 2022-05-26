from scripts.arch_data import ArchDataCollection, CLossV

TRAIN_SEARCH_COMMAND = 'python train_search.py --custom_loss --load_warmup --c_loss_w {}'


def main():
    arch_collection = ArchDataCollection()
    arch_collection.load()
    closs_w_instance_counters = {}
    for a in sorted(
            arch_collection.select((CLossV.D_LOSS_V4, CLossV.D_LOSS_V5), best_train_sys=None, has_acc=False).values(),
            key=lambda x: x.closs_w, reverse=True):
        if a.closs_w not in closs_w_instance_counters:
            closs_w_instance_counters[a.closs_w] = {}
        if a.closs_v not in closs_w_instance_counters[a.closs_w]:
            closs_w_instance_counters[a.closs_w][a.closs_v] = 0
        closs_w_instance_counters[a.closs_w][a.closs_v] += 1
    # print('w\t\t\t', 'V4\t\t\t', 'V5\t\t\t', 'V4 - V5')
    for closs_w in sorted(closs_w_instance_counters, reverse=True):
        v4_c = closs_w_instance_counters[closs_w][CLossV.D_LOSS_V4] if CLossV.D_LOSS_V4 in closs_w_instance_counters[
            closs_w] else 0
        v5_c = closs_w_instance_counters[closs_w][CLossV.D_LOSS_V5] if CLossV.D_LOSS_V5 in closs_w_instance_counters[
            closs_w] else 0
        # print('{}\t\t\t'.format(closs_w), '{}\t\t\t'.format(v4_c), '{}\t\t\t'.format(v5_c), v4_c - v5_c)
        if v4_c - v5_c > 0:
            # print(v4_c, '-', v5_c, '=', v4_c - v5_c)
            for _ in range(v4_c - v5_c):
                print(TRAIN_SEARCH_COMMAND.format(closs_w))
    print(len(arch_collection.select((CLossV.D_LOSS_V4,), best_train_sys=None, has_acc=False)))
    print(len(arch_collection.select((CLossV.D_LOSS_V5,), best_train_sys=None, has_acc=False)))
    print('\n\n')
    for a in sorted(arch_collection.select((CLossV.D_LOSS_V5,), best_train_sys=None, has_acc=False).values(),
                    key=lambda x: x.closs_w, reverse=True):
        if not a.model_acc:
            print(a.arch_id)


if __name__ == '__main__':
    main()
