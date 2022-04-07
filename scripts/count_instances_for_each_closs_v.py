from scripts.arch_data import ArchDataCollection, CLossV


def count_instances(collection):
    counter = 0
    for version in CLossV:
        archs = [x for x in filter(lambda y: y.closs_v == version, collection.archs.values())]
        counter += len(archs)
        print('{}: {}'.format(version, len(archs)))
    print('Total: {}'.format(counter))


def main():
    collection = ArchDataCollection()
    collection.load()
    count_instances(collection)


if __name__ == '__main__':
    main()
