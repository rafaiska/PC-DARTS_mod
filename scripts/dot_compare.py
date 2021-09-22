#!/usr/bin/env python
import sys


def _parse_edges(dot_file):
    edges = set()
    edge_names = list()
    lines = dot_file.readlines()
    for line in lines:
        if '->' in line and 'label' in line:
            splitted = line.split()
            src_node = splitted[0]
            dst_node = splitted[2]
            edge_name = splitted[3][1:-1]
            edges.add((src_node, dst_node, edge_name))
            edge_names.append(edge_name)
    return edges, edge_names


def _edge_qt_similarity(e_names_a, e_names_b):
    qt = 0
    for name in e_names_a:
        if name in e_names_b:
            e_names_b.remove(name)
            qt += 1
    return qt


def compare_dots(dot_a, dot_b):
    edges_a, e_names_a = _parse_edges(dot_a)
    edges_b, e_names_b = _parse_edges(dot_b)
    return len(edges_a.intersection(edges_b)), _edge_qt_similarity(e_names_a, e_names_b)


def main():
    dot_file_a = open(sys.argv[1], 'r')
    dot_file_b = open(sys.argv[2], 'r')
    similar_edges = compare_dots(dot_file_a, dot_file_b)
    print("Similarity in edges: {}".format(similar_edges))
    dot_file_a.close()
    dot_file_b.close()


if __name__ == '__main__':
    main()
