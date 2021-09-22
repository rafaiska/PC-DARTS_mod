#!/usr/bin/env python
import sys

import graphviz

import genotypes

OUTPUT_DIR = 'data/graphs'


def _create_pcdarts_nodes():
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'
    dot.node('0', 'c_{k-2}', shape='box', group='entry', fillcolor='green', style='filled')
    dot.node('1', 'c_{k-1}', shape='box', group='entry', fillcolor='green', style='filled')
    dot.node('2', '0', shape='box', group='inter', fillcolor='mediumorchid1', style='filled')
    dot.node('3', '1', shape='box', group='inter', fillcolor='mediumorchid1', style='filled')
    dot.node('4', '2', shape='box', group='inter', fillcolor='mediumorchid1', style='filled')
    dot.node('5', '3', shape='box', group='inter', fillcolor='mediumorchid1', style='filled')
    dot.node('6', 'c_{k}', shape='box', group='out', fillcolor='orange', style='filled')
    return dot


def _create_edges(dot, genotype, is_reduce):
    sub_geno = genotype[2] if is_reduce else genotype[0]
    for dst_node in range(2, 6):
        for i in range(2):
            op, src_node = sub_geno[(dst_node - 2) * 2 + i]
            dot.edge(str(src_node), str(dst_node), label=op)
    for src_node in range(2, 6):
        dot.edge(str(src_node), '6')


def genotype_to_graph(geno_id):
    dot_normal = _create_pcdarts_nodes()
    dot_reduce = _create_pcdarts_nodes()
    genotype = eval('genotypes.{}'.format(geno_id))
    assert type(genotype) == genotypes.Genotype
    _create_edges(dot_normal, genotype, False)
    _create_edges(dot_reduce, genotype, True)
    with open('{}/{}_normal.dot'.format(OUTPUT_DIR, geno_id), 'w') as fp:
        fp.write(dot_normal.source)
    with open('{}/{}_reduce.dot'.format(OUTPUT_DIR, geno_id), 'w') as fp:
        fp.write(dot_reduce.source)
    dot_normal.render('{}/{}_normal'.format(OUTPUT_DIR, geno_id), view=True)
    dot_reduce.render('{}/{}_reduce'.format(OUTPUT_DIR, geno_id), view=True)


def main():
    genotype_to_graph(sys.argv[1])


if __name__ == '__main__':
    main()
