from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
M2 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])
M3 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('dil_conv_3x3', 4), ('avg_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])
M5 = Genotype(normal=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
M6 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M7 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])
M8 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_5x5', 0)], reduce_concat=[2, 3, 4, 5])
M9 = Genotype(normal=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 4), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 1), ('avg_pool_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M10 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])
M11 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
M12 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=[2, 3, 4, 5])
M13 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=[2, 3, 4, 5])
M14 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5])
M15 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])
M16 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('skip_connect', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])
M17 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
M19 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1)], reduce_concat=[2, 3, 4, 5])
M20 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1)], reduce_concat=[2, 3, 4, 5])
M21 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
M22 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M23 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M24 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1)], reduce_concat=[2, 3, 4, 5])
M25 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M26 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M27 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
M28 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
M29 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
M30 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
M31 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('dil_conv_5x5', 0), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
M32 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
M33 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M34 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
M35 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
M36 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
M37 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
M38 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])
M39 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])
M40 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 3)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])
M41 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5])
M42 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5])

PCDARTS = PC_DARTS_cifar
M1 = PCDARTS

