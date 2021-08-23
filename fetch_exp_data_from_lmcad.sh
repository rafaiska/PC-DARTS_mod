#!/bin/bash
#scp /home/rafael/Projetos/PC-DARTS/*.py ra094324@lmcad-dl-2.ic.unicamp.br:~/workspace/PC-DARTS/
#scp /home/rafael/Projetos/PC-DARTS/*.sh ra094324@lmcad-dl-2.ic.unicamp.br:~/workspace/PC-DARTS/
rsync -r -v ra094324@lmcad-dl-2.ic.unicamp.br:~/workspace/PC-DARTS/search-EXP-* /home/rafael/Projetos/PC-DARTS/data/ --rsync-path=/home/msc2020-fra/ra094324/.local/bin/rsync