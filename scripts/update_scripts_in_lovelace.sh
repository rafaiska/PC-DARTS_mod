cat /home/rafael/Dropbox/Unicamp/Dissertação/EQM/xaveco.out
mkdir -p /tmp/lovelace_transfer
cd /tmp/lovelace_transfer || exit
#cp /home/rafael/Projetos/PC-DARTS_mod/scripts/arch_data.py /tmp/lovelace_transfer/
#cp /home/rafael/Projetos/PC-DARTS_mod/scripts/validation_net_profiling.py /tmp/lovelace_transfer/
#cp /home/rafael/Projetos/PC-DARTS_mod/.arch_data /tmp/lovelace_transfer/
cp /home/rafael/Projetos/PC-DARTS_mod/genotypes.py ./
scp -P 31459 ./* rafacsan@cenapad.unicamp.br:~/
#profiling job: 17433.ada