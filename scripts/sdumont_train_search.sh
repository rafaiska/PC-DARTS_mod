#!/bin/bash

#SBATCH -J PC-DARTS-TEST
#SBTACH --gpus-per-task 1
#SBATCH --ntasks 1

docker run --gpus all -v /scratch/unicamp-automl/rafael.sanchez/temp_data/:/workspace/PC-DARTS/data --rm pc_darts_original:latest /bin/bash -c "python ./train_search.py; mv search-EXP-* ./data/"