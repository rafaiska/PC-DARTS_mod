#!/bin/bash
rm -f nohup.out
nohup docker run --gpus all --rm -v /home/msc2020-fra/ra094324/workspace/:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel "./PC-DARTS/run_experiment.sh" 2> /dev/null &
