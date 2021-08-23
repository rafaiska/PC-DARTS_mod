#!/bin/bash
cd /workspace/PC-DARTS
#python train_search_p3.py --batch_size 64 --epochs 40 --custom_loss
python train_search_p3.py --batch_size 128 --epochs 50 --custom_loss
