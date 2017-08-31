#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python adaptation.py\
    --iterations    10000\
    --batch_size    128\
    --display       10\
    --lr            0.0002\
    --snapshot      5000
