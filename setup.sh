#!/bin/sh
module purge
module load cuda/9.0 cudnn/8.0-v7.0.5
PYTHONPATH=""
source pytorchenv/bin/activate
alias python='python3'
