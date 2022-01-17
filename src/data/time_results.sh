#!/bin/bash

# workers=(1, 2, 4, 8, 16)

# for n in "${workers[@]}"; do
#     python3 src/data/lfw_dataset.py -get_timing -num_workers $n
# done

num_workers=(0 1 2 4 8)
for n in "${num_workers[@]}"; do
    python3 src/data/lfw_dataset.py -get_timing -num_workers $n
done

