#!/bin/bash

python -m laion_clap.training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --datasetpath="/mnt/beegfs/home/koroshinadze/sketching_audio/Dev/CLAP/downloaded/datasets" \
    --precision="fp32" \
    --batch-size=96 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=2 \
    --use-bn-sync \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --warmup 3200 \
    --datasetnames "Clotho" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs 'logs' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --pretrained-audio '/mnt/beegfs/home/koroshinadze/sketching_audio/Dev/CLAP/downloaded/Pretrained Audio Encoder/HTSAT-fullset-imagenet-tiny-map=0.467.ckpt' \
    --prefetch-factor 2