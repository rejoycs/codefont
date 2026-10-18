#!/bin/bash

language=${1:-bengali}   # default=english if not provided

python -m scripts.infer \
    --DATA_PATH="./Dataset_${language}" \
    --BATCH_SIZE=64 \
    --EMB_DIM=64 \
    --RESULTS_PATH="./results_${language}_out"\
    --STYLE_ENCODER_PATH="./ckpt/best_style_encoder_${language}.pth"\
    --CONTENT_ENCODER_PATH="./ckpt/best_content_encoder_${language}.pth"\
    --DECODER_PATH="./ckpt/best_decoder_${language}.pth"\
    --IMG_SIZE=64\
    --DEVICE="cuda:0"

