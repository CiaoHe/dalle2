# dalle2

1. Train Decoder script

```
#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train_decoder.py \
        --batch-size 256 --max-batch-size 32 \
        --clip openai_clip\
        --num-epochs 30 \
        --save-path ./ckpt/decoder/bs32_learnvar  \
        --save-interval 120 \
        --eval-interval 2000 \
        --amp True