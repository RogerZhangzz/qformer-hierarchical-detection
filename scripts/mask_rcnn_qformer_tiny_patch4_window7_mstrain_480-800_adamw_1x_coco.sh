#!/bin/bash

RN_CMD="\
        tools/dist_train.sh configs/swin/mask_rcnn_qformer_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 \
        --cfg-options \
              model.pretrained=<PRETRAINED_PATH> \
              model.backbone.load_ema=True \
        --work-dir ./work_dirs/mask_rcnn_qformer_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco \
       "
       # --prof 100

bash -c "${RN_CMD}"
