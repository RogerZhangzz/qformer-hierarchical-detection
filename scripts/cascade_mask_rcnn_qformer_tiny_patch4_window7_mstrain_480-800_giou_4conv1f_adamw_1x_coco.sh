#!/bin/bash

RN_CMD="\
        tools/dist_train.sh configs/swin/cascade_mask_rcnn_qformer_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 \
        --cfg-options \
              model.pretrained=<path> \
              model.backbone.load_ema=True \
        --work-dir ./work_dirs/cascade_mask_rcnn_qformer_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco \
       "
       # --prof 100

bash -c "${RN_CMD}"
