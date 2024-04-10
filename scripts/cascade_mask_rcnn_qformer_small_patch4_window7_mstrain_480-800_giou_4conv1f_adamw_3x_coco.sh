#!/bin/bash

RN_CMD="\
        tools/dist_train.sh configs/swin/cascade_mask_rcnn_qformer_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 \
        --cfg-options \
              model.pretrained=qformer_small_patch4_window7_224/ckpt_epoch_287.pth \
              model.backbone.load_ema=True \
        --work-dir ./work_dirs/cascade_mask_rcnn_qformer_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco-coords_lambda1e-5-dpr32 \
       "
       # --prof 100

bash -c "${RN_CMD}"
