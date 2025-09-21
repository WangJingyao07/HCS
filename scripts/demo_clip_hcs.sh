#!/usr/bin/env bash
python eval/eval_clip_zeroshot.py \
  --config configs/clip_zeroshot.yaml \
  --use_hcs true \
  --hcs_grid "[(4,4),(8,8)]" \
  --hcs_topk "[6,8]"
