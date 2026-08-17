#!/bin/bash
# Run GT then PS training sequentially on local machine.

BASE=/Users/box/src/research2026/model
LOG=$BASE/logs/run_local_$(date +%Y%m%d_%H%M%S).txt
CFG=$BASE/cfgs/train_config_local.yaml
SCRIPT=$BASE/sft_bias.py

echo "Logging to $LOG"

python3 $SCRIPT --dataset gt --config $CFG 2>&1 | tee -a $LOG && \
python3 $SCRIPT --dataset ps --config $CFG 2>&1 | tee -a $LOG
