#!/bin/bash
# cd /path/to/src/
# ./train.sh

# Configuration
DATASET_DIR="/path/to/dataset"
VAL_SET_PATH="/path/to/dataset"
VAL_ANNOTATION_PATH="/path/to/dataset"

# DiT Training Configuration
MODEL_NAME_DIT="/path/to/model"
OUTPUT_DIR_DIT="/path/to/checkpoint"

# UNet Training Configuration
MODEL_NAME_UNET="/path/to/model"
OUTPUT_DIR_UNET="/path/to/checkpoint"

# SD3 Training Configuration
MODEL_NAME_SD3="/path/to/model"
OUTPUT_DIR_SD3="/path/to/checkpoint"

# Training Hyperparameters
RESOLUTION=512
TRAIN_BATCH_SIZE=2
DATALOADER_NUM_WORKERS=12
MAX_TRAIN_STEPS=30000
GRADIENT_ACCUMULATION_STEPS=4
MIXED_PRECISION="bf16"
CHECKPOINTING_STEPS=2000
VALIDATION_STEPS=2000
SEED=42
CONDITIONING_DROPOUT_PROB=0.05
RESUME_FROM_CHECKPOINT="latest"

# Prodigy Optimizer Settings
LEARNING_RATE=1.0
ADAM_WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.99

# echo "Starting DiT training in screen session 'rsedit_dit'..."
# screen -dmS rsedit_dit bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --mixed_precision=${MIXED_PRECISION} train_dit.py \
#   --pretrained_model_name_or_path=$MODEL_NAME_DIT \
#   --train_data_dir=$DATASET_DIR \
#   --resolution=$RESOLUTION \
#   --train_batch_size=$TRAIN_BATCH_SIZE \
#   --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
#   --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
#   --use_prodigy \
#   --learning_rate=$LEARNING_RATE \
#   --prodigy_safeguard_warmup \
#   --prodigy_use_bias_correction \
#   --adam_weight_decay=$ADAM_WEIGHT_DECAY \
#   --adam_beta1=$ADAM_BETA1 \
#   --adam_beta2=$ADAM_BETA2 \
#   --use_ema \
#   --max_train_steps=$MAX_TRAIN_STEPS \
#   --output_dir=$OUTPUT_DIR_DIT \
#   --mixed_precision=$MIXED_PRECISION \
#   --checkpointing_steps=$CHECKPOINTING_STEPS \
#   --checkpoints_total_limit=1 \
#   --validation_steps=$VALIDATION_STEPS \
#   --report_to='tensorboard' \
#   --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
#   --val_set_path=$VAL_SET_PATH \
#   --val_annotation_path=$VAL_ANNOTATION_PATH \
#   --seed=$SEED \
#   --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB"


# echo "Starting UNet training in screen session 'rsedit_unet'..."
# screen -dmS rsedit_unet bash -c "CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=${MIXED_PRECISION} train_unet.py \
#   --pretrained_model_name_or_path=$MODEL_NAME_UNET \
#   --train_data_dir=$DATASET_DIR \
#   --resolution=$RESOLUTION \
#   --train_batch_size=48 \
#   --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
#   --learning_rate=1e-5 \
#   --num_train_epochs=10 \
#   --max_train_steps=$MAX_TRAIN_STEPS \
#   --gradient_accumulation_steps=1 \
#   --output_dir=$OUTPUT_DIR_UNET \
#   --mixed_precision=$MIXED_PRECISION \
#   --checkpointing_steps=$CHECKPOINTING_STEPS \
#   --checkpoints_total_limit=1 \
#   --validation_steps=1000 \
#   --val_set_path=$VAL_SET_PATH \
#   --val_annotation_path=$VAL_ANNOTATION_PATH \
#   --report_to='tensorboard' \
#   --seed=$SEED \
#   --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB"

echo "Starting SD3 training in screen session 'rsedit_sd3'..."
screen -dmS rsedit_sd3 bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 --mixed_precision=${MIXED_PRECISION} train_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME_SD3 \
  --train_data_dir=$DATASET_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --use_prodigy \
  --learning_rate=$LEARNING_RATE \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --adam_beta1=$ADAM_BETA1 \
  --adam_beta2=$ADAM_BETA2 \
  --use_ema \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --output_dir=$OUTPUT_DIR_SD3 \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=1 \
  --validation_steps=$VALIDATION_STEPS \
  --report_to='tensorboard' \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --val_set_path=$VAL_SET_PATH \
  --val_annotation_path=$VAL_ANNOTATION_PATH \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB"

echo "Training sessions have been started in the background."
echo "Use 'screen -ls' to see running sessions."
echo "Use 'screen -r rsedit_dit', 'screen -r rsedit_unet', or 'screen -r rsedit_sd3' to attach to a session."
echo "To kill a session: screen -S rsedit_dit -X quit, screen -S rsedit_unet -X quit, or screen -S rsedit_sd3 -X quit"
