#!/bin/bash
# cd /path/to/src/
# ./ablation.sh

# Configuration
MODEL_NAME_V1_5="/path/to/model"
MODEL_NAME_V2_1="/path/to/model"
DATASET_DIR="/path/to/dataset"
VAL_SET_PATH="/path/to/dataset"
VAL_ANNOTATION_PATH="/path/to/dataset"

# Text Encoder Paths for Ablation Study
TEXT_ENCODER_A="/path/to/model"
TEXT_ENCODER_B="/path/to/model"
TEXT_ENCODER_C="/path/to/model"
TEXT_ENCODER_D="/path/to/model"

# Base Output Directory
BASE_OUTPUT_DIR="/path/to/checkpoint"

# Output Directories for Each Experiment
OUTPUT_DIR_A="${BASE_OUTPUT_DIR}/DGTRS-CLIP-ViT-L-14"
OUTPUT_DIR_B="${BASE_OUTPUT_DIR}/Git-RSCLIP-ViT-L-16"
OUTPUT_DIR_C="${BASE_OUTPUT_DIR}/Remote-CLIP-ViT-L-14"
OUTPUT_DIR_D="${BASE_OUTPUT_DIR}/OPENAI-CLIP-VIT-L-14"

# Training Hyperparameters
RESOLUTION=512
TRAIN_BATCH_SIZE=8
DATALOADER_NUM_WORKERS=12
MAX_TRAIN_STEPS=30000
GRADIENT_ACCUMULATION_STEPS=4
MIXED_PRECISION="bf16"
CHECKPOINTING_STEPS=2000
VALIDATION_STEPS=2000
SEED=42
CONDITIONING_DROPOUT_PROB=0.05

# Prodigy Optimizer Settings
LEARNING_RATE=1.0
ADAM_WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.99

# ============================================================================
# Ablation Experiment A: DGTRS-CLIP-ViT-L-14
# ============================================================================
echo "Starting Ablation Experiment A: DGTRS-CLIP-ViT-L-14 in screen session 'ablation_a'..."

# Check for existing checkpoint
RESUME_FLAG_A=""
if [ -d "$OUTPUT_DIR_A" ] && [ "$(ls -A $OUTPUT_DIR_A/checkpoint-* 2>/dev/null)" ]; then
    RESUME_FLAG_A="--resume_from_checkpoint=latest"
    echo "  Found existing checkpoint in $OUTPUT_DIR_A, will resume from latest"
fi

screen -dmS ablation_a bash -c "CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --mixed_precision=${MIXED_PRECISION} train_unet_text_ablation.py \
  --pretrained_model_name_or_path=$MODEL_NAME_V1_5 \
  --text_encoder_path=$TEXT_ENCODER_A \
  --train_data_dir=$DATASET_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --use_prodigy \
  --learning_rate=$LEARNING_RATE \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --adam_beta1=$ADAM_BETA1 \
  --adam_beta2=$ADAM_BETA2 \
  --use_ema \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --output_dir=$OUTPUT_DIR_A \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=1 \
  --validation_steps=$VALIDATION_STEPS \
  --val_set_path=$VAL_SET_PATH \
  --val_annotation_path=$VAL_ANNOTATION_PATH \
  --report_to='tensorboard' \
  --seed=$SEED \
  --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB \
  $RESUME_FLAG_A; exec bash"

# ============================================================================
# Ablation Experiment B: Git-RSCLIP-ViT-L-16
# ============================================================================
echo "Starting Ablation Experiment B: Git-RSCLIP-ViT-L-16 in screen session 'ablation_b'..."

# Check for existing checkpoint
RESUME_FLAG_B=""
if [ -d "$OUTPUT_DIR_B" ] && [ "$(ls -A $OUTPUT_DIR_B/checkpoint-* 2>/dev/null)" ]; then
    RESUME_FLAG_B="--resume_from_checkpoint=latest"
    echo "  Found existing checkpoint in $OUTPUT_DIR_B, will resume from latest"
fi

screen -dmS ablation_b bash -c "CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --mixed_precision=${MIXED_PRECISION} train_unet_text_ablation.py \
  --pretrained_model_name_or_path=$MODEL_NAME_V2_1 \
  --text_encoder_path=$TEXT_ENCODER_B \
  --train_data_dir=$DATASET_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --use_prodigy \
  --learning_rate=$LEARNING_RATE \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --adam_beta1=$ADAM_BETA1 \
  --adam_beta2=$ADAM_BETA2 \
  --use_ema \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --output_dir=$OUTPUT_DIR_B \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=1 \
  --validation_steps=$VALIDATION_STEPS \
  --val_set_path=$VAL_SET_PATH \
  --val_annotation_path=$VAL_ANNOTATION_PATH \
  --report_to='tensorboard' \
  --seed=$SEED \
  --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB \
  $RESUME_FLAG_B; exec bash"

# ============================================================================
# Ablation Experiment C: Remote-CLIP-ViT-L-14
# ============================================================================
echo "Starting Ablation Experiment C: Remote-CLIP-ViT-L-14 in screen session 'ablation_c'..."

# Check for existing checkpoint
RESUME_FLAG_C=""
if [ -d "$OUTPUT_DIR_C" ] && [ "$(ls -A $OUTPUT_DIR_C/checkpoint-* 2>/dev/null)" ]; then
    RESUME_FLAG_C="--resume_from_checkpoint=latest"
    echo "  Found existing checkpoint in $OUTPUT_DIR_C, will resume from latest"
fi

screen -dmS ablation_c bash -c "CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --mixed_precision=${MIXED_PRECISION} train_unet_text_ablation.py \
  --pretrained_model_name_or_path=$MODEL_NAME_V1_5 \
  --text_encoder_path=$TEXT_ENCODER_C \
  --train_data_dir=$DATASET_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --use_prodigy \
  --learning_rate=$LEARNING_RATE \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --adam_beta1=$ADAM_BETA1 \
  --adam_beta2=$ADAM_BETA2 \
  --use_ema \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --output_dir=$OUTPUT_DIR_C \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=1 \
  --validation_steps=$VALIDATION_STEPS \
  --val_set_path=$VAL_SET_PATH \
  --val_annotation_path=$VAL_ANNOTATION_PATH \
  --report_to='tensorboard' \
  --seed=$SEED \
  --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB \
  $RESUME_FLAG_C; exec bash"

# ============================================================================
# Ablation Experiment D: OPENAI-CLIP-VIT-L-14
# ============================================================================
echo "Starting Ablation Experiment D: OPENAI-CLIP-VIT-L-14 in screen session 'ablation_d'..."

# Check for existing checkpoint
RESUME_FLAG_D=""
if [ -d "$OUTPUT_DIR_D" ] && [ "$(ls -A $OUTPUT_DIR_D/checkpoint-* 2>/dev/null)" ]; then
    RESUME_FLAG_D="--resume_from_checkpoint=latest"
    echo "  Found existing checkpoint in $OUTPUT_DIR_D, will resume from latest"
fi

screen -dmS ablation_d bash -c "CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 --mixed_precision=${MIXED_PRECISION} train_unet_text_ablation.py \
  --pretrained_model_name_or_path=$MODEL_NAME_V1_5 \
  --text_encoder_path=$TEXT_ENCODER_D \
  --train_data_dir=$DATASET_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --use_prodigy \
  --learning_rate=$LEARNING_RATE \
  --prodigy_safeguard_warmup \
  --prodigy_use_bias_correction \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --adam_beta1=$ADAM_BETA1 \
  --adam_beta2=$ADAM_BETA2 \
  --use_ema \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --output_dir=$OUTPUT_DIR_D \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=1 \
  --validation_steps=$VALIDATION_STEPS \
  --val_set_path=$VAL_SET_PATH \
  --val_annotation_path=$VAL_ANNOTATION_PATH \
  --report_to='tensorboard' \
  --seed=$SEED \
  --conditioning_dropout_prob=$CONDITIONING_DROPOUT_PROB \
  $RESUME_FLAG_D; exec bash"

echo ""
echo "============================================================"
echo "Text Encoder Ablation Training Sessions Started"
echo "============================================================"
echo "All 4 ablation experiments have been launched in separate screen sessions."
echo ""
echo "Configuration:"
echo "  - Optimizer: Prodigy (LR=$LEARNING_RATE)"
echo "  - Text Encoder: Frozen (not trained)"
echo "  - EMA: Enabled"
echo "  - Batch Size: $TRAIN_BATCH_SIZE"
echo "  - Training Steps: $MAX_TRAIN_STEPS"
echo "  - Mixed Precision: $MIXED_PRECISION"
echo "  - Auto-Resume: Enabled (resumes from latest checkpoint if found)"
echo ""
echo "Experiments:"
echo "  A. DGTRS-CLIP-ViT-L-14   -> GPU 0 (screen: ablation_a)"
echo "  B. Git-RSCLIP-ViT-L-16   -> GPU 1 (screen: ablation_b)"
echo "  C. Remote-CLIP-ViT-L-14  -> GPU 2 (screen: ablation_c)"
echo "  D. OPENAI-CLIP-VIT-L-14 -> GPU 3 (screen: ablation_d)"
echo ""
echo "Output directories:"
echo "  A: $OUTPUT_DIR_A"
echo "  B: $OUTPUT_DIR_B"
echo "  C: $OUTPUT_DIR_C"
echo "  D: $OUTPUT_DIR_D"
echo ""
echo "Management Commands:"
echo "  - List sessions:     screen -ls"
echo "  - Attach to session: screen -r ablation_a (or ablation_b, ablation_c, ablation_d)"
echo "  - Detach from session: Ctrl+A then D"
echo "  - Kill a session:    screen -S ablation_a -X quit"
echo "  - Kill all:          screen -S ablation_a -X quit && screen -S ablation_b -X quit && screen -S ablation_c -X quit && screen -S ablation_d -X quit"
echo ""
echo "Monitor TensorBoard logs:"
echo "  tensorboard --logdir=$BASE_OUTPUT_DIR --port=6006"
echo "============================================================"

