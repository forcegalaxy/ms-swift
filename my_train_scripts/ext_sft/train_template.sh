#!/bin/bash

# 默认值
MODEL=""
DATASET=""
NUM_TRAIN_EPOCHS=4
MAX_LENGTH=16384
OUTPUT_DIR=""
SWANLAB_PROJECT="S1-matmind-msswift"
SWANLAB_EXP_NAME=""
MODEL_TYPE=""  # 新增参数

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --swanlab_project)
            SWANLAB_PROJECT="$2"
            shift 2
            ;;
        --swanlab_exp_name)
            SWANLAB_EXP_NAME="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$MODEL" || -z "$DATASET" || -z "$OUTPUT_DIR" || -z "$SWANLAB_EXP_NAME" ]]; then
    echo "缺少必要参数，请检查 model, dataset, output_dir, swanlab_exp_name 是否提供"
    exit 1
fi

# 构建命令（只保留实际的 CLI 参数）
CMD=(
    swift sft
    --model "$MODEL"
    --train_type full
    --dataset "$DATASET"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --torch_dtype bfloat16
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --learning_rate "$LEARNING_RATE"  # 1e-6
    --gradient_accumulation_steps 2
    --packing False
    --max_length "$MAX_LENGTH"
    --truncation_strategy delete
    --save_strategy epoch
    --logging_steps 2
    --split_dataset_ratio 0.05
    --eval_strategy steps
    --eval_steps 100
    --warmup_ratio 0.1
    --dataloader_num_workers 8
    --dataset_num_proc 32
    --save_only_model true
    --output_dir "$OUTPUT_DIR"
    --deepspeed zero2
    --attn_impl flash_attn
    --sequence_parallel_size 1
    --report_to swanlab
    --swanlab_project "$SWANLAB_PROJECT"
    --swanlab_exp_name "$SWANLAB_EXP_NAME"
    --swanlab_workspace "S1-Omni"
    --use_extended_vocab  true
    --vocab_init_method mean
    --resize_on_cpu true
    --science_tokenizer_path /nfs-13/wuxiaoyu/科学Omni/Sequence_tokenizer/qwen3_tokenizer/qwen3
)

# 如果 model_type 有值，则添加
if [[ -n "$MODEL_TYPE" ]]; then
    CMD+=(--model_type "$MODEL_TYPE")
fi

# # 使用 env 设置环境变量并执行命令
# env NPROC_PER_NODE=8 \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     SEQUENCE_PARALLEL_IMPL=ring_attention \
#     RING_HEAD_STRIDE=2 \
#     "${CMD[@]}"

# 使用 env 设置环境变量并执行命令
env NPROC_PER_NODE=8 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${CMD[@]}"