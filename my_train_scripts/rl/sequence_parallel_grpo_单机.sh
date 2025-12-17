#!/bin/bash

# 确保使用正确的 Swift 路径
export PYTHONPATH="/data02/home/zdhs0073/model_train/ms-swift-1001:$PYTHONPATH"
# 禁用 Python 字节码缓存 (调试用)
export PYTHONDONTWRITEBYTECODE=1

# 默认值
MODEL=""
DATASET=""
NUM_TRAIN_EPOCHS=1
MAX_LENGTH=2048
MAX_COMPLETION_LENGTH=1024
OUTPUT_DIR=""
SWANLAB_PROJECT="S1-msswift-grpo"
SWANLAB_EXP_NAME=""
MODEL_TYPE=""
SYSTEM_PROMPT=""
REWARD_FUNCS="accuracy format"
REWARD_WEIGHTS=""  # 新增:奖励函数权重,格式如 "0.8 0.2"
NUM_GENERATIONS=16
VLLM_TENSOR_PARALLEL_SIZE=8

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
        --max_completion_length)
            MAX_COMPLETION_LENGTH="$2"
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
        --system_prompt)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --reward_funcs)
            REWARD_FUNCS="$2"
            shift 2
            ;;
        --reward_weights)
            REWARD_WEIGHTS="$2"
            shift 2
            ;;
        --num_generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --vllm_tensor_parallel_size)
            VLLM_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --sequence_parallel_size)
            SEQUENCE_PARALLEL_SIZE="$2"
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
    echo "缺少必要参数,请检查 model, dataset, output_dir, swanlab_exp_name 是否提供"
    exit 1
fi

# 构建命令（只保留实际的 CLI 参数）
CMD=(
    swift rlhf
    --rlhf_type grpo
    --model "$MODEL"
    --train_type full
    --use_vllm true
    --vllm_mode server 
    --vllm_server_host gpunode20     # vmlogin02 或实际IP
    --vllm_server_port 8000 
    --async_generate true
    --dataset "$DATASET"
    --truncation_strategy delete
    --load_from_cache_file true
    --torch_dtype bfloat16
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_length "$MAX_LENGTH"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 16
    --split_dataset_ratio 0.01
    --save_strategy steps
    --save_steps 100
    --eval_steps 100
    --eval_strategy steps
    --learning_rate 1e-6
    --save_only_model true
    --logging_steps 2
    --output_dir "$OUTPUT_DIR"
    --warmup_ratio 0.1
    --dataloader_num_workers 8
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --reward_funcs scibench_math
    --num_generations "$NUM_GENERATIONS"
    --deepspeed zero3
    --temperature 0.6
    --top_p 0.95
    --top_k 80
    --attn_impl flash_attn
    --log_completions true
    --offload_optimizer true  # 在vLLM 推理阶段，释放模型和优化器占用的显存
    --offload_model true      # 在vLLM 推理阶段，释放模型和优化器占用的显存
    --padding_free false   # 配合 --attn_impl flash_attn 使用，但是不兼容sequence_parallel_size
    --dataloader_drop_last true
    --sleep_level 1  # 在训练阶段，释放 vLLM 占用的显存
    --log_entropy  # 记录训练中的熵值变化动态
    --report_to swanlab
    --swanlab_project "$SWANLAB_PROJECT"
    --swanlab_exp_name "$SWANLAB_EXP_NAME"
)

# --sequence_parallel_size 4
# 如果 model_type 有值,则添加
if [[ -n "$MODEL_TYPE" ]]; then
    CMD+=(--model_type "$MODEL_TYPE")
fi

# 如果 system_prompt 有值,则添加
if [[ -n "$SYSTEM_PROMPT" ]]; then
    CMD+=(--system "$SYSTEM_PROMPT")
fi

# 如果 reward_weights 有值,则添加
if [[ -n "$REWARD_WEIGHTS" ]]; then
    CMD+=(--reward_weights $REWARD_WEIGHTS)
fi

# # 使用 env 设置环境变量并执行命令
# env NPROC_PER_NODE=8 \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     PYTORCH_CUDA_ALLOC_CONF='' \
#     SEQUENCE_PARALLEL_IMPL=ring_attention \
#     RING_HEAD_STRIDE=2 \
#     "${CMD[@]}"

# 使用 env 设置环境变量并执行命令
env NPROC_PER_NODE=8 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${CMD[@]}"


