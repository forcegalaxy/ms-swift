#!/bin/bash

# 确保使用正确的 Swift 路径
export PYTHONPATH="/data02/home/zdhs0073/model_train/ms-swift-1001:$PYTHONPATH"
# 禁用 Python 字节码缓存 (调试用)
export PYTHONDONTWRITEBYTECODE=1

export NCCL_TIMEOUT=1800000
export TORCH_USE_CUDA_DSA=1
export DS_SKIP_CUDA_CHECK=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True




# 公共脚本目录
SCRIPT_DIR="/data02/home/zdhs0073/model_train/ms-swift-1001/my_train_scripts_1001"
TEMPLATE_SCRIPT="${SCRIPT_DIR}/sequence_parallel_grpo_单机.sh"

# 定义多个训练任务的参数（每个任务是一个参数字符串 + 任务名）
# 格式：任务名|参数字符串
declare -a TRAIN_TASKS=(
    "速度测试_Qwen3_8B_PHYSICS参数直接生成_SciBench参数改难_19144_Qwen3_235B_ep5_grpo_PHYSICS参数改难_21900_ScibenchMathORM|--model /data02/home/zdhs0073/model_train/ms-swift/saves_PHYSICS/训练集/PHYSICS参数直接生成_SciBench参数改难_训练集_qwen3_235B/Qwen3-8B_qwen3_235B蒸馏_19144_40k_1e_6/v0-20250928-011049/checkpoint-46905 \
     --dataset  /data02/home/zdhs0073/科学计算/公式数据生成pipeline/4_公式代码生成题目pipline_参数改难/step7_转为训练格式/训练格式数据/0921_10遍_PHYSICS_SciBench_参数改难_Qwen3_235B_训练_测试/train_21900_grpo.jsonl \
     --model_type qwen3
     --num_train_epochs 2 \
     --max_length 16384 \
     --max_completion_length 14336 \
     --output_dir /data02/home/zdhs0073/model_train/ms-swift-1001/saves_model/grpo/速度测试_Qwen3_8B_PHYSICS参数直接生成_SciBench参数改难_19144_Qwen3_235B_ep5_grpo_PHYSICS参数改难_21900 \
     --swanlab_project S1-msswift-grpo \
     --swanlab_exp_name 速度测试_Qwen3_8B_PHYSICS参数直接生成_SciBench参数改难_19144_Qwen3_235B_ep5_grpo_PHYSICS参数改难_21900 \
     --num_generations 16 \
     --vllm_server_host gpunode20 \
     --vllm_server_port 8000
    "
    #  --max_length 24576 \
    #  --max_completion_length 22528 \
    # --max_length 16384 \
    #  --max_completion_length 14336 \
    # "Qwen3-8B_CustomDataset|--model /data02/home/zdhs0073/model_train/model_download/Qwen3-8B \
    #  --dataset /path/to/your/dataset.jsonl \
    #  --num_train_epochs 2 \
    #  --max_length 4096 \
    #  --max_completion_length 2048 \
    #  --output_dir /data02/home/zdhs0073/model_train/ms-swift-1001/saves_grpo/Qwen3-8B_custom_grpo \
    #  --swanlab_project S1-msswift-grpo \
    #  --swanlab_exp_name GRPO_Qwen3-8B_custom \
    #  --reward_funcs accuracy \
    #  --num_generations 16"
)

wait_for_completion() {
    local pid=$1
    local script_name=$2

    echo "等待训练完成: $script_name (PID: $pid)"

    while kill -0 $pid 2>/dev/null; do
        echo "训练进行中... $(date)"
        sleep 300
    done

    wait $pid
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "训练成功完成: $script_name"
    else
        echo "训练失败: $script_name (退出码: $exit_code)"
        return 1
    fi
}

# 遍历每个任务
for task_entry in "${TRAIN_TASKS[@]}"; do
    # 提取任务名和参数
    TASK_NAME="${task_entry%%|*}"
    TASK_ARGS="${task_entry#*|}"

    echo "开始训练任务: $TASK_NAME"

    # 生成带任务名的日志文件名
    LOG_FILE="/data02/home/zdhs0073/model_train/ms-swift-1001/train_RL_log/train_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"

    nohup bash "$TEMPLATE_SCRIPT" $TASK_ARGS > "$LOG_FILE" 2>&1 &

    train_pid=$!

    sleep 5
    if ! kill -0 $train_pid 2>/dev/null; then
        echo "错误: 训练启动失败 - $TEMPLATE_SCRIPT"
        exit 1
    fi

    echo "训练已启动: $TEMPLATE_SCRIPT (PID: $train_pid, 日志: $LOG_FILE)"

    if ! wait_for_completion $train_pid "$TEMPLATE_SCRIPT"; then
        echo "训练失败,停止后续训练"
        exit 1
    fi

    echo "间隔等待 3 分钟..."
    sleep 180
done

echo "所有训练任务完成!"