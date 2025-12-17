#!/bin/bash
# nohup 
# nohup bash run_swift_matmind_v2_chat.sh  > /nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/nohup_log/train_MatMind_chat_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# pkill -9 -f "/nfs-13/wuxiaoyu/model_train/model/Qwen3-8B"
export NCCL_TIMEOUT=1800000
export TORCH_USE_CUDA_DSA=1
export DS_SKIP_CUDA_CHECK=1

# mkdir -p train_log

# 公共脚本目录
SCRIPT_DIR="/nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/sft"
TEMPLATE_SCRIPT="${SCRIPT_DIR}/train_template.sh"

# 定义多个训练任务的参数（每个任务是一个参数字符串 + 任务名）
# 格式：任务名|参数字符串
declare -a TRAIN_TASKS=(        
    "MatMind_chat_Qwen3_8B_53702_1e_6_16k_Annealing_5338_DS|--model /nfs-13/wuxiaoyu/model_train/MatMind/MatMind_chat_Qwen3_8B_53702_5e_6_16k/v1-20251204-110151/checkpoint-19578-MatMind_chat_Qwen3_8B_53702_5e_6_16k_ep6 \
     --dataset /nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/Annealing_5338_DS.jsonl \
     --num_train_epochs 7 \
     --learning_rate 5e-6 \
     --max_length 16384 \
     --output_dir /nfs-13/wuxiaoyu/model_train/MatMind/MatMind_chat_Qwen3_8B_53702_1e_6_16k_Annealing_5338_DS \
     --swanlab_project S1-matmind-msswift \
     --swanlab_exp_name MatMind_chat_Qwen3_8B_53702_1e_6_16k_Annealing_5338_DS"

    # "Qwen2.5-7B-Instruct_PHYSICS参数直接生成_SciBench参数改难_19144_训练集_Qwen3_235B|--model /data02/home/zdhs0073/model_train/model_download/Qwen2.5-7B-Instruct \
    #  --dataset /data02/home/zdhs0073/科学计算/公式数据生成pipeline/4_公式代码生成题目pipline_参数改难/step7_转为训练格式/训练格式数据/0928_10遍_PHYSICS_参数直接生成_SciBench_参数改难_Qwen3_235B_训练_测试/train_19144.jsonl \
    #  --num_train_epochs 7 \
    #  --max_length 32768 \
    #  --output_dir /data02/home/zdhs0073/model_train/ms-swift/saves_PHYSICS/训练集/PHYSICS参数直接生成_SciBench参数改难_训练集_qwen3_235B/Qwen2.5-7B-Instruct_qwen3_235B蒸馏_19144_32k_1e_6 \
    #  --swanlab_project S1-msswift \
    #  --swanlab_exp_name PHYSICS参数直接生成_SciBench参数改难_训练集__19144_Qwen2.5-7B-Instruct_qwen3_235B蒸馏_19144_32k_1e_6"
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
    LOG_FILE="/nfs-13/wuxiaoyu/model_train/train_log/train_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"

    nohup bash "$TEMPLATE_SCRIPT" $TASK_ARGS > "$LOG_FILE" 2>&1 &

    train_pid=$!

    sleep 5
    if ! kill -0 $train_pid 2>/dev/null; then
        echo "错误: 训练启动失败 - $TEMPLATE_SCRIPT"
        exit 1
    fi

    echo "训练已启动: $TEMPLATE_SCRIPT (PID: $train_pid, 日志: $LOG_FILE)"

    if ! wait_for_completion $train_pid "$TEMPLATE_SCRIPT"; then
        echo "训练失败，停止后续训练"
        exit 1
    fi

    echo "间隔等待 3 分钟..."
    sleep 180
done

echo "所有训练任务完成!"

