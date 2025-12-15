#!/bin/bash
# 使用扩展词表训练 Qwen 模型的示例脚本

# 切换到 Sequence_tokenizer 目录
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
MODEL_PATH="/path/to/Qwen3-8B"

# 数据集
DATASET="your-dataset"

# 输出目录（相对于 Sequence_tokenizer 目录）
OUTPUT_DIR="msswift/output/qwen_science_vocab"

# 训练配置
echo "🚀 使用扩展科学词表训练 Qwen 模型"
echo "================================================"
echo "工作目录: $(pwd)"
echo "模型: $MODEL_PATH"
echo "数据集: $DATASET"
echo "输出: $OUTPUT_DIR"
echo "================================================"

# 使用 ms-swift CLI 训练（集成完成后）
swift sft \
    --model_type qwen-7b-chat \
    --model_id_or_path $MODEL_PATH \
    --dataset $DATASET \
    --use_extended_vocab \
    --science_tokenizer_path qwen3_tokenizer/qwen3 \
    --init_method mean \
    --resize_on_cpu \
    --sft_type lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --bf16 true \
    --gradient_checkpointing true \
    --output_dir $OUTPUT_DIR \
    --save_steps 500 \
    --logging_steps 10

echo "✅ 训练完成！"
echo "模型保存在: $OUTPUT_DIR"
