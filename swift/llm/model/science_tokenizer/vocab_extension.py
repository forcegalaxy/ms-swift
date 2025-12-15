"""
词表扩展工具

从 qwen3_tokenizer/load_weight 移植的词表扩展功能
"""

import os
import sys
import torch
from typing import Optional
from transformers import AutoTokenizer


def initialize_new_embeddings(model, old_vocab_size: int, new_vocab_size: int, method: str = "mean"):
    """
    初始化新增的 embedding 权重

    Args:
        model: 模型实例
        old_vocab_size: 原始词表大小
        new_vocab_size: 扩展后词表大小
        method: 初始化方法 ('mean', 'nearest', 'random')
    """
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    input_embeddings_data = input_embeddings.weight.data
    output_embeddings_data = output_embeddings.weight.data

    if method == "mean":
        # 使用原始词表的平均值初始化
        input_mean = input_embeddings_data[:old_vocab_size].mean(dim=0)
        output_mean = output_embeddings_data[:old_vocab_size].mean(dim=0)
        input_embeddings_data[old_vocab_size:] = input_mean
        output_embeddings_data[old_vocab_size:] = output_mean
    elif method == "nearest":
        # 使用最后一个 token 的值初始化
        input_embeddings_data[old_vocab_size:] = input_embeddings_data[old_vocab_size - 1]
        output_embeddings_data[old_vocab_size:] = output_embeddings_data[old_vocab_size - 1]
    elif method == "random":
        # 使用随机值初始化（保持相同的标准差）
        input_std = input_embeddings_data[:old_vocab_size].std()
        output_std = output_embeddings_data[:old_vocab_size].std()
        num_new_tokens = new_vocab_size - old_vocab_size
        input_embeddings_data[old_vocab_size:] = torch.randn(
            num_new_tokens, input_embeddings_data.shape[1]
        ) * input_std
        output_embeddings_data[old_vocab_size:] = torch.randn(
            num_new_tokens, output_embeddings_data.shape[1]
        ) * output_std
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def extend_model_embeddings(
    model,
    old_vocab_size: int,
    new_vocab_size: int,
    init_method: str = "mean",
    resize_on_cpu: bool = True
):
    """
    扩展模型的 token embeddings

    Args:
        model: 模型实例
        old_vocab_size: 原始词表大小
        new_vocab_size: 新词表大小
        init_method: 初始化方法
        resize_on_cpu: 是否在 CPU 上进行扩展（避免 OOM）
    """
    current_embed_size = model.get_input_embeddings().weight.shape[0]

    if current_embed_size >= new_vocab_size:
        print(f"✅ 模型已包含足够的 embeddings ({current_embed_size} >= {new_vocab_size})")
        return

    print(f"📊 扩展 embeddings: {old_vocab_size} -> {new_vocab_size}")

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if resize_on_cpu:
        # 在 CPU 上扩展，避免 GPU OOM
        original_device = next(model.parameters()).device
        print(f"   移动模型到 CPU 进行扩展...")
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.resize_token_embeddings(new_vocab_size)

        print(f"   移动模型回 {original_device}...")
        model = model.to(original_device)
    else:
        model.resize_token_embeddings(new_vocab_size)

    # 初始化新增的 embeddings
    print(f"   初始化新 embeddings (方法: {init_method})...")
    initialize_new_embeddings(model, old_vocab_size, new_vocab_size, method=init_method)

    print(f"✅ Embeddings 扩展完成")


def load_science_tokenizer(tokenizer_path: Optional[str] = None):
    """
    加载科学词表 tokenizer

    Args:
        tokenizer_path: QwenScienceTokenizer 路径，如果为 None 则使用默认路径

    Returns:
        QwenScienceTokenizer 实例
    """
    try:
        # 确定 tokenizer 路径
        if tokenizer_path is None:
            # 默认路径：从 msswift/swift/llm/model/science_tokenizer -> Sequence_tokenizer
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上5级: science_tokenizer -> model -> llm -> swift -> msswift -> Sequence_tokenizer
            sequence_tokenizer_dir = os.path.abspath(os.path.join(current_dir, '../../../../..'))
            tokenizer_path = os.path.join(sequence_tokenizer_dir, 'qwen3_tokenizer/qwen3')

        # 添加 Sequence_tokenizer 到 Python 路径（如果需要）
        sequence_tokenizer_dir = os.path.abspath(os.path.join(tokenizer_path, '../..'))
        if sequence_tokenizer_dir not in sys.path:
            sys.path.insert(0, sequence_tokenizer_dir)

        # 导入 QwenScienceTokenizer
        from qwen3_tokenizer.qwen3.tokenization_qwen_science import QwenScienceTokenizer

        print(f"🔧 加载科学词表: {tokenizer_path}")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"科学词表路径不存在: {tokenizer_path}\n"
                f"请确认 qwen3_tokenizer 在正确的位置"
            )

        tokenizer = QwenScienceTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✅ 科学词表加载成功，大小: {len(tokenizer)}")

        return tokenizer

    except ImportError as e:
        raise ImportError(
            f"无法加载 QwenScienceTokenizer: {e}\n"
            f"请确保:\n"
            f"1. qwen3_tokenizer 在 Sequence_tokenizer 目录下\n"
            f"2. tokenization_qwen_science.py 文件存在\n"
            f"3. 路径配置正确: {tokenizer_path if tokenizer_path else '默认路径'}"
        )
    except Exception as e:
        raise RuntimeError(f"加载科学词表时出错: {e}")
