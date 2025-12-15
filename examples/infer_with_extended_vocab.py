#!/usr/bin/env python3
"""
使用扩展科学词表进行推理的示例

运行前请确保：
1. 已完成词表扩展功能集成
2. 在 Sequence_tokenizer 目录下执行
"""

import os
import sys

# 确保在 Sequence_tokenizer 目录下
sequence_tokenizer_dir = '/Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer'
os.chdir(sequence_tokenizer_dir)
sys.path.insert(0, os.path.join(sequence_tokenizer_dir, 'msswift'))


def example_1_test_tokenizer():
    """示例1：测试科学词表加载"""
    print("=" * 80)
    print("示例1：测试科学词表加载和分词")
    print("=" * 80)

    from swift.llm.model.science_tokenizer import load_science_tokenizer

    # 加载科学词表
    tokenizer = load_science_tokenizer()
    print(f"✅ 词表大小: {len(tokenizer)}")

    # 测试科学符号分词
    test_cases = [
        "分子式 <SMILES>CCO</SMILES> 是乙醇",
        "蛋白质序列 <FASTA>MKTAYIAKQRQISFVKSHFSRQLE</FASTA>",
        "IUPAC 名称 <IUPAC>ethanol</IUPAC>",
    ]

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\n文本: {text}")
        print(f"Token 数: {len(tokens)}")
        print(f"Tokens: {' | '.join(tokens[:10])}...")  # 只显示前10个


def example_2_basic_inference():
    """示例2：基础推理（需要完成集成）"""
    print("\n" + "=" * 80)
    print("示例2：使用扩展词表进行基础推理")
    print("=" * 80)

    # 注意：需要先完成集成才能运行
    print("TODO: 完成集成后，可以使用以下代码进行推理")
    print("""
from swift.llm import get_model_tokenizer, infer_main

# 加载模型（扩展词表）
model, tokenizer = get_model_tokenizer(
    'qwen/Qwen-7B-Chat',
    use_extended_vocab=True,
    torch_dtype='bfloat16',
    device_map='auto'
)

# 推理
query = "分析这个分子式：<SMILES>CC(=O)Oc1ccccc1C(=O)O</SMILES>"
# response = infer(model, tokenizer, query)
# print(f"回答: {response}")
    """)


def example_3_cli_inference():
    """示例3：使用 CLI 进行推理"""
    print("\n" + "=" * 80)
    print("示例3：使用 CLI 进行推理")
    print("=" * 80)

    print("在 Sequence_tokenizer 目录下执行:")
    print("""
# 推理命令
swift infer \\
    --ckpt_dir msswift/output/qwen_science_vocab/xxx \\
    --use_extended_vocab \\
    --science_tokenizer_path qwen3_tokenizer/qwen3

# 或使用默认词表路径
swift infer \\
    --ckpt_dir msswift/output/qwen_science_vocab/xxx \\
    --use_extended_vocab
    """)


def example_4_compare_tokenization():
    """示例4：对比原始词表和扩展词表"""
    print("\n" + "=" * 80)
    print("示例4：对比原始词表和扩展词表的分词效果")
    print("=" * 80)

    test_text = "分子式 <SMILES>CC(=O)Oc1ccccc1C(=O)O</SMILES> 表示阿司匹林"

    print(f"测试文本: {test_text}\n")

    try:
        # 加载扩展词表
        from swift.llm.model.science_tokenizer import load_science_tokenizer
        tokenizer_extended = load_science_tokenizer()

        # 分词
        tokens_extended = tokenizer_extended.tokenize(test_text)

        print("扩展词表分词:")
        print(f"  Token 数量: {len(tokens_extended)}")
        print(f"  Tokens: {tokens_extended}")

        # 对比原始词表（需要加载原始 tokenizer）
        print("\n原始词表分词:")
        print("  (需要加载原始 Qwen tokenizer 进行对比)")

    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保已完成词表扩展模块的集成")


def example_5_batch_inference():
    """示例5：批量推理科学问题"""
    print("\n" + "=" * 80)
    print("示例5：批量推理科学问题")
    print("=" * 80)

    queries = [
        "这个 SMILES 表示什么分子：<SMILES>CCO</SMILES>",
        "分析蛋白质序列：<FASTA>MKTAYIAKQRQISFVKSHFSRQLE</FASTA>",
        "<IUPAC>ethanol</IUPAC> 的中文名称是什么？",
        "解释 DNA 的双螺旋结构",
    ]

    print("测试问题：")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")

    print("\n使用 Python API 批量推理:")
    print("""
from swift.llm import get_model_tokenizer

model, tokenizer = get_model_tokenizer(
    'qwen/Qwen-7B-Chat',
    use_extended_vocab=True,
)

for query in queries:
    response = infer(model, tokenizer, query)
    print(f"Q: {query}")
    print(f"A: {response}\\n")
    """)


if __name__ == '__main__':
    print("🔬 扩展科学词表推理示例")
    print("工作目录:", os.getcwd())
    print("=" * 80)

    # 运行示例
    try:
        example_1_test_tokenizer()
    except Exception as e:
        print(f"❌ 示例1失败: {e}")

    example_2_basic_inference()
    example_3_cli_inference()

    try:
        example_4_compare_tokenization()
    except Exception as e:
        print(f"❌ 示例4失败: {e}")

    example_5_batch_inference()

    print("\n" + "=" * 80)
    print("💡 提示：")
    print("1. 确保在 Sequence_tokenizer 目录下执行")
    print("2. 需要先完成词表扩展功能的集成")
    print("3. 修改示例中的模型和数据路径")
    print("=" * 80)
