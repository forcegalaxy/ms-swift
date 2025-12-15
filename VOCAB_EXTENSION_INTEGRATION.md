# 词表扩展功能集成指南

## 📋 概述

将 `qwen3_tokenizer/load_weight` 中的词表扩展功能集成到 `msswift` 中，实现：
1. 词表修改和扩展（QwenScienceTokenizer）
2. 使用扩展词表进行训练
3. 使用扩展词表进行推理

## 🏗️ 项目结构

```
Sequence_tokenizer/
├── qwen3_tokenizer/
│   ├── qwen3/                              # QwenScienceTokenizer
│   │   └── tokenization_qwen_science.py
│   └── load_weight/                        # 现有的词表扩展实现
│       ├── src/
│       │   ├── model_loader.py             # 模型加载和推理
│       │   ├── trainer.py                  # 训练功能
│       │   └── data_utils.py               # 数据处理
│       └── scripts/
│           ├── train.py
│           └── infer.py
└── msswift/                                # ms-swift 框架（需要集成）
    ├── swift/
    │   ├── llm/
    │   │   ├── model/
    │   │   │   ├── science_tokenizer/      # 新增：词表扩展模块
    │   │   │   ├── register.py             # 修改
    │   │   │   └── utils.py                # 修改
    │   │   ├── train/
    │   │   │   └── tuner.py                # 修改
    │   │   └── infer/
    │   │       └── infer.py                # 修改
    └── examples/                            # 新增：使用示例
```

## 🔨 实现步骤

### 步骤1：创建词表扩展模块

在 `msswift/swift/llm/model/` 下创建 `science_tokenizer` 模块：

```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer/msswift
mkdir -p swift/llm/model/science_tokenizer
```

#### 1.1 创建 `__init__.py`

```python
"""
科学词表扩展模块
提供对 SMILES、FASTA、IUPAC 等科学符号的词表扩展支持
"""

from .vocab_extension import (
    initialize_new_embeddings,
    extend_model_embeddings,
    load_science_tokenizer
)

__all__ = [
    'initialize_new_embeddings',
    'extend_model_embeddings',
    'load_science_tokenizer',
]
```

#### 1.2 创建 `vocab_extension.py`

从 `qwen3_tokenizer/load_weight/src/model_loader.py` 移植词表扩展功能。

关键函数：
- `initialize_new_embeddings()` - 初始化新增的 embedding
- `extend_model_embeddings()` - 扩展模型 embeddings
- `load_science_tokenizer()` - 加载科学词表

### 步骤2：修改模型加载逻辑

#### 2.1 修改 `swift/llm/model/register.py`

在 `get_model_tokenizer()` 函数中添加参数：

```python
def get_model_tokenizer(
    model_id_or_path: str,
    torch_dtype: Optional[Dtype] = None,
    load_model: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    # 新增参数
    use_extended_vocab: bool = False,
    science_tokenizer_path: Optional[str] = None,
    init_method: str = "mean",
    resize_on_cpu: bool = True,
    **kwargs
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """加载模型和 tokenizer"""

    # ... 原有加载逻辑 ...

    # 新增：扩展词表支持
    if use_extended_vocab:
        from .science_tokenizer import (
            load_science_tokenizer,
            extend_model_embeddings
        )

        # 记录原始词表大小
        old_vocab_size = len(tokenizer)

        # 加载科学词表
        tokenizer = load_science_tokenizer(science_tokenizer_path)
        new_vocab_size = len(tokenizer)

        logger.info(f"扩展词表: {old_vocab_size} -> {new_vocab_size}")

        # 扩展模型 embeddings
        if model is not None:
            extend_model_embeddings(
                model,
                old_vocab_size,
                new_vocab_size,
                init_method=init_method,
                resize_on_cpu=resize_on_cpu
            )

    return model, tokenizer
```

### 步骤3：添加训练参数支持

#### 3.1 修改训练参数文件

在 `swift/llm/argument/` 相关文件中添加参数：

```python
@dataclass
class SftArguments:
    # ... 原有参数 ...

    # 词表扩展相关参数
    use_extended_vocab: bool = field(
        default=False,
        metadata={'help': '是否使用扩展科学词表'}
    )

    science_tokenizer_path: Optional[str] = field(
        default=None,
        metadata={'help': '科学词表路径，默认为 qwen3_tokenizer/qwen3'}
    )

    init_method: str = field(
        default='mean',
        metadata={'help': '新 token embedding 初始化方法: mean, nearest, random'}
    )

    resize_on_cpu: bool = field(
        default=True,
        metadata={'help': '是否在 CPU 上进行 embedding 扩展（避免 OOM）'}
    )
```

#### 3.2 修改训练逻辑

在 `swift/llm/train/tuner.py` 中调用：

```python
model, tokenizer = get_model_tokenizer(
    args.model_id_or_path,
    torch_dtype=args.torch_dtype,
    use_extended_vocab=args.use_extended_vocab,
    science_tokenizer_path=args.science_tokenizer_path,
    init_method=args.init_method,
    resize_on_cpu=args.resize_on_cpu,
    **kwargs
)
```

### 步骤4：添加推理支持

#### 4.1 修改 `swift/llm/infer/infer.py`

在推理时也添加词表扩展支持：

```python
def infer_main(args):
    model, tokenizer = get_model_tokenizer(
        args.model_id_or_path,
        use_extended_vocab=args.use_extended_vocab,
        science_tokenizer_path=args.science_tokenizer_path,
        **kwargs
    )
    # ... 继续推理 ...
```

### 步骤5：添加 CLI 支持

在 CLI 脚本中添加参数：

```python
# swift/cli/sft.py 和 swift/cli/infer.py
parser.add_argument('--use_extended_vocab', action='store_true',
                   help='使用扩展科学词表')
parser.add_argument('--science_tokenizer_path', type=str, default=None,
                   help='科学词表路径')
parser.add_argument('--init_method', type=str, default='mean',
                   choices=['mean', 'nearest', 'random'],
                   help='新 token embedding 初始化方法')
parser.add_argument('--resize_on_cpu', action='store_true', default=True,
                   help='在 CPU 上扩展 embedding')
```

## 📚 路径说明

### 重要：相对路径配置

由于 `msswift` 和 `qwen3_tokenizer` 在同一父目录 `Sequence_tokenizer` 下，路径配置如下：

```python
# 在 vocab_extension.py 中
def load_science_tokenizer(tokenizer_path: Optional[str] = None):
    if tokenizer_path is None:
        # 默认路径：从 msswift 目录到 qwen3_tokenizer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # msswift/swift/llm/model/science_tokenizer -> Sequence_tokenizer
        parent_dir = os.path.abspath(os.path.join(current_dir, '../../../../../..'))
        tokenizer_path = os.path.join(parent_dir, 'qwen3_tokenizer/qwen3')

    from qwen3_tokenizer.qwen3.tokenization_qwen_science import QwenScienceTokenizer
    tokenizer = QwenScienceTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    return tokenizer
```

### 使用时的路径

```bash
# 训练时（在 Sequence_tokenizer 目录下）
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

swift sft \
    --model_type qwen-7b-chat \
    --use_extended_vocab \
    --science_tokenizer_path qwen3_tokenizer/qwen3

# 或使用默认路径（自动查找）
swift sft \
    --model_type qwen-7b-chat \
    --use_extended_vocab
```

## 🚀 使用示例

### 训练示例

```bash
# 在 Sequence_tokenizer 目录下执行
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

# 使用扩展词表训练
swift sft \
    --model_type qwen-7b-chat \
    --dataset your-dataset \
    --use_extended_vocab \
    --sft_type lora \
    --output_dir msswift/output/qwen_science

# 使用原始词表训练（默认）
swift sft \
    --model_type qwen-7b-chat \
    --dataset your-dataset \
    --sft_type lora \
    --output_dir msswift/output/qwen_normal
```

### 推理示例

```bash
# 使用扩展词表推理
swift infer \
    --ckpt_dir msswift/output/qwen_science/xxx \
    --use_extended_vocab

# Python API
python << EOF
from swift.llm import get_model_tokenizer

model, tokenizer = get_model_tokenizer(
    'qwen/Qwen-7B-Chat',
    use_extended_vocab=True,
)

# 推理
query = "分析分子式：<SMILES>CCO</SMILES>"
response = infer(model, tokenizer, query)
print(response)
EOF
```

## ✅ 完成检查清单

- [ ] 创建 `msswift/swift/llm/model/science_tokenizer/` 目录
- [ ] 创建 `__init__.py` 和 `vocab_extension.py`
- [ ] 修改 `msswift/swift/llm/model/register.py`
- [ ] 修改 `msswift/swift/llm/argument/` 相关文件
- [ ] 修改 `msswift/swift/llm/train/tuner.py`
- [ ] 修改 `msswift/swift/llm/infer/infer.py`
- [ ] 修改 `msswift/swift/cli/sft.py`
- [ ] 修改 `msswift/swift/cli/infer.py`
- [ ] 创建使用示例
- [ ] 测试训练功能
- [ ] 测试推理功能

## 🧪 测试

### 1. 测试词表加载

```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

python -c "
import sys
sys.path.insert(0, 'msswift')
from swift.llm.model.science_tokenizer import load_science_tokenizer

tokenizer = load_science_tokenizer()
print(f'✅ 词表大小: {len(tokenizer)}')

# 测试科学符号
text = '分子式：<SMILES>CCO</SMILES>'
tokens = tokenizer.tokenize(text)
print(f'Tokens: {tokens}')
"
```

### 2. 测试训练

```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

swift sft \
    --model_type qwen-7b-chat \
    --dataset alpaca-zh \
    --use_extended_vocab \
    --num_train_epochs 1 \
    --sft_type lora \
    --output_dir msswift/output/test
```

## ⚠️ 注意事项

1. **执行目录**：所有命令都在 `Sequence_tokenizer` 目录下执行
2. **相对路径**：使用相对路径引用 `qwen3_tokenizer`
3. **Python 路径**：确保 Python 能找到 `qwen3_tokenizer` 模块
4. **模型保存**：训练后的模型保存在 `msswift/output/` 下

## 📁 文件结构总览

```
Sequence_tokenizer/
├── qwen3_tokenizer/
│   ├── qwen3/
│   │   └── tokenization_qwen_science.py    # QwenScienceTokenizer
│   └── load_weight/
│       └── src/
│           ├── model_loader.py              # 源代码（参考）
│           ├── trainer.py
│           └── data_utils.py
└── msswift/                                 # 集成目标
    ├── swift/llm/model/science_tokenizer/   # 新增模块
    │   ├── __init__.py                      # ✅ 已创建
    │   └── vocab_extension.py               # ✅ 已创建
    ├── swift/llm/model/register.py          # 需修改
    ├── swift/llm/train/tuner.py             # 需修改
    ├── swift/llm/infer/infer.py             # 需修改
    └── examples/                             # 使用示例
        ├── train_with_extended_vocab.sh      # ✅ 已创建
        └── infer_with_extended_vocab.py      # ✅ 已创建
```

---

**开始集成吧！** 🚀
