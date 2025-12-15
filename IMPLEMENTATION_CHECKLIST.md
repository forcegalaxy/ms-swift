# 词表扩展集成实施检查清单

## 📋 总览

将词表扩展功能集成到 msswift 的完整检查清单。

**项目路径**：`/Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer`

## ✅ 第一阶段：准备工作（已完成）

- [x] 分析项目结构
- [x] 创建词表扩展模块
  - [x] `msswift/swift/llm/model/science_tokenizer/__init__.py`
  - [x] `msswift/swift/llm/model/science_tokenizer/vocab_extension.py`
- [x] 创建使用示例
  - [x] `msswift/examples/train_with_extended_vocab.sh`
  - [x] `msswift/examples/infer_with_extended_vocab.py`
- [x] 创建集成文档
  - [x] `msswift/VOCAB_EXTENSION_INTEGRATION.md`
  - [x] `INTEGRATION_QUICKSTART.md`
  - [x] `msswift/IMPLEMENTATION_CHECKLIST.md`（本文件）

## 🚧 第二阶段：核心功能集成（待完成）

### 2.1 修改模型加载逻辑

- [ ] **修改 `msswift/swift/llm/model/register.py`**

**位置**：找到 `get_model_tokenizer()` 函数（大约在第 200-400 行）

**步骤**：

1. 在函数签名中添加参数：
```python
def get_model_tokenizer(
    model_id_or_path: str,
    torch_dtype: Optional[Dtype] = None,
    load_model: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    # >>> 添加以下参数 <<<
    use_extended_vocab: bool = False,
    science_tokenizer_path: Optional[str] = None,
    init_method: str = "mean",
    resize_on_cpu: bool = True,
    # >>> 结束 <<<
    **kwargs
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
```

2. 在函数返回前添加词表扩展逻辑：
```python
    # ... 原有的模型和 tokenizer 加载逻辑 ...

    # >>> 添加以下代码（在 return 之前）<<<
    if use_extended_vocab:
        from .science_tokenizer import (
            load_science_tokenizer,
            extend_model_embeddings
        )

        old_vocab_size = len(tokenizer)
        tokenizer = load_science_tokenizer(science_tokenizer_path)
        new_vocab_size = len(tokenizer)

        logger.info(f"扩展词表: {old_vocab_size} -> {new_vocab_size}")

        if model is not None:
            extend_model_embeddings(
                model,
                old_vocab_size,
                new_vocab_size,
                init_method=init_method,
                resize_on_cpu=resize_on_cpu
            )
    # >>> 结束 <<<

    return model, tokenizer
```

3. 测试：
```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer
python -c "
import sys
sys.path.insert(0, 'msswift')
from swift.llm import get_model_tokenizer

# 仅测试 tokenizer 加载
model, tokenizer = get_model_tokenizer(
    'qwen/Qwen-7B-Chat',
    use_extended_vocab=True,
    load_model=False
)
print(f'✅ 词表大小: {len(tokenizer)}')
"
```

### 2.2 添加训练参数

- [ ] **找到训练参数定义文件**

可能的位置：
- `msswift/swift/llm/argument/train_args.py`
- `msswift/swift/llm/argument/sft_args.py`
- 或其他 `msswift/swift/llm/argument/*.py` 文件

查找方法：
```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer/msswift
grep -r "class.*Arguments" swift/llm/argument/
```

- [ ] **在训练参数类中添加字段**

找到类似 `SftArguments` 或 `TrainArguments` 的类，添加：

```python
@dataclass
class SftArguments:
    # ... 原有参数 ...

    # >>> 添加以下参数 <<<
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
        metadata={'help': 'embedding 初始化方法: mean, nearest, random'}
    )

    resize_on_cpu: bool = field(
        default=True,
        metadata={'help': '是否在 CPU 上扩展 embedding（避免 OOM）'}
    )
    # >>> 结束 <<<
```

### 2.3 修改训练逻辑

- [ ] **修改 `msswift/swift/llm/train/tuner.py`**

**步骤**：

1. 找到调用 `get_model_tokenizer()` 的位置

查找方法：
```bash
grep -n "get_model_tokenizer" msswift/swift/llm/train/tuner.py
```

2. 传递词表扩展参数：
```python
model, tokenizer = get_model_tokenizer(
    args.model_id_or_path,
    torch_dtype=args.torch_dtype,
    # ... 其他原有参数 ...
    # >>> 添加以下参数 <<<
    use_extended_vocab=args.use_extended_vocab,
    science_tokenizer_path=args.science_tokenizer_path,
    init_method=args.init_method,
    resize_on_cpu=args.resize_on_cpu,
    # >>> 结束 <<<
    **kwargs
)
```

### 2.4 修改推理逻辑

- [ ] **修改 `msswift/swift/llm/infer/infer.py`**

**步骤**：

1. 找到推理参数定义（可能在文件开头或单独的参数文件中）

2. 添加参数（如果有单独的 `InferArguments` 类）：
```python
@dataclass
class InferArguments:
    # ... 原有参数 ...

    # >>> 添加 <<<
    use_extended_vocab: bool = field(default=False)
    science_tokenizer_path: Optional[str] = field(default=None)
    # >>> 结束 <<<
```

3. 在调用 `get_model_tokenizer()` 时传递参数：
```python
model, tokenizer = get_model_tokenizer(
    args.model_id_or_path,
    # ... 原有参数 ...
    # >>> 添加 <<<
    use_extended_vocab=args.use_extended_vocab,
    science_tokenizer_path=args.science_tokenizer_path,
    # >>> 结束 <<<
)
```

## 🔧 第三阶段：CLI 支持（可选）

### 3.1 修改训练 CLI

- [ ] **修改 `msswift/swift/cli/sft.py`**

在参数解析部分添加：
```python
parser.add_argument(
    '--use_extended_vocab',
    action='store_true',
    help='使用扩展科学词表'
)

parser.add_argument(
    '--science_tokenizer_path',
    type=str,
    default=None,
    help='科学词表路径（默认：qwen3_tokenizer/qwen3）'
)

parser.add_argument(
    '--init_method',
    type=str,
    default='mean',
    choices=['mean', 'nearest', 'random'],
    help='embedding 初始化方法'
)

parser.add_argument(
    '--resize_on_cpu',
    action='store_true',
    default=True,
    help='在 CPU 上扩展 embedding'
)
```

### 3.2 修改推理 CLI

- [ ] **修改 `msswift/swift/cli/infer.py`**

添加相同的参数（至少 `--use_extended_vocab` 和 `--science_tokenizer_path`）

## 🧪 第四阶段：测试验证

### 4.1 单元测试

- [ ] **测试词表加载**
```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

python -c "
import sys
sys.path.insert(0, 'msswift')
from swift.llm.model.science_tokenizer import load_science_tokenizer

tokenizer = load_science_tokenizer()
print(f'✅ 词表大小: {len(tokenizer)}')

# 测试分词
text = '<SMILES>CCO</SMILES>'
tokens = tokenizer.tokenize(text)
print(f'Tokens: {tokens}')
"
```

- [ ] **测试 embedding 扩展**
```bash
python -c "
import sys
sys.path.insert(0, 'msswift')
from swift.llm.model.science_tokenizer import (
    load_science_tokenizer,
    extend_model_embeddings
)
from transformers import AutoModelForCausalLM

# 加载小模型测试
model = AutoModelForCausalLM.from_pretrained(
    'qwen/Qwen-1_8B',
    device_map='cpu',
    trust_remote_code=True
)

tokenizer = load_science_tokenizer()
extend_model_embeddings(model, 151669, len(tokenizer))
print(f'✅ Embedding 扩展成功: {model.get_input_embeddings().weight.shape[0]}')
"
```

### 4.2 集成测试

- [ ] **测试模型加载（不加载权重）**
```bash
python -c "
import sys
sys.path.insert(0, 'msswift')
from swift.llm import get_model_tokenizer

model, tokenizer = get_model_tokenizer(
    'qwen/Qwen-7B-Chat',
    use_extended_vocab=True,
    load_model=False
)
print(f'✅ 词表大小: {len(tokenizer)}')
"
```

- [ ] **测试训练流程（小规模测试）**
```bash
cd /Users/wuxiaoyu/zkwg/科学Omni/Science_Encoder/Sequence_tokenizer

# 修改 examples/train_with_extended_vocab.sh 中的模型路径和数据集
# 然后执行
bash msswift/examples/train_with_extended_vocab.sh
```

- [ ] **测试推理**
```bash
# 使用训练后的模型推理
python msswift/examples/infer_with_extended_vocab.py
```

### 4.3 功能验证

- [ ] **对比分词效果**

测试相同文本在原始词表和扩展词表下的分词差异：
```python
from transformers import AutoTokenizer
from swift.llm.model.science_tokenizer import load_science_tokenizer

text = "分子式 <SMILES>CC(=O)Oc1ccccc1C(=O)O</SMILES> 是阿司匹林"

# 原始词表
tokenizer_original = AutoTokenizer.from_pretrained('qwen/Qwen-7B-Chat')
tokens_original = tokenizer_original.tokenize(text)

# 扩展词表
tokenizer_extended = load_science_tokenizer()
tokens_extended = tokenizer_extended.tokenize(text)

print(f"原始词表: {len(tokens_original)} tokens")
print(f"扩展词表: {len(tokens_extended)} tokens")
print(f"减少: {len(tokens_original) - len(tokens_extended)} tokens")
```

- [ ] **验证训练效果**

训练后验证模型是否能正确处理科学符号

## 📚 第五阶段：文档和清理

- [ ] 更新 `msswift/README.md`（如果需要）
- [ ] 创建使用教程
- [ ] 添加常见问题解答
- [ ] 清理测试代码

## 🐛 常见问题排查

### 问题1：找不到 QwenScienceTokenizer

**症状**：`ImportError: cannot import name 'QwenScienceTokenizer'`

**解决**：
1. 检查文件是否存在：
```bash
ls qwen3_tokenizer/qwen3/tokenization_qwen_science.py
```

2. 检查 Python 路径：
```python
import sys
print(sys.path)
```

### 问题2：CUDA Out of Memory

**症状**：训练时显存不足

**解决**：
1. 确保使用 `resize_on_cpu=True`
2. 使用 LoRA 微调
3. 减小 batch size
4. 使用 DeepSpeed Zero-3

### 问题3：词表扩展后推理失败

**症状**：训练后推理时报错

**解决**：
推理时也要指定 `--use_extended_vocab`

## ✨ 完成标准

集成完成后，应该能够：

- ✅ 使用 CLI 进行扩展词表训练
- ✅ 使用 CLI 进行扩展词表推理
- ✅ 使用 Python API 加载扩展词表模型
- ✅ 科学符号分词效果优于原始词表
- ✅ 训练和推理流程稳定无错误

## 📊 进度追踪

**开始日期**：____________________

**第二阶段**：
- 2.1 修改模型加载逻辑：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 2.2 添加训练参数：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 2.3 修改训练逻辑：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 2.4 修改推理逻辑：[ ] 未开始 [ ] 进行中 [ ] 已完成

**第三阶段**：
- 3.1 修改训练 CLI：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 3.2 修改推理 CLI：[ ] 未开始 [ ] 进行中 [ ] 已完成

**第四阶段**：
- 4.1 单元测试：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 4.2 集成测试：[ ] 未开始 [ ] 进行中 [ ] 已完成
- 4.3 功能验证：[ ] 未开始 [ ] 进行中 [ ] 已完成

**完成日期**：____________________

---

**祝集成顺利！** 🚀

如有问题，请参考：
- [INTEGRATION_QUICKSTART.md](../INTEGRATION_QUICKSTART.md) - 快速指南
- [VOCAB_EXTENSION_INTEGRATION.md](VOCAB_EXTENSION_INTEGRATION.md) - 详细文档
