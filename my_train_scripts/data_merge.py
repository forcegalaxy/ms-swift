#!/usr/bin/env python3
"""
JSONL文件合并工具
支持按条数或比例抽取数据,可选数据打乱功能
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Union
import argparse


class JSONLMerger:
    """JSONL文件合并器"""
    
    def __init__(self, 
                 input_folder: str,
                 output_file: str,
                 max_lines: Optional[int] = None,
                 sample_ratio: Optional[float] = None,
                 shuffle: bool = False,
                 seed: Optional[int] = None,
                 exclude_files: Optional[List[str]] = None):
        """
        初始化合并器
        
        Args:
            input_folder: 输入文件夹路径
            output_file: 输出文件路径
            max_lines: 每个文件最大抽取条数(与sample_ratio二选一)
            sample_ratio: 每个文件抽取比例,0-1之间(与max_lines二选一)
            shuffle: 是否打乱数据顺序
            seed: 随机种子,用于可复现的打乱
            exclude_files: 要排除的文件名列表(可以是完整文件名或不带扩展名)
        """
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.max_lines = max_lines
        self.sample_ratio = sample_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.exclude_files = exclude_files or []
        
        # 参数验证
        if max_lines is not None and sample_ratio is not None:
            raise ValueError("max_lines和sample_ratio不能同时指定")
        
        if sample_ratio is not None and not (0 < sample_ratio <= 1):
            raise ValueError("sample_ratio必须在0到1之间")
        
        if not self.input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
    
    def get_jsonl_files(self) -> List[Path]:
        """获取所有JSONL文件(排除指定文件)"""
        all_jsonl_files = list(self.input_folder.glob("*.jsonl"))
        
        if not all_jsonl_files:
            raise ValueError(f"文件夹中没有找到.jsonl文件: {self.input_folder}")
        
        # 构建排除文件名集合(支持带扩展名和不带扩展名两种形式)
        exclude_set = set()
        for name in self.exclude_files:
            exclude_set.add(name)
            # 如果提供的名称不带.jsonl扩展名,也添加带扩展名的版本
            if not name.endswith('.jsonl'):
                exclude_set.add(f"{name}.jsonl")
            # 如果提供的名称带.jsonl扩展名,也添加不带扩展名的版本
            else:
                exclude_set.add(name[:-6])  # 去掉.jsonl
        
        # 过滤文件
        filtered_files = []
        excluded_files = []
        
        for file_path in all_jsonl_files:
            file_name = file_path.name
            file_stem = file_path.stem
            
            if file_name in exclude_set or file_stem in exclude_set:
                excluded_files.append(file_name)
            else:
                filtered_files.append(file_path)
        
        # 显示排除信息
        if excluded_files:
            print(f"已排除 {len(excluded_files)} 个文件: {', '.join(excluded_files)}")
        
        if not filtered_files:
            raise ValueError(f"排除指定文件后,没有剩余的.jsonl文件")
        
        return sorted(filtered_files)
    
    def read_and_sample(self, file_path: Path) -> List[dict]:
        """
        读取并采样单个JSONL文件
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            采样后的数据列表
        """
        data = []
        
        # 读取所有数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: {file_path.name} 第{line_num}行JSON解析失败: {e}")
                    continue
        
        original_count = len(data)
        
        # 根据参数进行采样
        if self.max_lines is not None:
            # 按条数限制
            sample_count = min(self.max_lines, original_count)
        elif self.sample_ratio is not None:
            # 按比例限制
            sample_count = max(1, int(original_count * self.sample_ratio))
        else:
            # 不限制,全部采样
            sample_count = original_count
        
        # 采样
        if sample_count < original_count:
            data = random.sample(data, sample_count)
        
        print(f"  {file_path.name}: {original_count}条 -> 采样{sample_count}条")
        
        return data
    
    def merge(self) -> int:
        """
        执行合并操作
        
        Returns:
            合并后的总条数
        """
        print(f"开始合并JSONL文件...")
        print(f"输入文件夹: {self.input_folder}")
        print(f"输出文件: {self.output_file}")
        
        if self.exclude_files:
            print(f"排除文件: {', '.join(self.exclude_files)}")
        
        if self.max_lines:
            print(f"每个文件最大条数: {self.max_lines}")
        elif self.sample_ratio:
            print(f"每个文件采样比例: {self.sample_ratio * 100}%")
        else:
            print(f"采样模式: 全部数据")
        
        print(f"打乱顺序: {'是' if self.shuffle else '否'}")
        if self.seed is not None:
            print(f"随机种子: {self.seed}")
        print()
        
        # 获取所有JSONL文件
        jsonl_files = self.get_jsonl_files()
        print(f"找到{len(jsonl_files)}个JSONL文件\n")
        
        # 读取并采样所有文件
        all_data = []
        for file_path in jsonl_files:
            file_data = self.read_and_sample(file_path)
            all_data.extend(file_data)
        
        # 打乱数据
        if self.shuffle:
            print(f"\n打乱数据顺序...")
            random.shuffle(all_data)
        
        # 写入输出文件
        print(f"\n写入输出文件...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        total_count = len(all_data)
        print(f"\n合并完成!")
        print(f"总条数: {total_count}")
        print(f"输出文件: {self.output_file}")
        
        return total_count


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='合并JSONL文件工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 合并所有数据
  python data_merge.py -i ./data -o merged.jsonl
  
  # 每个文件最多抽取100条
  python data_merge.py -i ./data -o merged.jsonl -n 100
  
  # 每个文件抽取50%的数据
  python data_merge.py -i ./data -o merged.jsonl -r 0.5
  
  # 合并并打乱数据
  python data_merge.py -i ./data -o merged.jsonl --shuffle
  
  # 排除指定文件
  python data_merge.py -i ./data -o merged.jsonl -e test.jsonl bad_data
  
  # 使用随机种子确保可复现
  python data_merge.py -i ./data -o merged.jsonl --shuffle --seed 42
  
   # 使用随机种子确保可复现
  python data_merge.py -i /nfs-13/wuxiaoyu/model_train/S1_data -o merged.jsonl -n 50000 --shuffle --seed 42  -e science_lixiangang_r1_correct_84010_20250528.jsonl
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='输入文件夹路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件路径')
    parser.add_argument('-n', '--max-lines', type=int,
                        help='每个文件最大抽取条数')
    parser.add_argument('-r', '--ratio', type=float,
                        help='每个文件抽取比例(0-1之间)')
    parser.add_argument('-e', '--exclude', nargs='+',
                        help='要排除的文件名(可多个,支持带或不带.jsonl扩展名)')
    parser.add_argument('--shuffle', action='store_true',
                        help='打乱数据顺序')
    parser.add_argument('--seed', type=int,
                        help='随机种子(用于可复现的打乱和采样)')
    
    args = parser.parse_args()
    
    try:
        merger = JSONLMerger(
            input_folder=args.input,
            output_file=args.output,
            max_lines=args.max_lines,
            sample_ratio=args.ratio,
            shuffle=args.shuffle,
            seed=args.seed,
            exclude_files=args.exclude
        )
        merger.merge()
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
    
    
'''
 python data_merge.py -i /nfs-13/wuxiaoyu/model_train/S1_data/processed_benchmarks -o Annealing.jsonl -n 50000 --shuffle --seed 42 
 python data_merge.py -i /nfs-13/wuxiaoyu/model_train/S1_data/processed_benchmarks_多条_DeepSeek-R1-0528 -o Annealing_多条.jsonl -n 50000 --shuffle --seed 42

# 251127
(swift_py311) wuxiaoyu@wg-4-12:~/model_train/ms-swift/my_train_scripts$   python data_merge.py -i /nfs-13/wuxiaoyu/model_train/S1_data -o merged.jsonl -n 50000 --shuffle --seed 42  -e science_lixiangang_r1_correct_84010_20250528.jsonl
开始合并JSONL文件...
输入文件夹: /nfs-13/wuxiaoyu/model_train/S1_data
输出文件: merged.jsonl
排除文件: science_lixiangang_r1_correct_84010_20250528.jsonl
每个文件最大条数: 50000
打乱顺序: 是
随机种子: 42

已排除 1 个文件: science_lixiangang_r1_correct_84010_20250528.jsonl
找到5个JSONL文件

  sft_chem_latest_20250708_17408_R1_good.jsonl: 17408条 -> 采样17408条
  sft_chem_latest_20250708_610_R1_failed.jsonl: 610条 -> 采样610条
  18018
  sft_mat_latest_20250708_474.jsonl: 474条 -> 采样474条
  sft_mat_scentific_20250612_1243条.jsonl: 1243条 -> 采样1243条
  sft_matsciinstruct_latest_20250708_35371.jsonl: 35371条 -> 采样35371条
  37088

打乱数据顺序...

写入输出文件...

合并完成!
总条数: 55106
输出文件: merged.jsonl
'''