#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import matplotlib.pyplot as plt

# Global tokenizer for multiprocessing
tokenizer = None

def init_worker(model_path):
    """Initialize tokenizer for each worker process"""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def load_tokenizer(model_path):
    """Load the gpt-oss-20b tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded tokenizer from: {model_path}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def process_single_line(line_data):
    """Process a single line of data with global tokenizer - for multiprocessing"""
    global tokenizer
    line_num, line = line_data
    try:
        # Parse JSON line
        data = json.loads(line.strip())
        
        # Extract messages
        messages = data.get("messages", [])
        if not messages:
            return None, None, f"Warning: No messages found in line {line_num + 1}"
        
        # Calculate token count for this sample
        token_count = process_messages(messages, tokenizer)
        
        return token_count, data, None
        
    except json.JSONDecodeError as e:
        return None, None, f"Error parsing JSON on line {line_num + 1}: {e}"
    except Exception as e:
        return None, None, f"Error processing line {line_num + 1}: {e}"

def process_messages(messages, tokenizer):
    """Process OAI format messages and calculate token count"""
    total_tokens = 0
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        # Combine role and content as they would appear in training
        full_text = f"{role}: {content}"
        
        # Tokenize and count tokens
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        total_tokens += len(tokens)
    
    return total_tokens

def calculate_token_statistics(jsonl_file_path, model_path, max_token_length=None, output_path=None, num_processes=None):
    """Calculate token length statistics for the dataset using multiprocessing"""
    
    if num_processes is None:
        # Use fewer processes for better performance with heavy tokenizer loading
        num_processes = min(8, cpu_count())  # Cap at 8 processes
    
    print(f"Using {num_processes} processes for parallel processing")
    
    # Read all lines first
    print(f"Reading file: {jsonl_file_path}")
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines to process: {len(lines)}")
    
    # Prepare data for multiprocessing
    line_data = [(i, line) for i, line in enumerate(lines)]
    
    # Process lines in parallel with worker initialization
    token_counts = []
    filtered_data = []
    total_samples = 0
    
    print("Processing lines in parallel...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        results = list(tqdm(
            pool.imap(process_single_line, line_data, chunksize=100),  # Use chunking for better efficiency
            total=len(line_data),
            desc="Processing"
        ))
    
    # Collect results
    for token_count, data, error in results:
        if error:
            print(error)
            continue
            
        if token_count is not None and data is not None:
            token_counts.append(token_count)
            total_samples += 1
            
            # Filter data based on max_token_length if specified
            if max_token_length is None or token_count <= max_token_length:
                filtered_data.append(data)
    
    # Calculate statistics
    if not token_counts:
        print("No valid samples found!")
        return
    
    token_counts = np.array(token_counts)
    
    print("\n" + "="*50)
    print("TOKEN LENGTH STATISTICS")
    print("="*50)
    print(f"Total samples processed: {total_samples}")
    print(f"Average token length: {np.mean(token_counts):.2f}")
    print(f"Median token length: {np.median(token_counts):.2f}")
    print(f"Min token length: {np.min(token_counts)}")
    print(f"Max token length: {np.max(token_counts)}")
    print(f"Standard deviation: {np.std(token_counts):.2f}")
    print(f"95th percentile: {np.percentile(token_counts, 95):.2f}")
    print(f"99th percentile: {np.percentile(token_counts, 99):.2f}")
    
    # Token length distribution
    print("\nTOKEN LENGTH DISTRIBUTION:")
    print("-" * 30)
    bins = [0, 1024, 2048, 4096, 8192, 16384, 32768, 40960, float('inf')]
    bin_labels = ['0-1K', '1K-2K', '2K-4K', '4K-8K', '8K-16K', '16K-32K', '32K-40k', '40K+']
    
    for i in range(len(bins)-1):
        count = np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
        percentage = count / len(token_counts) * 100
        print(f"{bin_labels[i]}: {count} samples ({percentage:.1f}%)")
    
    # Save filtered data if max_token_length is specified and output_path is provided
    if max_token_length is not None and output_path is not None:
        print(f"\nFiltered samples (≤ {max_token_length} tokens): {len(filtered_data)}")
        print(f"Saving filtered data to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in filtered_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Successfully saved {len(filtered_data)} samples to {output_path}")
    elif max_token_length is not None:
        print(f"\nFiltered samples (≤ {max_token_length} tokens): {len(filtered_data)}")
        print("Note: Use --output_path to save filtered data")

def compute_token_counts_for_file(jsonl_file_path, model_path, num_processes):
    """Compute token counts for all samples in a single JSONL file."""
    print(f"Reading file: {jsonl_file_path}")
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines to process in {Path(jsonl_file_path).name}: {len(lines)}")

    line_data = [(i, line) for i, line in enumerate(lines)]

    token_counts = []
    print("Processing lines in parallel...")
    with Pool(processes=num_processes, initializer=init_worker, initargs=(model_path,)) as pool:
        results = list(tqdm(
            pool.imap(process_single_line, line_data, chunksize=100),
            total=len(line_data),
            desc=f"Processing {Path(jsonl_file_path).name}"
        ))

    for token_count, data, error in results:
        if error:
            print(error)
            continue
        if token_count is not None:
            token_counts.append(token_count)

    return np.array(token_counts)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_length_distributions_improved(file_token_counts_map, bins, bin_labels, plot_path, percentage=True):
    """
    改进版：提供多种可视化方案
    """
    
    # 方案1：分面图（多子图）- 推荐用于文件数量较多的情况
    def plot_faceted(output_name):
        files = list(file_token_counts_map.keys())
        n_files = len(files)
        
        # 计算子图布局（每行3个）
        n_cols = 3
        n_rows = (n_files + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_files > 1 else [axes]
        
        x = np.arange(len(bin_labels))
        
        for idx, (fname, token_counts) in enumerate(file_token_counts_map.items()):
            ax = axes[idx]
            
            if token_counts.size == 0:
                y = np.zeros(len(bin_labels))
            else:
                counts = np.array([
                    np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
                    for i in range(len(bins)-1)
                ])
                y = counts / token_counts.size * 100.0 if percentage else counts
            
            # 简化文件名显示
            short_name = fname.replace('.jsonl', '').replace('.json', '')
            if len(short_name) > 40:
                short_name = short_name[:37] + '...'
            
            ax.plot(x, y, marker='o', linewidth=2, markersize=6, color='#2E86AB')
            ax.fill_between(x, y, alpha=0.3, color='#2E86AB')
            ax.set_title(short_name, fontsize=10, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax.set_ylabel('Percentage (%)' if percentage else 'Count')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
        
        # 隐藏多余的子图
        for idx in range(n_files, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Saved faceted plot to: {output_name}")
        plt.close()
    
    # 方案2：热力图
    def plot_heatmap(output_name):
        files = list(file_token_counts_map.keys())
        data_matrix = []
        
        for fname, token_counts in file_token_counts_map.items():
            if token_counts.size == 0:
                row = np.zeros(len(bin_labels))
            else:
                counts = np.array([
                    np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
                    for i in range(len(bins)-1)
                ])
                row = counts / token_counts.size * 100.0 if percentage else counts
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 简化文件名
        short_names = []
        for f in files:
            name = f.replace('.jsonl', '').replace('.json', '')
            if len(name) > 50:
                name = name[:47] + '...'
            short_names.append(name)
        
        plt.figure(figsize=(12, max(8, len(files) * 0.4)))
        sns.heatmap(data_matrix, 
                    xticklabels=bin_labels, 
                    yticklabels=short_names,
                    annot=True, 
                    fmt='.1f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Percentage (%)' if percentage else 'Count'},
                    linewidths=0.5)
        
        plt.title('Token Length Distribution Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Token Length Bins', fontsize=12)
        plt.ylabel('Dataset Files', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_name}")
        plt.close()
    
    # 方案3：改进的单图版本（使用更好的颜色和样式）
    def plot_single_improved(output_name):
        plt.figure(figsize=(14, 8))
        x = np.arange(len(bin_labels))
        
        # 使用更好的颜色方案
        colors = plt.cm.tab20(np.linspace(0, 1, len(file_token_counts_map)))
        
        for idx, (fname, token_counts) in enumerate(file_token_counts_map.items()):
            if token_counts.size == 0:
                y = np.zeros(len(bin_labels))
            else:
                counts = np.array([
                    np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
                    for i in range(len(bins)-1)
                ])
                y = counts / token_counts.size * 100.0 if percentage else counts
            
            # 简化标签
            short_name = fname.replace('.jsonl', '').replace('.json', '')
            if len(short_name) > 35:
                short_name = short_name[:32] + '...'
            
            plt.plot(x, y, marker='o', label=short_name, 
                    linewidth=2, markersize=5, color=colors[idx], alpha=0.8)
        
        plt.xticks(x, bin_labels, rotation=45, ha='right')
        plt.xlabel("Token Length Bins", fontsize=12, fontweight='bold')
        plt.ylabel("Percentage (%)" if percentage else "Count", fontsize=12, fontweight='bold')
        plt.title("Token Length Distribution Comparison", fontsize=14, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Saved improved single plot to: {output_name}")
        plt.close()
    
    # 方案4：分组对比（选择几个重点文件）
    def plot_focused(output_name, top_n=5):
        """只显示样本量最大的top_n个文件"""
        # 按样本数量排序
        sorted_files = sorted(file_token_counts_map.items(), 
                            key=lambda x: x[1].size, 
                            reverse=True)[:top_n]
        
        plt.figure(figsize=(12, 7))
        x = np.arange(len(bin_labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_files)))
        
        for idx, (fname, token_counts) in enumerate(sorted_files):
            if token_counts.size == 0:
                y = np.zeros(len(bin_labels))
            else:
                counts = np.array([
                    np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
                    for i in range(len(bins)-1)
                ])
                y = counts / token_counts.size * 100.0 if percentage else counts
            
            short_name = fname.replace('.jsonl', '').replace('.json', '')
            if len(short_name) > 40:
                short_name = short_name[:37] + '...'
            
            plt.plot(x, y, marker='o', label=f"{short_name} (n={token_counts.size})", 
                    linewidth=2.5, markersize=7, color=colors[idx], alpha=0.9)
        
        plt.xticks(x, bin_labels, rotation=45, ha='right', fontsize=11)
        plt.xlabel("Token Length Bins", fontsize=13, fontweight='bold')
        plt.ylabel("Percentage (%)" if percentage else "Count", fontsize=13, fontweight='bold')
        plt.title(f"Token Length Distribution - Top {top_n} Largest Datasets", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Saved focused plot to: {output_name}")
        plt.close()
    
    # 生成所有版本
    base_path = Path(plot_path).stem
    base_dir = Path(plot_path).parent
    
    # 1. 分面图（推荐）
    plot_faceted(base_dir / f"{base_path}_faceted.png")
    
    # 2. 热力图
    plot_heatmap(base_dir / f"{base_path}_heatmap.png")
    
    # 3. 改进的单图
    plot_single_improved(base_dir / f"{base_path}_improved.png")
    
    # 4. 聚焦图（只显示前5大数据集）
    plot_focused(base_dir / f"{base_path}_top5.png", top_n=5)
    
    print("\n✨ 已生成4种不同风格的可视化图表：")
    print(f"  1. {base_path}_faceted.png  - 分面图（推荐用于详细对比）")
    print(f"  2. {base_path}_heatmap.png  - 热力图（推荐用于整体模式）")
    print(f"  3. {base_path}_improved.png - 改进的线图（所有文件）")
    print(f"  4. {base_path}_top5.png     - 聚焦图（前5大数据集）")


# 使用示例：
# 在原脚本的main()函数中，将原来的plot_length_distributions替换为：
# plot_length_distributions_improved(file_token_counts_map, bins, bin_labels, args.plot_path, percentage=True)


def plot_length_distributions(file_token_counts_map, bins, bin_labels, plot_path, percentage=True):
    """Plot length distribution lines for multiple files on one figure."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(bin_labels))

    for fname, token_counts in file_token_counts_map.items():
        if token_counts.size == 0:
            y = np.zeros(len(bin_labels))
        else:
            counts = np.array([
                np.sum((token_counts >= bins[i]) & (token_counts < bins[i+1]))
                for i in range(len(bins)-1)
            ])
            y = counts / token_counts.size * 100.0 if percentage else counts
        plt.plot(x, y, marker='o', label=fname)

    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel("Token length bins")
    plt.ylabel("Percentage (%)" if percentage else "Count")
    plt.title("Token length distribution per file")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate token length statistics for JSONL dataset or a directory of JSONL files')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_path', type=str, help='Path to a single JSONL data file')
    group.add_argument('--data_dir', type=str, help='Directory containing JSONL data files (.jsonl)')

    parser.add_argument('--model_path', type=str,
                       default='/nfs-13/wuxiaoyu/model_train/model/Qwen3-8B',
                       help='Path to the model/tokenizer directory')
    parser.add_argument('--max_token_length', type=int, default=None,
                       help='Maximum token length to filter data (optional, single-file mode)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for filtered data (optional, single-file mode)')
    parser.add_argument('--num_processes', type=int, default=16,
                       help='Number of processes to use for parallel processing')
    parser.add_argument('--plot_path', type=str, default='token_length_distributions.png',
                       help='Output path for the plot when using --data_dir')

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return

    # Directory mode: process all jsonl files and plot distributions on one figure
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists() or not data_dir.is_dir():
            print(f"Error: Directory not found: {data_dir}")
            return

        files = sorted([p for p in data_dir.glob('*.jsonl') if p.is_file()])
        if not files:
            print(f"No .jsonl files found in: {data_dir}")
            return

        print(f"Found {len(files)} JSONL files in {data_dir}")

        file_token_counts_map = {}
        for fp in files:
            token_counts = compute_token_counts_for_file(str(fp), args.model_path, args.num_processes)
            file_token_counts_map[fp.name] = token_counts

        # Define bins and labels (same as single-file stats)
        bins = [0, 1024, 2048, 4096, 8192, 16384, 32768, 40960, 80960, float('inf')]
        bin_labels = ['0-1K', '1K-2K', '2K-4K', '4K-8K', '8K-16K', '16K-32K', '32K-40k', '40K-80k', '80K+']

        plot_length_distributions_improved(file_token_counts_map, bins, bin_labels, args.plot_path, percentage=True)
        return

    # Single-file mode: keep original behavior
    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            return
        calculate_token_statistics(str(data_path), str(model_path), args.max_token_length, args.output_path, args.num_processes)

if __name__ == "__main__":
    main()
    # Examples:
    # Single file statistics and optional filtering
    # python token_len_count.py --data_path ./oai_survey_logs_250924.jsonl --model_path /data02/home/zdhs0071/base_checkpoints/gpt-oss-20b --max_token_length 40960 --output_path ./oai_survey_logs_250924_max_len_40k.jsonl --num_processes 8
    # python token_len_count.py --data_path /nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/chem_18018_mat_37088.jsonl  --num_processes 8 --plot_path ./token_length_distributions.png
    # python token_len_count.py --data_path /nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/Annealing_890.jsonl  --num_processes 8 --plot_path ./token_length_distributions.png
    
    # Directory mode: plot distributions for all files in the directory
    # python token_len_count.py --data_dir ./data_jsonl_dir --model_path /data02/home/zdhs0071/base_checkpoints/gpt-oss-20b --num_processes 8 --plot_path ./token_length_distributions.png
    
    
    # python token_len_count.py --data_dir /data02/group/wenge/wenge_src_data/long-context-250929/downsample_v1.2_251014 --model_path /data02/home/zdhs0071/base_checkpoints/Qwen3-8B --num_processes 8
    
    
    '''
    ==================================================
    TOKEN LENGTH STATISTICS
    ==================================================
    Total samples processed: 890
    Average token length: 3465.28
    Median token length: 2037.00
    Min token length: 257
    Max token length: 40847
    Standard deviation: 5389.86
    95th percentile: 16571.00
    99th percentile: 26817.20

    TOKEN LENGTH DISTRIBUTION:
    ------------------------------
    0-1K: 286 samples (32.1%)
    1K-2K: 160 samples (18.0%)
    2K-4K: 294 samples (33.0%)
    4K-8K: 77 samples (8.7%)
    8K-16K: 25 samples (2.8%)
    16K-32K: 45 samples (5.1%)
    32K-40k: 3 samples (0.3%)
    40K+: 0 samples (0.0%)

    '''