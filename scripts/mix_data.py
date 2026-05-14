#!/usr/bin/env python3
"""
混合多个JSONL数据集，随机打乱后合并到一个新文件。

用法:
    python mix_data.py --inputs file1.jsonl file2.jsonl ... --output merged.jsonl [--seed 42]
    python /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/scripts/mix_data.py \
        --inputs_dir /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/mix_codealpaca_20k_nemotron_40k_orcamath_10k \
        --output /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/cache/data/regen_data/mix_codealpaca_20k_nemotron_40k_orcamath_10k/merged.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List


def read_jsonl(file_path: str) -> List[dict]:
    """读取JSONL文件，返回所有记录列表。"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: 解析错误 {file_path}:{line_num} - {e}", file=sys.stderr)
    return records


def write_jsonl(records: List[dict], file_path: str) -> None:
    """将记录列表写入JSONL文件。"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def mix_datasets(input_files: List[str], output_file: str, seed: int = None) -> dict:
    """
    混合多个JSONL数据集。

    Args:
        input_files: 输入JSONL文件路径列表
        output_file: 输出文件路径
        seed: 随机种子（可选）

    Returns:
        包含统计信息的字典
    """
    if seed is not None:
        random.seed(seed)

    all_records = []
    stats = {
        'input_files': {},
        'total_input': 0,
        'total_output': 0,
        'seed': seed
    }

    # 读取所有文件
    for file_path in input_files:
        print(f"Reading: {file_path}")
        records = read_jsonl(file_path)
        all_records.extend(records)
        stats['input_files'][file_path] = len(records)
        stats['total_input'] += len(records)
        print(f"  -> {len(records)} records")

    # 随机打乱
    print(f"\nShuffling {len(all_records)} records...")
    random.shuffle(all_records)

    # 写入输出文件
    write_jsonl(all_records, output_file)
    stats['total_output'] = len(all_records)

    print(f"\nSaved to: {output_file}")
    print(f"Total records: {len(all_records)}")

    return stats


def get_jsonl_files_from_dir(directory: str) -> List[str]:
    """从目录中获取所有JSONL文件路径。"""
    dir_path = Path(directory)
    files = sorted([str(f) for f in dir_path.glob('*.jsonl')])
    return files


def main():
    parser = argparse.ArgumentParser(
        description='混合多个JSONL数据集，随机打乱后合并',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 混合指定文件
  python mix_data.py --inputs a.jsonl b.jsonl c.jsonl --output merged.jsonl

  # 使用随机种子（可复现）
  python mix_data.py --inputs a.jsonl b.jsonl --output merged.jsonl --seed 42

  # 混合目录下所有jsonl文件
  python mix_data.py --inputs_dir /path/to/data --output merged.jsonl
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--inputs', '-i',
        nargs='+',
        help='输入JSONL文件路径列表'
    )
    group.add_argument(
        '--inputs_dir', '-d',
        help='输入目录，自动读取所有.jsonl文件'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（用于可复现的打乱）'
    )

    args = parser.parse_args()

    # 获取输入文件列表
    if args.inputs_dir:
        input_files = get_jsonl_files_from_dir(args.inputs_dir)
        if not input_files:
            print(f"Error: 目录 {args.inputs_dir} 中没有找到.jsonl文件", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(input_files)} .jsonl files in {args.inputs_dir}")
    else:
        input_files = args.inputs

    # 检查文件是否存在
    for f in input_files:
        if not Path(f).exists():
            print(f"Error: 文件不存在: {f}", file=sys.stderr)
            sys.exit(1)

    # 执行混合
    stats = mix_datasets(input_files, args.output, args.seed)

    # 打印统计
    print("\n" + "="*50)
    print("Mixing Statistics:")
    print("="*50)
    for file_path, count in stats['input_files'].items():
        print(f"  {file_path}: {count}")
    print(f"\n  Total: {stats['total_output']} records")
    if stats['seed'] is not None:
        print(f"  Random seed: {stats['seed']}")


if __name__ == '__main__':
    main()
