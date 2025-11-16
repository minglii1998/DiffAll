#!/usr/bin/env python3
"""
统计各个task的inference情况
统计每个模式下：
- 哪些模型被inference了
- 每个模型有多少jsonl item
- 有多少能够提取出difficulty数字
"""

import json
import os
import argparse
import glob
import re
from collections import defaultdict

# 定义所有可能的模式
DIFF_MODES = {
    'diff': 'model_results/diff',
    'diff_role_weak': 'model_results/diff_role_weak',
    'diff_role_medium': 'model_results/diff_role_medium',
    'diff_role_strong': 'model_results/diff_role_strong'
}

DIRECT_MODES = {
    'direct_try1': 'model_results/direct_try1',
    'direct_role_weak': 'model_results/direct_role_weak',
    'direct_role_medium': 'model_results/direct_role_medium',
    'direct_role_strong': 'model_results/direct_role_strong'
}

# 所有模式的合并（用于验证）
ALL_MODES = {**DIFF_MODES, **DIRECT_MODES}

# 定义所有可能的任务
ALL_TASKS = ['Cambridge', 'SAT_math', 'SAT_reading', 'USMLE']

# 定义每个任务的期望数量
EXPECTED_COUNTS = {
    'Cambridge': 793,
    'SAT_math': 1075,
    'SAT_reading': 1338,
    'USMLE': 667
}


def extract_predicted_difficulty(model_response):
    """
    从model_response中提取模型预测的difficulty值
    
    优先查找 \boxed{数字}，如果没有则用正则找最后的数字值
    """
    if not model_response:
        return None
    
    # 方法1: 查找 \boxed{数字} 格式
    boxed_pattern = r'\\boxed\{(\d+(?:\.\d+)?)\}'
    match = re.search(boxed_pattern, model_response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # 方法2: 查找 **数字** 格式（加粗的数字，通常是最终答案）
    bold_pattern = r'\*\*(\d+(?:\.\d+)?)\*\*'
    matches = list(re.finditer(bold_pattern, model_response))
    if matches:
        # 取最后一个匹配（通常是最终答案）
        try:
            return float(matches[-1].group(1))
        except ValueError:
            pass
    
    # 方法3: 查找 "Final Value:" 或 "Final Difficulty Value:" 后面的数字
    final_patterns = [
        r'Final\s+(?:Difficulty\s+)?Value[:\s]+(\d+(?:\.\d+)?)',
        r'Final\s+Difficulty[:\s]+(\d+(?:\.\d+)?)',
        r'difficulty\s+(?:value|level|is|of)\s+(?:is\s+)?(?:around\s+)?(\d+(?:\.\d+)?)',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, model_response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # 方法4: 查找所有数字，取最后一个合理的数字（在0-100范围内）
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', model_response)
    if all_numbers:
        # 从后往前找，找到第一个在合理范围内的数字
        for num_str in reversed(all_numbers):
            try:
                num = float(num_str)
                if 0 <= num <= 100:  # difficulty通常在0-100范围内
                    return num
            except ValueError:
                continue
    
    return None


def analyze_jsonl_file(jsonl_file, task, check_extraction=True):
    """
    分析单个jsonl文件
    
    Args:
        jsonl_file: jsonl文件路径
        task: 任务名称
        check_extraction: 是否检查提取情况
    
    Returns:
        dict: {
            'total_items': int,
            'items_with_difficulty': int,
            'items_with_model_response': int,
            'items_with_extracted_difficulty': int (如果check_extraction=True),
            'extraction_rate': float (如果check_extraction=True)
        }
    """
    stats = {
        'total_items': 0,
        'items_with_difficulty': 0,
        'items_with_model_response': 0
    }
    
    if check_extraction:
        stats['items_with_extracted_difficulty'] = 0
        stats['extraction_rate'] = 0.0
    
    if not os.path.exists(jsonl_file):
        return stats
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    stats['total_items'] += 1
                    
                    # 检查是否有difficulty字段
                    if 'difficulty' in data and data['difficulty'] is not None:
                        stats['items_with_difficulty'] += 1
                    
                    # 检查是否有model_response字段
                    model_response = data.get('model_response', '')
                    if model_response:
                        stats['items_with_model_response'] += 1
                        
                        # 如果检查提取情况，尝试提取difficulty
                        if check_extraction:
                            extracted = extract_predicted_difficulty(model_response)
                            if extracted is not None:
                                stats['items_with_extracted_difficulty'] += 1
                            
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        # 计算提取率（如果检查提取）
        if check_extraction and stats['items_with_model_response'] > 0:
            stats['extraction_rate'] = stats['items_with_extracted_difficulty'] / stats['items_with_model_response']
        elif check_extraction:
            stats['extraction_rate'] = 0.0
            
    except Exception as e:
        print(f"Warning: Failed to read file {jsonl_file}: {e}")
    
    return stats


def analyze_mode(task, mode, check_extraction=True, verbose=False):
    """
    分析指定task和mode的inference情况
    
    Args:
        task: 任务名称
        mode: 模式名称（如 'diff', 'diff_role_weak', 'direct', 'direct_role_weak'）
        check_extraction: 是否检查提取情况
        verbose: 是否输出每个模型的详细信息
    
    Returns:
        dict: 统计结果
    """
    if mode not in ALL_MODES:
        print(f"Error: Unknown mode '{mode}'")
        print(f"Available modes: {list(ALL_MODES.keys())}")
        return None
    
    result_dir = ALL_MODES[mode]
    
    if not os.path.exists(result_dir):
        print(f"Warning: Directory does not exist: {result_dir}")
        return None
    
    # 查找所有匹配的jsonl文件
    pattern = os.path.join(result_dir, f"{task}_*_results.jsonl")
    jsonl_files = glob.glob(pattern)
    
    print(f"\n{'='*80}")
    print(f"Analyzing mode: {mode}")
    print(f"Directory: {result_dir}")
    print(f"Task: {task}")
    print(f"Found {len(jsonl_files)} model result files")
    print(f"{'='*80}")
    
    if len(jsonl_files) == 0:
        print("No result files found")
        return None
    
    # 分析每个模型
    model_stats = {}
    total_summary = {
        'total_models': 0,
        'total_items': 0,
        'total_items_with_difficulty': 0,
        'total_items_with_model_response': 0
    }
    if check_extraction:
        total_summary['total_items_with_extracted_difficulty'] = 0
    
    for jsonl_file in sorted(jsonl_files):
        # 从文件名提取model名
        basename = os.path.basename(jsonl_file)
        name_without_suffix = basename.replace('_results.jsonl', '')
        
        if not name_without_suffix.startswith(f"{task}_"):
            continue
        
        model_name = name_without_suffix[len(task) + 1:]
        
        # 跳过Gemma系列的模型
        if model_name.startswith('Gemma'):
            if verbose:
                print(f"\nSkipping model: {model_name} (Gemma series)")
            continue
        
        if verbose:
            print(f"\nAnalyzing model: {model_name}")
            print(f"  File: {basename}")
        
        stats = analyze_jsonl_file(jsonl_file, task, check_extraction=check_extraction)
        model_stats[model_name] = stats
        
        # 更新总计
        total_summary['total_models'] += 1
        total_summary['total_items'] += stats['total_items']
        total_summary['total_items_with_difficulty'] += stats['items_with_difficulty']
        total_summary['total_items_with_model_response'] += stats['items_with_model_response']
        if check_extraction:
            total_summary['total_items_with_extracted_difficulty'] += stats['items_with_extracted_difficulty']
        
        if verbose:
            print(f"  Total items: {stats['total_items']}")
            print(f"  Items with difficulty field: {stats['items_with_difficulty']}")
            print(f"  Items with model_response: {stats['items_with_model_response']}")
            if check_extraction:
                print(f"  Successfully extracted difficulty: {stats['items_with_extracted_difficulty']}")
                print(f"  Extraction rate: {stats['extraction_rate']:.2%}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary Statistics ({mode}):")
    print(f"{'='*80}")
    print(f"Number of models: {total_summary['total_models']}")
    print(f"Total items: {total_summary['total_items']}")
    print(f"Items with difficulty field: {total_summary['total_items_with_difficulty']}")
    print(f"Items with model_response: {total_summary['total_items_with_model_response']}")
    if check_extraction:
        print(f"Successfully extracted difficulty: {total_summary['total_items_with_extracted_difficulty']}")
        if total_summary['total_items_with_model_response'] > 0:
            overall_extraction_rate = total_summary['total_items_with_extracted_difficulty'] / total_summary['total_items_with_model_response']
            print(f"Overall extraction rate: {overall_extraction_rate:.2%}")
        else:
            print(f"Overall extraction rate: 0.00%")
    
    # Print model list (always output)
    print(f"\nModel List:")
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        if check_extraction:
            print(f"  - {model_name}: {stats['total_items']} items, "
                  f"extraction rate {stats['extraction_rate']:.2%}")
        else:
            print(f"  - {model_name}: {stats['total_items']} items")
    
    return {
        'mode': mode,
        'result_dir': result_dir,
        'model_stats': model_stats,
        'summary': total_summary
    }


def check_incomplete_inference(task, mode, model_stats, check_incomplete=True):
    """
    检查哪些模型没有inference完
    
    Args:
        task: 任务名称
        mode: 模式名称
        model_stats: 模型统计字典
        check_incomplete: 是否检查未完成情况
    
    Returns:
        list: 未完成的模型列表 [(model_name, expected, actual), ...]
    """
    if not check_incomplete:
        return []
    
    if task not in EXPECTED_COUNTS:
        return []
    
    expected_count = EXPECTED_COUNTS[task]
    incomplete_models = []
    
    for model_name, stats in model_stats.items():
        actual_count = stats['total_items']
        if actual_count < expected_count:
            incomplete_models.append((model_name, expected_count, actual_count))
    
    return incomplete_models


def main():
    parser = argparse.ArgumentParser(
        description="统计各个task的inference情况"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=False,
        help="任务名称 (如: Cambridge, SAT_math, SAT_reading, USMLE)"
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="分析所有任务（忽略--task参数）"
    )
    parser.add_argument(
        "--meta_mode",
        type=str,
        default="diff",
        choices=["diff", "direct"],
        help="Meta模式：'diff' 或 'direct'（默认：diff）"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        required=False,
        choices=list(ALL_MODES.keys()),
        help=f"模式名称，可选: {', '.join(ALL_MODES.keys())}"
    )
    parser.add_argument(
        "--all_modes",
        action="store_true",
        help="分析所有模式（根据--meta_mode选择diff或direct系列，忽略--mode参数）"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="输出JSON文件路径（可选，用于保存统计结果）"
    )
    parser.add_argument(
        "--check_extraction",
        action="store_true",
        default=False,
        help="是否统计提取情况（默认：False，只统计inference数量）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出每个模型的详细信息（默认：False，只输出汇总统计）"
    )
    parser.add_argument(
        "--check_incomplete",
        action="store_true",
        default=False,
        help="检查并打印所有未完成inference的情况（默认：False）"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_tasks and not args.task:
        parser.error("Must provide --task or --all_tasks")
    
    # Determine tasks to analyze
    if args.all_tasks:
        tasks_to_analyze = ALL_TASKS
        print(f"Analyzing all tasks: {', '.join(tasks_to_analyze)}")
    else:
        tasks_to_analyze = [args.task]
    
    # Select mode dictionary based on meta_mode
    if args.meta_mode == "diff":
        modes_dict = DIFF_MODES
        meta_label = "diff"
    else:
        modes_dict = DIRECT_MODES
        meta_label = "direct"
    
    # Validate mode arguments
    if not args.all_modes and not args.mode:
        parser.error("Must provide --mode or --all_modes")
    
    results = []
    all_incomplete = []  # Collect all incomplete cases
    
    # Iterate through all tasks
    for task in tasks_to_analyze:
        print(f"\n{'#'*80}")
        print(f"# Analyzing task: {task}")
        print(f"{'#'*80}")
        
        if args.all_modes:
            # Analyze all modes (select mode series based on meta_mode)
            print(f"Analyzing all {meta_label} modes for task '{task}'...")
            for mode in modes_dict.keys():
                result = analyze_mode(task, mode, 
                                     check_extraction=args.check_extraction,
                                     verbose=args.verbose)
                if result:
                    results.append(result)
                    # 检查未完成情况
                    if args.check_incomplete:
                        incomplete = check_incomplete_inference(
                            task, mode, result['model_stats'], 
                            check_incomplete=args.check_incomplete
                        )
                        for model_name, expected, actual in incomplete:
                            all_incomplete.append({
                                'task': task,
                                'mode': mode,
                                'model': model_name,
                                'expected': expected,
                                'actual': actual,
                                'missing': expected - actual
                            })
        else:
            # Analyze specified mode
            if not args.mode:
                parser.error("Must provide --mode or use --all_modes")
            result = analyze_mode(task, args.mode,
                                 check_extraction=args.check_extraction,
                                 verbose=args.verbose)
            if result:
                results.append(result)
                # 检查未完成情况
                if args.check_incomplete:
                    incomplete = check_incomplete_inference(
                        task, args.mode, result['model_stats'],
                        check_incomplete=args.check_incomplete
                    )
                    for model_name, expected, actual in incomplete:
                        all_incomplete.append({
                            'task': task,
                            'mode': args.mode,
                            'model': model_name,
                            'expected': expected,
                            'actual': actual,
                            'missing': expected - actual
                        })
    
    # Print all incomplete cases
    if args.check_incomplete and all_incomplete:
        print(f"\n{'='*80}")
        print(f"Incomplete Inference Summary:")
        print(f"{'='*80}")
        print(f"Found {len(all_incomplete)} incomplete cases\n")
        
        # Group by task and mode
        by_task_mode = {}
        for item in all_incomplete:
            key = (item['task'], item['mode'])
            if key not in by_task_mode:
                by_task_mode[key] = []
            by_task_mode[key].append(item)
        
        for (task, mode), items in sorted(by_task_mode.items()):
            print(f"\nTask: {task}, Mode: {mode}")
            print(f"  Expected count: {items[0]['expected']}")
            print(f"  Incomplete models:")
            for item in sorted(items, key=lambda x: x['missing'], reverse=True):
                print(f"    - {item['model']}: Actual {item['actual']} / Expected {item['expected']} "
                      f"(Missing {item['missing']} items)")
    elif args.check_incomplete:
        print(f"\n{'='*80}")
        print(f"Incomplete Inference Summary:")
        print(f"{'='*80}")
        print("All models have completed inference!")
    
    # Save results to JSON file (if specified)
    if args.output_file and results:
        output_data = {
            'tasks': tasks_to_analyze if args.all_tasks else [args.task],
            'meta_mode': args.meta_mode,
            'results': results
        }
        
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.output_file}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

