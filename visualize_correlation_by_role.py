#!/usr/bin/env python3
"""
可视化不同role条件下模型预测difficulty和真实difficulty的Spearman相关系数
比较每个模型在4种不同role条件下的表现
"""

import json
import os
import argparse
import glob
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm  # 建议添加 tqdm 显示进度

# 设置字体 (注：如果需要显示中文，请确保系统安装了 SimHei 或类似字体并添加到此处)
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义4种role条件及其对应的目录
ROLE_CONDITIONS = {
    'diff': 'model_results/diff',
    'diff_role_weak': 'model_results/diff_role_weak',
    'diff_role_medium': 'model_results/diff_role_medium',
    'diff_role_strong': 'model_results/diff_role_strong'
}

# 定义颜色方案
COLORS = {
    'diff': '#1f77b4',       # 蓝色
    'diff_role_weak': '#ff7f0e',   # 橙色
    'diff_role_medium': '#2ca02c', # 绿色
    'diff_role_strong': '#d62728'  # 红色
}

# 条件标签（用于图例）
CONDITION_LABELS = {
    'diff': 'No Role',
    'diff_role_weak': 'Weak Role',
    'diff_role_medium': 'Medium Role',
    'diff_role_strong': 'Strong Role'
}

# 定义分类任务（使用easy/medium/hard）
CLASSIFICATION_TASKS = {'SAT_math', 'SAT_reading'}


def convert_difficulty_to_numeric(difficulty_str):
    """
    将difficulty字符串转换为数值
    easy -> 0, medium -> 1, hard -> 2
    """
    if not isinstance(difficulty_str, str):
        return None
    
    difficulty_lower = difficulty_str.lower().strip()
    
    if difficulty_lower == 'easy':
        return 0
    elif difficulty_lower == 'medium':
        return 1
    elif difficulty_lower == 'hard':
        return 2
    else:
        return None


def extract_predicted_difficulty_classification(model_response):
    """
    从model_response中提取分类difficulty（easy/medium/hard）
    """
    if not model_response:
        return None
    
    # 查找所有easy/medium/hard的出现位置
    pattern = r'\b(easy|medium|hard)\b'
    matches = list(re.finditer(pattern, model_response, re.IGNORECASE))
    
    if matches:
        # 取最后一个匹配
        last_match = matches[-1]
        difficulty_str = last_match.group(1).lower()
        return difficulty_str
    
    return None


def extract_predicted_difficulty(model_response):
    """
    从model_response中提取模型预测的difficulty值（连续值）
    """
    if not model_response:
        return None
    
    # 方法1: 查找 \boxed{数字} 格式
    boxed_pattern = r'\\boxed\s*\{\s*(\d+(?:\.\d+)?)\s*\}' # 稍微放宽正则以允许空格
    match = re.search(boxed_pattern, model_response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # 方法2: 查找 **数字** 格式
    bold_pattern = r'\*\*\s*(\d+(?:\.\d+)?)\s*\*\*'
    matches = list(re.finditer(bold_pattern, model_response))
    if matches:
        try:
            return float(matches[-1].group(1))
        except ValueError:
            pass
    
    # 方法3: 查找特定关键词
    final_patterns = [
        r'Final\s+(?:Difficulty\s+)?Value[:\s]+(\d+(?:\.\d+)?)',
        r'Final\s+Difficulty[:\s]+(\d+(?:\.\d+)?)',
        r'difficulty\s+(?:value|level|is|of)\s+(?:is\s+)?(?:around\s+)?(\d+(?:\.\d+)?)',
        r'score[:\s]+(\d+(?:\.\d+)?)' # 增加 score 关键词
    ]
    for pattern in final_patterns:
        match = re.search(pattern, model_response, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                # 简单的合理性检查，避免提取到年份等
                if 0 <= val <= 100: 
                    return val
            except ValueError:
                continue
    
    # 方法4: 兜底 - 查找所有数字，取最后一个在合理范围内的数字
    # 注意：这可能会误判，例如 "Step 5", "Top 10" 等
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', model_response)
    if all_numbers:
        for num_str in reversed(all_numbers):
            try:
                num = float(num_str)
                # 排除整数 1-5，除非它们是唯一的数字，因为它们常用于列表序号
                # 这里只保留 0-100 的范围检查
                if 0 <= num <= 100:
                    return num
            except ValueError:
                continue
    
    return None


def calculate_correlation_from_jsonl(jsonl_file, task=None):
    """
    从jsonl文件计算correlation
    """
    if not os.path.exists(jsonl_file):
        return None
    
    is_classification = task in CLASSIFICATION_TASKS if task else False
    
    true_difficulties = []
    predicted_difficulties = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # --- FIX: 修复 0 值丢失的 bug ---
                    true_diff = data.get('difficulty')
                    if true_diff is None:
                        true_diff = data.get('Difficulty')
                    
                    if true_diff is None:
                        continue
                    # -------------------------------

                    model_response = data.get('model_response', '')
                    
                    if is_classification:
                        # 分类任务
                        predicted_diff_str = extract_predicted_difficulty_classification(model_response)
                        if predicted_diff_str is None:
                            continue
                        predicted_diff = convert_difficulty_to_numeric(predicted_diff_str)
                        
                        true_diff_numeric = convert_difficulty_to_numeric(str(true_diff)) # 确保转为str
                        if true_diff_numeric is None:
                            continue
                        
                        true_difficulties.append(true_diff_numeric)
                        predicted_difficulties.append(predicted_diff)
                    else:
                        # 连续值任务
                        try:
                            true_diff_float = float(true_diff)
                        except (ValueError, TypeError):
                            continue
                        
                        predicted_diff = extract_predicted_difficulty(model_response)
                        
                        if predicted_diff is not None:
                            true_difficulties.append(true_diff_float)
                            predicted_difficulties.append(predicted_diff)
                        
                except json.JSONDecodeError:
                    continue
        
        # 至少需要2个点才能计算相关系数，且不能全是常数
        if len(true_difficulties) < 2:
            return None
        
        # 转换为numpy数组
        true_arr = np.array(true_difficulties)
        pred_arr = np.array(predicted_difficulties)
        
        # 检查方差是否为0（全是一样的数值会导致 spearman 返回 nan）
        if np.std(true_arr) == 0 or np.std(pred_arr) == 0:
            return None

        correlation, p_value = spearmanr(true_arr, pred_arr)
        
        if np.isnan(correlation) or np.isinf(correlation):
            return None
        
        return {
            'spearman_correlation': float(correlation),
            'p_value': float(p_value),
            'num_samples': len(true_difficulties)
        }
    except Exception as e:
        print(f"Error processing {jsonl_file}: {e}")
        return None


def load_all_role_results(task='Cambridge'):
    all_results = {}
    
    # 遍历条件
    for condition, result_dir in ROLE_CONDITIONS.items():
        if not os.path.exists(result_dir):
            print(f"警告: 目录不存在，跳过: {result_dir}")
            continue
        
        print(f"\n正在处理条件: {condition} ({result_dir})")
        # 严格匹配 task 开头的文件
        pattern = os.path.join(result_dir, f"{task}_*_results.jsonl")
        jsonl_files = glob.glob(pattern)
        
        if not jsonl_files:
            print(f"  未找到匹配 {task} 的文件")
            continue
            
        print(f"  找到 {len(jsonl_files)} 个文件")
        
        # 使用 tqdm 显示进度
        for jsonl_file in tqdm(jsonl_files, desc="Processing Models"):
            basename = os.path.basename(jsonl_file)
            name_without_suffix = basename.replace('_results.jsonl', '')
            
            # 提取模型名，例如 Cambridge_gpt-4 -> gpt-4
            if not name_without_suffix.startswith(f"{task}_"):
                continue
            
            model_name = name_without_suffix[len(task) + 1:]
            
            if 'gemma' in model_name.lower():
                continue
            
            correlation_data = calculate_correlation_from_jsonl(jsonl_file, task=task)
            
            if correlation_data is None:
                continue
            
            if model_name not in all_results:
                all_results[model_name] = {}
            
            all_results[model_name][condition] = correlation_data
    
    return all_results


def plot_grouped_bar_chart(all_results, task='Cambridge', output_file=None, figsize=(20, 8)):
    models = sorted(all_results.keys())
    conditions = list(ROLE_CONDITIONS.keys())
    
    if len(models) == 0:
        print("错误: 没有找到任何模型数据")
        return
    
    data_matrix = []
    for model in models:
        model_data = []
        for condition in conditions:
            if condition in all_results[model]:
                corr = all_results[model][condition]['spearman_correlation']
                model_data.append(corr)
            else:
                model_data.append(np.nan)
        data_matrix.append(model_data)
    
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_models = len(models)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions # 动态调整宽度以适应
    x = np.arange(n_models)
    
    for i, condition in enumerate(conditions):
        values = data_matrix[:, i]
        # 计算偏移量，使组居中
        offset = (i - (n_conditions - 1) / 2) * bar_width
        x_positions = x + offset
        
        valid_mask = ~np.isnan(values)
        valid_x = x_positions[valid_mask]
        valid_values = values[valid_mask]
        
        color = COLORS[condition]
        ax.bar(valid_x, valid_values, bar_width, 
               label=CONDITION_LABELS[condition], 
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5) # 减小边框宽度
        
        # 仅当柱子较少时才显示数值，防止重叠
        if n_models < 15:
            for x_pos, val in zip(valid_x, valid_values):
                ax.text(x_pos, val + 0.01, f'{val:.2f}', # 改为2位小数更整洁
                       ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=0)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    
    # 动态调整 Y 轴
    all_valid_values = data_matrix[~np.isnan(data_matrix)]
    if len(all_valid_values) > 0:
        y_max = all_valid_values.max()
        y_min = all_valid_values.min()
        # 给顶部和底部留出空间
        ax.set_ylim([min(0, y_min - 0.05), y_max + 0.1])
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, bbox_to_anchor=(1, 1)) # 图例放外面防止遮挡
    ax.set_title(f'Spearman Correlation by Model and Role Condition - {task}',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存到: {output_file}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="Cambridge")
    parser.add_argument("--output_file", "-o", type=str, default=None)
    parser.add_argument("--figsize", type=str, default="20,8")
    
    args = parser.parse_args()
    
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        figsize = (20, 8)
    
    if not args.output_file:
        if not os.path.exists('model_results'):
            os.makedirs('model_results') # 确保输出目录存在
        args.output_file = f"model_results/{args.task}_correlation_by_role.png"
    
    print(f"正在从所有role条件目录加载 {args.task} 任务的结果...")
    all_results = load_all_role_results(task=args.task)
    
    if not all_results:
        print("错误: 没有找到任何结果")
        return
    
    # 打印简要统计
    print(f"\n成功加载 {len(all_results)} 个模型")
    
    plot_grouped_bar_chart(all_results, task=args.task, output_file=args.output_file, figsize=figsize)
    print("\n可视化完成！")

if __name__ == "__main__":
    main()