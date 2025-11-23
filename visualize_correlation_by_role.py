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

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
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
    'diff': '#1f77b4',  # 蓝色
    'diff_role_weak': '#ff7f0e',  # 橙色
    'diff_role_medium': '#2ca02c',  # 绿色
    'diff_role_strong': '#d62728'  # 红色
}

# 条件标签（用于图例）
CONDITION_LABELS = {
    'diff': 'No Role',
    'diff_role_weak': 'Weak Role',
    'diff_role_medium': 'Medium Role',
    'diff_role_strong': 'Strong Role'
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


def calculate_correlation_from_jsonl(jsonl_file):
    """
    从jsonl文件计算correlation
    
    Args:
        jsonl_file: jsonl文件路径
    
    Returns:
        dict: 包含correlation统计信息的字典，如果失败返回None
    """
    if not os.path.exists(jsonl_file):
        return None
    
    true_difficulties = []
    predicted_difficulties = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 获取真实difficulty
                    true_diff = data.get('difficulty')
                    if true_diff is None:
                        continue
                    
                    # 获取模型响应
                    model_response = data.get('model_response', '')
                    
                    # 提取预测的difficulty
                    predicted_diff = extract_predicted_difficulty(model_response)
                    
                    if predicted_diff is not None:
                        true_difficulties.append(true_diff)
                        predicted_difficulties.append(predicted_diff)
                        
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        if len(true_difficulties) < 2:
            return None
        
        # 转换为numpy数组
        true_difficulties = np.array(true_difficulties)
        predicted_difficulties = np.array(predicted_difficulties)
        
        # 计算Spearman相关系数
        correlation, p_value = spearmanr(true_difficulties, predicted_difficulties)
        
        # 检查nan或inf
        if np.isnan(correlation) or np.isinf(correlation):
            return None
        
        return {
            'spearman_correlation': float(correlation),
            'p_value': float(p_value),
            'num_samples': len(true_difficulties)
        }
    except Exception as e:
        return None


def load_all_role_results(task='Cambridge'):
    """
    从所有role条件目录加载结果
    
    Args:
        task: 任务名称
    
    Returns:
        dict: {model: {condition: correlation_data}}
    """
    all_results = {}
    
    for condition, result_dir in ROLE_CONDITIONS.items():
        if not os.path.exists(result_dir):
            print(f"警告: 目录不存在，跳过: {result_dir}")
            continue
        
        print(f"\n正在处理条件: {condition} ({result_dir})")
        pattern = os.path.join(result_dir, f"{task}_*_results.jsonl")
        jsonl_files = glob.glob(pattern)
        
        print(f"  找到 {len(jsonl_files)} 个文件")
        
        for jsonl_file in jsonl_files:
            # 从文件名提取model名
            basename = os.path.basename(jsonl_file)
            name_without_suffix = basename.replace('_results.jsonl', '')
            
            if not name_without_suffix.startswith(f"{task}_"):
                continue
            
            model_name = name_without_suffix[len(task) + 1:]
            
            # 跳过Gemma系列的模型
            if 'gemma' in model_name.lower():
                print(f"  跳过Gemma系列模型: {model_name}")
                continue
            
            print(f"  计算: {model_name}")
            correlation_data = calculate_correlation_from_jsonl(jsonl_file)
            
            if correlation_data is None:
                print(f"    警告: 无法计算correlation，跳过")
                continue
            
            if model_name not in all_results:
                all_results[model_name] = {}
            
            all_results[model_name][condition] = correlation_data
    
    return all_results


def plot_grouped_bar_chart(all_results, task='Cambridge', output_file=None, figsize=(20, 8)):
    """
    绘制分组柱状图，比较每个模型在不同role条件下的表现
    
    Args:
        all_results: 从load_all_role_results返回的结果字典
        task: 任务名称
        output_file: 输出图片文件路径
        figsize: 图片大小
    """
    # 收集所有模型和条件
    models = sorted(all_results.keys())
    conditions = list(ROLE_CONDITIONS.keys())
    
    if len(models) == 0:
        print("错误: 没有找到任何模型数据")
        return
    
    # 准备数据
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图参数
    n_models = len(models)
    n_conditions = len(conditions)
    bar_width = 0.2
    x = np.arange(n_models)
    
    # 绘制每个条件的柱状图
    bars = []
    for i, condition in enumerate(conditions):
        values = data_matrix[:, i]
        # 过滤掉nan值用于绘图
        valid_mask = ~np.isnan(values)
        x_positions = x + i * bar_width
        
        # 只绘制有效值
        valid_x = x_positions[valid_mask]
        valid_values = values[valid_mask]
        
        color = COLORS[condition]
        bar = ax.bar(valid_x, valid_values, bar_width, 
                    label=CONDITION_LABELS[condition], 
                    color=color, alpha=0.8, edgecolor='black', linewidth=1)
        bars.append(bar)
        
        # 在柱状图上添加数值标签
        for x_pos, val in zip(valid_x, valid_values):
            ax.text(x_pos, val + 0.01, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 设置x轴
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
    ax.set_xticks(x + bar_width * (n_conditions - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    
    # 设置y轴范围
    all_valid_values = data_matrix[~np.isnan(data_matrix)]
    if len(all_valid_values) > 0:
        y_min = min(0, all_valid_values.min() - 0.1)
        y_max = all_valid_values.max() + 0.15
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([-0.1, 0.5])
    
    # 添加网格和图例
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 添加标题
    ax.set_title(f'Spearman Correlation by Model and Role Condition - {task}',
                fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化不同role条件下模型预测difficulty和真实difficulty的Spearman相关系数"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="Cambridge",
        help="要可视化的任务名称 (默认: Cambridge)"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="输出图片文件路径 (默认: model_results/{task}_correlation_by_role.png)"
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="20,8",
        help="图片大小，格式: width,height (默认: 20,8)"
    )
    
    args = parser.parse_args()
    
    # 解析figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        figsize = (20, 8)
    
    # 确定输出文件路径
    if not args.output_file:
        args.output_file = f"model_results/{args.task}_correlation_by_role.png"
    
    # 从所有role条件目录加载结果
    print(f"正在从所有role条件目录加载 {args.task} 任务的结果...")
    all_results = load_all_role_results(task=args.task)
    
    if not all_results:
        print("错误: 没有找到任何结果")
        return
    
    # 打印统计信息
    print(f"\n成功加载 {len(all_results)} 个模型的结果:")
    for model_name, model_data in sorted(all_results.items()):
        print(f"  {model_name}:")
        for condition, data in model_data.items():
            print(f"    {CONDITION_LABELS[condition]}: correlation={data['spearman_correlation']:.4f}, p={data['p_value']:.2e}")
    
    # 绘制分组柱状图
    print(f"\n正在生成分组柱状图...")
    plot_grouped_bar_chart(
        all_results,
        task=args.task,
        output_file=args.output_file,
        figsize=figsize
    )
    
    print("\n可视化完成！")


if __name__ == "__main__":
    main()

