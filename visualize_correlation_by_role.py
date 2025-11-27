#!/usr/bin/env python3
"""
可视化不同role条件下模型预测difficulty和真实difficulty的Spearman相关系数
比较每个模型在4种不同role条件下的表现 (Research Bold + Default Colors)
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
from tqdm import tqdm

# ==================== 1. 样式配置 ====================

# 字体优先使用 Arial
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# === 全局加粗设置 ===
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['figure.titleweight'] = 'bold'

# 线条样式
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.width'] = 1.2
matplotlib.rcParams['ytick.major.width'] = 1.2

# ==================== 2. 配色方案 (还原默认) ====================
# 使用 Matplotlib 经典的 "tab10" 默认配色
COLORS = {
    'diff':             '#1f77b4',  # 默认蓝 (tab:blue)
    'diff_role_weak':   '#ff7f0e',  # 默认橙 (tab:orange)
    'diff_role_medium': '#2ca02c',  # 默认绿 (tab:green)
    'diff_role_strong': '#d62728'   # 默认红 (tab:red)
}

ROLE_CONDITIONS = {
    'diff': 'model_results/diff',
    'diff_role_weak': 'model_results/diff_role_weak',
    'diff_role_medium': 'model_results/diff_role_medium',
    'diff_role_strong': 'model_results/diff_role_strong'
}

CONDITION_LABELS = {
    'diff': 'No Role',
    'diff_role_weak': 'Weak Role',
    'diff_role_medium': 'Medium Role',
    'diff_role_strong': 'Strong Role'
}

# 模型分组
MODEL_GROUPS = [
    # Group 1: OpenAI
    [
        ('gpt35', 'GPT-3.5-Turbo'),
        ('gpt4o', 'GPT-4o'),
        ('gpt4omini', 'GPT-4o-mini'),
        ('gpt41', 'GPT-4.1'),
        ('gpt41mini', 'GPT-4.1-mini'),
        ('o4mini', 'GPT-o4-mini'),
        ('gpt5', 'GPT-5'),
    ],
    # Group 2: Standard Open Source
    [
        ('Llama2_7B', 'Llama2-7B'),
        ('Llama2_13B', 'Llama2-13B'),
        ('Llama3_1_8B', 'Llama3.1-8B'),
        ('Qwen25_7B', 'Qwen2.5-7B'),
        ('Qwen25_32B', 'Qwen2.5-32B'),
        ('Phi3_4k', 'Phi3'),
        ('Phi35_4k', 'Phi3.5'),
        ('Phi4', 'Phi4'),
        ('Qwen3_8B_NR', 'Qwen3-8B'),
        ('Qwen3_32B_NR', 'Qwen3-32B'),
    ],
    # Group 3: Reasoning
    [
        ('deepseekR1', 'DeepSeek-R1'), 
        ('QwQ32B', 'QWQ-32B'),
        ('R1_Distill_Qwen_32B', 'R1-Distill-Qwen32B'),
        ('Qwen3_32B_R', 'Qwen3-32B (R)'), 
    ]
]

CLASSIFICATION_TASKS = {'SAT_math', 'SAT_reading'}

# ===========================================

def convert_difficulty_to_numeric(difficulty_str):
    if not isinstance(difficulty_str, str): return None
    d = difficulty_str.lower().strip()
    return 0 if d == 'easy' else 1 if d == 'medium' else 2 if d == 'hard' else None

def extract_predicted_difficulty_classification(model_response):
    if not model_response: return None
    matches = list(re.finditer(r'\b(easy|medium|hard)\b', model_response, re.IGNORECASE))
    return matches[-1].group(1).lower() if matches else None

def extract_predicted_difficulty(model_response):
    if not model_response: return None
    match = re.search(r'\\boxed\s*\{\s*(\d+(?:\.\d+)?)\s*\}', model_response)
    if match: return float(match.group(1))
    matches = list(re.finditer(r'\*\*\s*(\d+(?:\.\d+)?)\s*\*\*', model_response))
    if matches: return float(matches[-1].group(1))
    patterns = [
        r'Final\s+(?:Difficulty\s+)?Value[:\s]+(\d+(?:\.\d+)?)',
        r'Final\s+Difficulty[:\s]+(\d+(?:\.\d+)?)',
        r'difficulty\s+(?:value|level|is|of)\s+(?:is\s+)?(?:around\s+)?(\d+(?:\.\d+)?)',
        r'score[:\s]+(\d+(?:\.\d+)?)'
    ]
    for p in patterns:
        m = re.search(p, model_response, re.IGNORECASE)
        if m: 
            val = float(m.group(1))
            if 0 <= val <= 100: return val
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', model_response)
    if nums:
        for n_str in reversed(nums):
            try:
                val = float(n_str)
                if 0 <= val <= 100: return val
            except: continue
    return None

def calculate_correlation_from_jsonl(jsonl_file, task=None):
    if not os.path.exists(jsonl_file): return None
    is_cls = task in CLASSIFICATION_TASKS if task else False
    true_diffs, pred_diffs = [], []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    td = data.get('difficulty')
                    if td is None: td = data.get('Difficulty')
                    if td is None: continue
                    resp = data.get('model_response', '')
                    if is_cls:
                        pd_str = extract_predicted_difficulty_classification(resp)
                        if pd_str is None: continue
                        pd = convert_difficulty_to_numeric(pd_str)
                        td_num = convert_difficulty_to_numeric(str(td))
                        if td_num is not None:
                            true_diffs.append(td_num)
                            pred_diffs.append(pd)
                    else:
                        try: td_float = float(td)
                        except: continue
                        pd = extract_predicted_difficulty(resp)
                        if pd is not None:
                            true_diffs.append(td_float)
                            pred_diffs.append(pd)
                except: continue
        if len(true_diffs) < 2: return None
        ta, pa = np.array(true_diffs), np.array(pred_diffs)
        if np.std(ta) == 0 or np.std(pa) == 0: return None
        corr, p = spearmanr(ta, pa)
        if np.isnan(corr): return None
        return {'spearman_correlation': corr, 'p_value': p}
    except: return None

def load_all_role_results(task='Cambridge'):
    all_results = {}
    for condition, result_dir in ROLE_CONDITIONS.items():
        if not os.path.exists(result_dir): continue
        pattern = os.path.join(result_dir, f"{task}_*_results.jsonl")
        files = glob.glob(pattern)
        for fpath in tqdm(files, desc=f"Loading {condition}", leave=False):
            base = os.path.basename(fpath).replace('_results.jsonl', '')
            if not base.startswith(f"{task}_"): continue
            raw_model_name = base[len(task)+1:]
            if 'gemma' in raw_model_name.lower(): continue
            res = calculate_correlation_from_jsonl(fpath, task)
            if res:
                if raw_model_name not in all_results: all_results[raw_model_name] = {}
                all_results[raw_model_name][condition] = res
    return all_results

def plot_grouped_bar_chart(all_results, task, output_file, figsize):
    models_to_plot = []
    for g_idx, group in enumerate(MODEL_GROUPS):
        for raw_name, display_name in group:
            if raw_name in all_results:
                models_to_plot.append({'raw': raw_name, 'display': display_name, 'group': g_idx})

    if not models_to_plot:
        print("Error: No matching model data found.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    conditions = list(ROLE_CONDITIONS.keys())
    n_conds = len(conditions)
    
    # === 紧凑布局参数 ===
    bar_width = 0.22
    standard_step = 1.0
    extra_group_gap = 0.35
    
    x_centers = []
    group_boundaries = []
    current_x = 0
    
    for i, item in enumerate(models_to_plot):
        if i == 0:
            current_x = 0
        else:
            prev_group = models_to_plot[i-1]['group']
            curr_group = item['group']
            
            if curr_group != prev_group:
                step = standard_step + extra_group_gap
                current_x += step
                boundary_pos = current_x - (step / 2)
                group_boundaries.append(boundary_pos)
            else:
                current_x += standard_step
        x_centers.append(current_x)
    
    # 绘制柱状图
    for c_idx, condition in enumerate(conditions):
        vals = []
        xs = []
        for i, item in enumerate(models_to_plot):
            raw = item['raw']
            if condition in all_results[raw]:
                vals.append(all_results[raw][condition]['spearman_correlation'])
            else:
                vals.append(np.nan)
            
            offset = (c_idx - (n_conds - 1) / 2) * bar_width
            xs.append(x_centers[i] + offset)
            
        xs = np.array(xs)
        vals = np.array(vals)
        mask = ~np.isnan(vals)
        
        ax.bar(xs[mask], vals[mask], width=bar_width, 
               label=CONDITION_LABELS[condition], 
               color=COLORS[condition], 
               alpha=0.85,          # 稍微透明一点点，避免默认色太刺眼
               edgecolor='black', 
               linewidth=0.8, 
               zorder=3)

        for x, v in zip(xs[mask], vals[mask]):
            ax.text(x, v + 0.02, f'{v:.2f}', 
                    ha='center', va='bottom', 
                    fontsize=7, fontweight='bold', rotation=90)

    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

    ax.set_xticks(x_centers)
    display_names = [m['display'] for m in models_to_plot]
    ax.set_xticklabels(display_names, rotation=40, ha='right', fontsize=11, fontweight='bold')
    
    total_group_width = bar_width * n_conds
    left_limit = x_centers[0] - (total_group_width / 2) - 0.4
    right_limit = x_centers[-1] + (total_group_width / 2) + 0.4
    ax.set_xlim(left_limit, right_limit)

    ax.set_ylabel('Spearman Correlation', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Model Performance by Role Condition - {task}', fontsize=16, fontweight='bold', pad=15)
    
    all_vals = []
    for m in all_results.values():
        for d in m.values():
            all_vals.append(d['spearman_correlation'])
    
    if all_vals:
        y_max = max(all_vals)
        y_min = min(all_vals)
        ax.set_ylim([min(0, y_min - 0.05), y_max + 0.15])

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.4, linewidth=1.0, zorder=0)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="Cambridge")
    parser.add_argument("--output_file", "-o", type=str, default=None)
    parser.add_argument("--figsize", type=str, default="16,6") 
    args = parser.parse_args()
    
    try: figsize = tuple(map(float, args.figsize.split(',')))
    except: figsize = (16, 6)
    
    if not args.output_file:
        if not os.path.exists('model_results'): os.makedirs('model_results')
        args.output_file = f"model_results/{args.task}_bold_default.png"
    
    print(f"Loading results for task: {args.task}...")
    results = load_all_role_results(args.task)
    
    print("Generating bold plot with default colors...")
    plot_grouped_bar_chart(results, args.task, args.output_file, figsize)
    print("Done.")

if __name__ == "__main__":
    main()