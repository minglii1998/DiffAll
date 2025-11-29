#!/usr/bin/env python3
"""
使用 GPT-4o mini 来判断模型 response 是否正确。

需求：
1. 每个 input_dir 对应一个新的结果目录，例如：
   - model_results/direct_role_medium -> model_results/direct_role_medium_result
   - 目录结构与原来一致，文件名不变
   - 每条数据新增一个布尔字段 "Correct"，只取 True/False
2. GPT 的 evaluation prompt 不需要 explanation，只要输出 CORRECT / INCORRECT，方便正则提取。
3. 不需要给 Question，只给 Options、GT 选项和模型选项，让 GPT 判断两者是否一致。
4. Options 的提取规则：
   - Cambridge: 从 "\\nOptions:\\n" 到 "\\nReference Passage:" 之间
   - SAT_math: 从 "\\nOptions:\\n" 之后
   - SAT_reading: 从 "\\nOptions:\\n" 到 "\\nReference Passage:" 之间
   - USMLE: 从 "\\n(A)" 之后（包括 "(A)"）就是选项
"""

import os
import json
import argparse
import glob
import re
from tqdm import tqdm
from openai import OpenAI


# 定义任务和对应的答案格式（主要用于识别文件，不再在 prompt 里使用）
TASK_FORMATS = {
    'Cambridge': {
        'answer_format': 'letter',  # a, b, c, d (小写)
    },
    'SAT_math': {
        'answer_format': 'mixed',  # 可能是字母 (A, B, C, D) 或数字
    },
    'SAT_reading': {
        'answer_format': 'letter',  # A, B, C, D (大写)
    },
    'USMLE': {
        'answer_format': 'letter',  # A, B, C, D, E (大写)
    }
}


def extract_ground_truth(item):
    """
    从item中提取ground truth答案
    优先查找 "answer"，如果没有则查找 "Correct Answer" (可能在ori_data中)
    """
    # 方法1: 直接查找 "answer" 字段
    if 'answer' in item and item['answer'] is not None:
        answer = str(item['answer']).strip()
        if answer:
            return answer
    
    # 方法2: 在ori_data中查找 "Correct Answer"
    if 'ori_data' in item and isinstance(item['ori_data'], dict):
        ori_data = item['ori_data']
        if 'Correct Answer' in ori_data:
            answer = str(ori_data['Correct Answer']).strip()
            if answer:
                return answer
        if 'answer' in ori_data:
            answer = str(ori_data['answer']).strip()
            if answer:
                return answer
        # USMLE可能使用Answer_Key
        if 'Answer_Key' in ori_data:
            answer = str(ori_data['Answer_Key']).strip()
            if answer:
                return answer
    
    return None


def extract_model_answer(model_response, task):
    """
    从model_response中提取模型给出的答案
    优先查找 \boxed{} 格式，然后尝试其他方法
    """
    if not model_response:
        return None
    
    # 方法1: 查找 \boxed{答案} 格式
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, model_response)
    if match:
        answer = match.group(1).strip()
        # 清理可能的格式，如 \boxed{A} 或 \boxed{161.53} 或 \boxed{(-4, 0)}
        answer = answer.replace('$', '').replace('\\', '').strip()
        # 移除可能的括号
        answer = answer.strip('()')
        # 对于 Cambridge，选项通常为小写字母
        if task == 'Cambridge' and len(answer) == 1 and answer.isalpha():
            return answer.lower()
        return answer
    
    # 方法2: 查找 "Final answer:" 或类似格式
    final_patterns = [
        r'Final\s+answer[:\s]+([A-Ea-e]|\d+(?:\.\d+)?)',
        r'answer[:\s]+([A-Ea-e]|\d+(?:\.\d+)?)',
        r'correct\s+answer[:\s]+([A-Ea-e]|\d+(?:\.\d+)?)',
    ]
    for pattern in final_patterns:
        match = re.search(pattern, model_response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return answer
    
    # 方法3: 查找最后一个选项字母 (A-E) 或数字
    # 对于字母答案
    letter_pattern = r'\b([A-Ea-e])\b'
    matches = list(re.finditer(letter_pattern, model_response))
    if matches:
        # 取最后一个匹配
        answer = matches[-1].group(1)
        # Cambridge 使用小写，其他使用大写
        if task == 'Cambridge':
            return answer.lower()
        else:
            return answer.upper()
    
    # 对于数字答案（SAT Math可能有数字答案）
    if task == 'SAT_math':
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        matches = list(re.finditer(number_pattern, model_response))
        if matches:
            # 取最后一个匹配
            answer = matches[-1].group(1)
            return answer
    
    return None


def extract_options_block(processed_text, task):
    """
    根据 task 类型，从 processed_text 中抽取 Options 部分。
    规则：
      - Cambridge: "\\nOptions:\\n" 到 "\\nReference Passage:" 之间
      - SAT_math: "\\nOptions:\\n" 之后到文本末尾
      - SAT_reading: "\\nOptions:\\n" 到 "\\nReference Passage:" 之间
      - USMLE: 从 "\\n(A)" 之后（包括 "(A)"）到文本末尾
    """
    if not isinstance(processed_text, str):
        return ""

    text = processed_text

    if task in ("Cambridge", "SAT_reading"):
        start_tag = "\nOptions:\n"
        ref_tag = "\nReference Passage:"
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return ""
        start_idx += len(start_tag)
        end_idx = text.find(ref_tag, start_idx)
        if end_idx == -1:
            end_idx = len(text)
        return text[start_idx:end_idx].strip()

    if task == "SAT_math":
        start_tag = "\nOptions:\n"
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return ""
        start_idx += len(start_tag)
        return text[start_idx:].strip()

    if task == "USMLE":
        # USMLE: 选项从 "\n(A)" 开始（包括 "(A)"）
        start_tag = "\n(A)"
        start_idx = text.find(start_tag)
        if start_idx == -1:
            return ""
        # 保留 "(A)..."
        return text[start_idx + 1 :].strip()

    return ""


def create_judgment_prompt(options_text, model_response_text, ground_truth):
    """
    创建用于判断正确性的 prompt。

    只提供：
      - Options 文本
      - ground truth 选项（标签或内容）
      - 模型完整回答（或 </think> 之后的部分）

    让 GPT 从模型回答中自己抽取最终选择的选项，然后与 ground truth 对比，
    只输出 CORRECT / INCORRECT。
    """
    prompt = f"""You need to judge the correctness of the model's response based on the options and the ground truth option.

Options:
{options_text}

Ground truth option (correct answer): {ground_truth}

[Model Response]
{model_response_text}
[End of Model Response]

Carefully read the options and the model response.
If the model's final chosen answer matches the ground truth option (same option label or clearly the same content), you should answer CORRECT.
Otherwise you should answer INCORRECT.

You MUST respond with EXACTLY one word:
- CORRECT
- INCORRECT

Do NOT output anything else."""
    return prompt


def judge_with_gpt4omini(client, prompt, max_retries=5):
    """
    使用GPT-4o mini判断答案是否正确
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise evaluator. "
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )

            response_text = (response.choices[0].message.content or "").strip()
            upper = response_text.upper()

            # 只接受单词 CORRECT / INCORRECT，其余认为失败
            if upper == "CORRECT":
                return True
            if upper == "INCORRECT":
                return False

            print(f"  Unexpected GPT output: {response_text}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                continue
            else:
                print(f"  Error after {max_retries} attempts: {e}")
                return None

    return None


def process_jsonl_file(jsonl_file, task, client, output_dir, overwrite=False):
    """
    处理单个jsonl文件，判断每个item的正确性
    """
    if not os.path.exists(jsonl_file):
        print(f"  Warning: File does not exist: {jsonl_file}")
        return
    
    # 确定输出文件路径
    basename = os.path.basename(jsonl_file)
    output_file = os.path.join(output_dir, basename)
    
    # 如果输出文件已存在且不覆盖，检查已处理的项目
    processed_items = set()
    if os.path.exists(output_file) and not overwrite:
        print(f"  Reading existing results from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    # 使用processed_text作为唯一标识
                    if 'processed_text' in obj:
                        processed_items.add(obj['processed_text'])
                except:
                    continue
        print(f"  Found {len(processed_items)} previously processed items.")
    
    # 读取输入文件
    items_to_process = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # 检查是否已处理
                if 'processed_text' in item:
                    if item['processed_text'] not in processed_items:
                        items_to_process.append(item)
            except:
                continue
    
    print(f"  Processing {len(items_to_process)} items...")
    
    # 决定写入模式
    out_mode = 'a' if os.path.exists(output_file) and not overwrite else 'w'
    
    # 处理每个item
    with open(output_file, out_mode, encoding='utf-8') as out_f:
        if out_mode == 'a':
            out_f.seek(0, os.SEEK_END)
        
        for idx, item in enumerate(tqdm(items_to_process, desc=f"  Processing {basename}", leave=False)):
            # 提取 ground truth 和 model response
            ground_truth = extract_ground_truth(item)
            model_response = item.get('model_response', '')

            is_correct = False  # 默认 False，确保 "Correct" 始终是 True/False

            if not ground_truth:
                print(f"    Warning: No ground truth found for item {idx}")
            elif not model_response:
                print(f"    Warning: No model response for item {idx}")
            else:
                # 根据是否存在 </think> 来决定传给 GPT 的内容
                response_for_gpt = model_response
                think_tag = "</think>"
                pos = model_response.rfind(think_tag)
                if pos != -1:
                    response_for_gpt = model_response[pos + len(think_tag):].strip()

                processed_text = item.get('processed_text', '')
                options_text = extract_options_block(processed_text, task)
                prompt = create_judgment_prompt(
                    options_text=options_text,
                    model_response_text=response_for_gpt,
                    ground_truth=str(ground_truth),
                )
                gpt_result = judge_with_gpt4omini(client, prompt)
                if gpt_result is None:
                    print(f"    Warning: GPT judgment failed for item {idx}")
                else:
                    is_correct = bool(gpt_result)

            # 仅新增一个布尔字段 "Correct"
            item["Correct"] = is_correct

            # 写入结果
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()
    
    print(f"  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="使用GPT-4o mini判断模型response是否正确"
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='OpenAI API key (default: read from OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--input_dirs',
        type=str,
        nargs='+',
        default=[
            'model_results/direct_role_medium',
            'model_results/direct_role_strong',
            'model_results/direct_role_weak',
            'model_results/direct_try1'
        ],
        help='输入目录列表（包含jsonl文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='model_results',
        help='输出根目录（默认: model_results），每个输入目录会在该目录下创建 *_result 子目录'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        choices=['Cambridge', 'SAT_math', 'SAT_reading', 'USMLE'],
        help=''
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的输出文件（默认：追加）'
    )
    
    args = parser.parse_args()
    
    # 初始化OpenAI client
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    print(f"Using GPT-4o mini for judgment")
    print(f"Input directories: {args.input_dirs}")
    print(f"Output root directory: {args.output_dir}")
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个输入目录
    for input_dir in args.input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory does not exist: {input_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing directory: {input_dir}")
        print(f"{'='*80}")
        
        # 为该 input_dir 创建对应的 *_result 目录
        base_name = os.path.basename(os.path.normpath(input_dir))
        result_dir_name = f"{base_name}_result"
        output_dir_for_input = os.path.join(args.output_dir, result_dir_name)
        os.makedirs(output_dir_for_input, exist_ok=True)
        
        # 查找所有jsonl文件
        pattern = os.path.join(input_dir, "*.jsonl")
        jsonl_files = glob.glob(pattern)
        
        if not jsonl_files:
            print(f"  No jsonl files found in {input_dir}")
            continue
        
        # 按任务分组处理
        task_files = {}
        for jsonl_file in jsonl_files:
            basename = os.path.basename(jsonl_file)
            # 从文件名提取任务名
            for task in TASK_FORMATS.keys():
                if basename.startswith(f"{task}_"):
                    if task not in task_files:
                        task_files[task] = []
                    task_files[task].append(jsonl_file)
                    break
        
        # 处理每个任务
        for task, files in task_files.items():
            if args.task and task != args.task:
                continue
            
            print(f"\nTask: {task} ({len(files)} files)")
            
            for jsonl_file in sorted(files):
                process_jsonl_file(jsonl_file, task, client, output_dir_for_input, args.overwrite)
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

