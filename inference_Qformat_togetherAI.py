import os
import json
import argparse
from tqdm import tqdm
from together import Together


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run inference on Cambridge MCQ test data with DeepSeek R1 via Together AI.")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1",
                        help='The model to use for inference (default: deepseek-ai/DeepSeek-R1)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Together AI API key (if not set, will use TOGETHER_API_KEY env variable)')
    parser.add_argument('--input_file', type=str, default='EduAgent_related/Cambridge_mcq_test_pub_Qformat.json',
                        help='Path to the input JSON file (Cambridge MCQ test data)')
    parser.add_argument('--output_file', type=str, default='EduAgent_related/model_results/Cambridge_mcq_results.jsonl',
                        help='Path to save the updated data as JSONL')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate for each prompt (default: 2048)')
    parser.add_argument('--start_ratio', type=float, default=0.0,
                        help='Start ratio in [0,1). Defaults to 0.0')
    parser.add_argument('--end_ratio', type=float, default=1.0,
                        help='End ratio in (0,1]. Defaults to 1.0 (process to end).')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='System prompt to use for the conversation (optional)')
    parser.add_argument('--prefix_prompt', type=str, default=None,
                        help='Prefix prompt to prepend to the user prompt (optional)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for the model (default: 0.0)')
    parser.add_argument('--append_answer', action='store_true',
                        help='If true, append the correct answer from data to the user prompt')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode for responses')
    args = parser.parse_args()

    print(f"Using model: {args.model_name}")
    
    # Initialize Together AI client
    if args.api_key:
        client = Together(api_key=args.api_key)
    else:
        # Will use TOGETHER_API_KEY environment variable
        client = Together()

    # Function to run inference on a single prompt
    def run_inference(prompt: str, item: dict = None) -> str:
        """
        Generate a response for the given prompt using Together AI API.
        """
        # Build the user prompt with prefix if provided
        user_content = prompt
        if args.prefix_prompt:
            user_content = args.prefix_prompt + prompt
        
        # Append answer if requested and available
        if args.append_answer and item and 'answer' in item:
            answer = item['answer']
            if answer and str(answer).strip():
                user_content += f"\nCorrect answer: {str(answer).upper()}"
        
        # Build messages array
        messages = []
        
        # Add system prompt if provided
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": user_content})
        
        try:
            if args.stream:
                # Streaming mode
                stream = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stream=True,
                )
                
                response_text = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                
                return response_text
            else:
                # Non-streaming mode
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error during API call: {e}")
            raise

    # Ensure output has the correct .jsonl extension
    if not args.output_file.endswith('.jsonl'):
        args.output_file += '.jsonl'

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # -----------------------------
    # Determine items already processed
    # -----------------------------
    processed_items = set()
    response_key = f"model_response"
    if os.path.exists(args.output_file):
        print(f"Reading already processed items from {args.output_file} to avoid re-processing...")
        with open(args.output_file, 'r', encoding='utf-8') as prev_f:
            for line in prev_f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines
                if isinstance(obj, dict) and 'processed_text' in obj and response_key in obj:
                    # Use processed_text as unique identifier for each item
                    processed_items.add(obj['processed_text'])
        print(f"Found {len(processed_items)} previously processed items.")

    # Read the input JSON (list of items)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON must contain a list of items.")

    print(f"Loaded {len(data)} items from {args.input_file}")

    # Determine slicing purely based on ratios
    if not (0.0 <= args.start_ratio < args.end_ratio <= 1.0):
        raise ValueError("start_ratio must be >=0 and < end_ratio <=1.")

    start_idx = int(len(data) * args.start_ratio)
    end_idx = int(len(data) * args.end_ratio)

    # Clamp indices (in case end_idx equals len(data))
    end_idx = min(end_idx, len(data))

    data = data[start_idx:end_idx]
    print(f"Processing items ratio range: [{args.start_ratio}, {args.end_ratio}) => index [{start_idx}, {end_idx}) -> {len(data)} items")

    # Decide write mode: append if file exists, else write
    out_mode = 'a' if os.path.exists(args.output_file) else 'w'

    # Open the output .jsonl file
    with open(args.output_file, out_mode, encoding='utf-8') as out_f:
        if out_mode == 'a':
            out_f.seek(0, os.SEEK_END)  # ensure we are at end for appending

        for idx, item in enumerate(tqdm(data, desc="Processing items")):
            # Validate item structure
            if not isinstance(item, dict):
                print(f"Warning: Not a dictionary item, skipping: {item}")
                continue
            if 'processed_text' not in item:
                print(f"Warning: Missing 'processed_text' field in item: {item}")
                continue

            prompt_text = item['processed_text']
            if not isinstance(prompt_text, str):
                print(f"Warning: processed_text field is not string, skipping: {prompt_text}")
                continue

            # Skip if we've already processed this item
            if prompt_text in processed_items:
                # Item is already in the output file with a response – skip re-processing
                continue

            # Run inference
            try:
                response_text = run_inference(prompt_text, item)
            except Exception as e:
                # Catch any errors, log and continue
                print(f"[ERROR] Skipping item index {idx}: {e}")
                continue
            
            # Add model response to the item
            item[response_key] = response_text

            # Write updated item to .jsonl
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

            # Mark as processed to avoid duplicates within the same run
            processed_items.add(prompt_text)

    print(f"\nProcessing complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()