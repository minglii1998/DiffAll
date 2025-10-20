import json

def process_sat_math_json(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    filtered_count = 0
    total_questions = len(data)
    
    for item in data:
        has_table = bool(item.get("Table"))
        has_figure = bool(item.get("Figure"))
        
        # Skip questions that have table or figure
        if has_table or has_figure:
            filtered_count += 1
            continue
            
        processed_text = "Below is a SAT Math question.\nQuestion: " +  item.get("Question", "") + "\n" + item.get("Item Stem", "")

        choices = []
        if item.get("Choice A"):
            choices.append("(A) " + item["Choice A"])
        if item.get("Choice B"):
            choices.append("(B) " + item["Choice B"])
        if item.get("Choice C"):
            choices.append("(C) " + item["Choice C"])
        if item.get("Choice D"):
            choices.append("(D) " + item["Choice D"])
        
        if choices:
            processed_text += "\nOptions:\n" + "\n".join(choices)

        difficulty = item.get("Question Difficulty", "N/A")
        
        processed_item = {
            "processed_text": processed_text.strip(),
            "Difficulty": difficulty,
            "answer": item.get("Correct Answer", "N/A"),
            "ori_data": item
        }
        processed_data.append(processed_item)
    
    print(f"Total questions: {total_questions}")
    print(f"Questions filtered out (with table/figure): {filtered_count}")
    print(f"Questions kept (without table/figure): {len(processed_data)}")
    print(f"Percentage kept: {len(processed_data)/total_questions*100:.2f}%")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, indent=4)

if __name__ == '__main__':
    input_file = '/nfshomes/minglii/scratch/DiffAll/SAT_math/SAT_math_1385_ori_format.json'
    output_file = '/nfshomes/minglii/scratch/DiffAll/SAT_math/processed_SAT_math_output.json'
    process_sat_math_json(input_file, output_file)
    print(f'Successfully processed {input_file} into {output_file}')
