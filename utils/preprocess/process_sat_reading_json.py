#!/usr/bin/env python3
"""
Script to process SAT Reading data into a standardized format,
similar to process_sat_math_json.py
"""

import json
from pathlib import Path

def process_sat_reading_json(input_file_path, output_file_path):
    """Process SAT Reading data into standardized format."""
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    total_questions = len(data)
    
    print(f"Processing {total_questions} SAT Reading questions...")
    
    for item in data:
        # Extract the main components
        prompt = item.get("Prompt", "")
        question_text = item.get("Question Text", "")
        
        # Build the processed text similar to SAT Math format
        processed_text = "Below is a SAT Reading and Writing question.\n"
        
        # Add question
        if question_text:
            processed_text += "Question: " + question_text + "\n"
        
        # Add choices
        choices = []
        if item.get("A"):
            choices.append("(A) " + item["A"])
        if item.get("B"):
            choices.append("(B) " + item["B"])
        if item.get("C"):
            choices.append("(C) " + item["C"])
        if item.get("D"):
            choices.append("(D) " + item["D"])
        
        if choices:
            processed_text += "Options:\n" + "\n".join(choices)

        # Add prompt if available
        if prompt:
            processed_text += "\nReference Passage: " + prompt + "\n"
        
        # Extract difficulty
        difficulty = item.get("Difficulty", "N/A")
        
        # Create processed item
        processed_item = {
            "processed_text": processed_text.strip(),
            "Difficulty": difficulty,
            "answer": item.get("Correct Answer", "N/A"),
            "ori_data": item
        }
        processed_data.append(processed_item)
    
    print(f"Total questions processed: {len(processed_data)}")
    print(f"Success rate: {len(processed_data)/total_questions*100:.2f}%")

    # Save processed data
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, indent=4, ensure_ascii=False)
    
    return len(processed_data)

def main():
    """Main function to process SAT Reading data."""
    sat_reading_dir = Path("/nfshomes/minglii/scratch/DiffAll/SAT_reading")
    input_file = sat_reading_dir / "SAT_reading_1338_ori_format.json"
    output_file = sat_reading_dir / "SAT_reading_Qformat.json"
    
    print("=== SAT Reading Data Processing ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        processed_count = process_sat_reading_json(input_file, output_file)
        print(f"✅ Successfully processed {processed_count} questions")
        print(f"📁 Output saved to: {output_file}")
        
        # Show file size
        file_size = output_file.stat().st_size
        print(f"📊 Output file size: {file_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")

if __name__ == '__main__':
    main()
