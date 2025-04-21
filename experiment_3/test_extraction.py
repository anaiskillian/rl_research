import csv
import json
import re

def extract_boxed_answer(text):
    """Extract the boxed answer from the response."""
    if not isinstance(text, str):
        return None
        
    # Try to find answer after #### marker
    hash_pattern = r"####\s*(-?\d+(?:\.\d+)?)"
    hash_match = re.search(hash_pattern, text)
    if hash_match:
        return hash_match.group(1).strip()
    
    # Try to find answer after ### marker
    triple_hash_pattern = r"###\s*(-?\d+(?:\.\d+)?)"
    triple_hash_match = re.search(triple_hash_pattern, text)
    if triple_hash_match:
        return triple_hash_match.group(1).strip()
    
    # Try to find LaTeX boxed answer
    boxed_pattern = r"\\boxed{([^}]+)}"
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try to find answer at the end of the text
    end_pattern = r"(-?\d+(?:\.\d+)?)\s*$"
    end_match = re.search(end_pattern, text)
    if end_match:
        return end_match.group(1).strip()
    
    # If no boxed answer found, look for other patterns
    patterns = [
        r"(?:final answer|answer is|=)\s*(-?\d+(?:\.\d+)?)",  # matches "answer is 42" or "= 42"
        r"(-?\d+(?:\.\d+)?)\s*$",  # matches number at the end of string
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return None

def test_extraction():
    # Process just first 5 rows from each file
    results = {
        'cot': [],
        'nocot': []
    }
    
    # Process CoT results
    print("\nTesting CoT file:")
    with open('/n/netscratch/kdbrantley_lab/Lab/akillian/results2/cot/results_shot2_cot.csv', 'r') as f:
        reader = csv.DictReader(f)
        # Print CSV structure
        print("\nCSV Structure:")
        print(f"Columns: {reader.fieldnames}")
        
        for i, row in enumerate(reader):
            if i >= 5:  # Only process first 5 rows
                break
            
            response = row['first_turn_response']
            boxed_answer = extract_boxed_answer(response)
            ground_truth = row['ground_truth_answer']
            ground_truth_answer = extract_boxed_answer(ground_truth)
            
            print(f"\nRow {i+1}:")
            print(f"Response length: {len(response)} characters")
            print(f"Response preview: {response[:100]}...")
            print(f"Response end: ...{response[-100:]}")
            print(f"Extracted answer: {boxed_answer}")
            print(f"Ground truth answer: {ground_truth_answer}")
            
            results['cot'].append({
                'problem_idx': row['problem_idx'],
                'problem': row['problem'],
                'boxed_answer': boxed_answer,
                'ground_truth': ground_truth_answer
            })
    
    # Process NoCoT results
    print("\nTesting NoCoT file:")
    with open('/n/netscratch/kdbrantley_lab/Lab/akillian/results2/nocot/results_shot2_nocot.csv', 'r') as f:
        reader = csv.DictReader(f)
        # Print CSV structure
        print("\nCSV Structure:")
        print(f"Columns: {reader.fieldnames}")
        
        for i, row in enumerate(reader):
            if i >= 5:  # Only process first 5 rows
                break
                
            response = row['first_turn_response']
            boxed_answer = extract_boxed_answer(response)
            ground_truth = row['ground_truth_answer']
            ground_truth_answer = extract_boxed_answer(ground_truth)
            
            print(f"\nRow {i+1}:")
            print(f"Response length: {len(response)} characters")
            print(f"Response preview: {response[:100]}...")
            print(f"Response end: ...{response[-100:]}")
            print(f"Extracted answer: {boxed_answer}")
            print(f"Ground truth answer: {ground_truth_answer}")
            
            results['nocot'].append({
                'problem_idx': row['problem_idx'],
                'problem': row['problem'],
                'boxed_answer': boxed_answer,
                'ground_truth': ground_truth_answer
            })
    
    # Save results to JSON file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to test_results.json")

if __name__ == '__main__':
    test_extraction() 