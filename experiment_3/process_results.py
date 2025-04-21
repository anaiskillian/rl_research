import pandas as pd
import json
import re

def extract_boxed_answer(text):
    """Extract the boxed answer from the response."""
    match = re.search(r'\\boxed{([^}]+)}', text)
    if match:
        return match.group(1)
    return None

def process_results():
    # Read the CSV files
    cot_df = pd.read_csv('/n/netscratch/kdbrantley_lab/Lab/akillian/results2/cot/results_shot2_cot.csv')
    nocot_df = pd.read_csv('/n/netscratch/kdbrantley_lab/Lab/akillian/results2/nocot/results_shot2_nocot.csv')
    
    results = {
        'cot': [],
        'nocot': []
    }
    
    # Process CoT results
    for idx, row in cot_df.iterrows():
        boxed_answer = extract_boxed_answer(row['first_turn_response'])
        ground_truth = row['ground_truth_answer']
        
        results['cot'].append({
            'problem_idx': row['problem_idx'],
            'problem': row['problem'],
            'boxed_answer': boxed_answer,
            'ground_truth': ground_truth
        })
    
    # Process NoCoT results
    for idx, row in nocot_df.iterrows():
        boxed_answer = extract_boxed_answer(row['first_turn_response'])
        ground_truth = row['ground_truth_answer']
        
        results['nocot'].append({
            'problem_idx': row['problem_idx'],
            'problem': row['problem'],
            'boxed_answer': boxed_answer,
            'ground_truth': ground_truth
        })
    
    # Save results to JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    process_results() 