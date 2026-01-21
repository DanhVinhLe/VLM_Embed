#!/usr/bin/env python3
"""
Parse evaluation results and generate a CSV file inside the input directory.
Auto-detects CLS or VQA task and sorts rows/columns according to specific paper standards.

Usage:
    python parse_eval_results.py --input path/to/CLS_results_folder
    python parse_eval_results.py --input path/to/VQA_results_folder
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s' # Simplified log format
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION: ORDERING & NAMING ---

# 1. Column Order (Datasets)
CLS_DATASETS_ORDER = ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397"]
VQA_DATASETS_ORDER = ["OK-VQA", "A-OKVQA", "DocVQA", "I-VQA", "ChartQA", "Visual7W"]

# 2. Row Order (Methods) - Priorities
# The script checks if the folder name contains these keywords to sort.
METHOD_PRIORITY = [
    "Teacher",
    "Student",
    "RKD",
    "MSE",
    "CKD",
    "EMO",
    "EM-KD",
    "Our"
]

# 3. Display Names (Optional: Map folder keywords to fancy names in CSV)
DISPLAY_NAMES_MAP = {
    "Teacher": "Teacher (Thirukovalluru et al., 2025)",
    "Student": "Student (Vasu et al., 2025)",
    "RKD": "RKD (Park et al., 2019)",
    "MSE": "MSE",
    "CKD": "CKD (Wilf et al., 2025)",
    "EMO": "EMO (Truong et al., 2025)",
    "EM-KD": "EM-KD (Feng et al., 2025b)",
    "Our": "Our"
}

def get_method_sort_key(exp_name: str) -> int:
    """Returns a sorting index based on METHOD_PRIORITY."""
    exp_lower = exp_name.lower()
    for index, key in enumerate(METHOD_PRIORITY):
        if key.lower() in exp_lower:
            return index
    return 999  # Put unknown methods at the bottom

def get_display_name(exp_name: str) -> str:
    """Returns the fancy display name if a keyword matches, else returns original name."""
    exp_lower = exp_name.lower()
    for key, display_str in DISPLAY_NAMES_MAP.items():
        # Check exact match or likely substring match
        if key.lower() == exp_lower or key.lower() in exp_lower.split('_'):
             return display_str
    return exp_name

# ----------------------------------------

def find_experiments(root_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find immediate subdirectories containing score files.
    Assumes structure: root_dir/Method_Name/*.json
    """
    experiments = []
    
    # We look for folders inside the root_dir that contain JSONs
    if not root_dir.exists():
        return []

    # Iterate over items in root_dir
    for item in root_dir.iterdir():
        if item.is_dir():
            # Check if this folder contains any _score.json files
            score_files = list(item.glob('*_score.json'))
            if score_files:
                experiments.append((item.name, item))
    
    # Sort experiments based on the custom Method Priority
    experiments.sort(key=lambda x: get_method_sort_key(x[0]))
    
    return experiments

def parse_score_file(score_file: Path) -> float:
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
            # Support both 'acc' and 'accuracy' keys, or multiply by 100 if needed
            val = data.get('acc', data.get('accuracy', None))
            return val
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to parse {score_file}: {e}")
        return None

def collect_results(experiments: List[Tuple[str, Path]]) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Collects results and determines if this is a CLS or VQA task.
    """
    results = {}
    all_datasets_found = set()
    
    for exp_name, exp_path in experiments:
        results[exp_name] = {}
        score_files = list(exp_path.glob('*_score.json'))
        
        for score_file in score_files:
            # Clean dataset name: remove '_score' suffix
            dataset_name = score_file.stem.replace('_score', '')
            
            # Simple normalization to match the Config lists (case insensitive check usually safer)
            # But here we assume filenames match the config keys roughly
            
            val = parse_score_file(score_file)
            if val is not None:
                results[exp_name][dataset_name] = val
                all_datasets_found.add(dataset_name)

    # Detect Task Type based on intersection with known lists
    cls_overlap = len(all_datasets_found.intersection(set(CLS_DATASETS_ORDER)))
    vqa_overlap = len(all_datasets_found.intersection(set(VQA_DATASETS_ORDER)))
    
    if vqa_overlap > cls_overlap:
        task_type = 'VQA'
        logger.info(f"-> Detected Task: VQA (Found datasets: {all_datasets_found})")
    else:
        task_type = 'CLS'
        logger.info(f"-> Detected Task: CLS (Found datasets: {all_datasets_found})")
        
    return results, task_type

def write_csv(results: Dict[str, Dict[str, float]], 
              task_type: str, 
              output_file: Path) -> None:
    
    # Select the correct column order
    if task_type == 'VQA':
        columns = VQA_DATASETS_ORDER
    else:
        columns = CLS_DATASETS_ORDER
        
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header: Method + Dataset Columns + Avg
        header = ['Method'] + columns + ['Avg']
        writer.writerow(header)
        
        # Rows: Already sorted by experiment priority in find_experiments
        # But results dict keys are the original folder names
        
        # We need to iterate through results in the sorted order of methods
        # Create a sorted list of keys based on priority again to be safe
        sorted_exp_names = sorted(results.keys(), key=get_method_sort_key)
        
        for exp_name in sorted_exp_names:
            row_data = results[exp_name]
            
            # Prepare CSV Row
            display_name = get_display_name(exp_name)
            row = [display_name]
            
            valid_scores = []
            
            for dataset in columns:
                # Try to find the dataset in the results (handle potential minor naming mismatches if needed)
                # Here we assume exact match or simple inclusion
                score = None
                
                # Direct lookup
                if dataset in row_data:
                    score = row_data[dataset]
                else:
                    # Fallback: try to find dataset name inside the key (e.g. key="ImageNet-1K_val" vs "ImageNet-1K")
                    for k, v in row_data.items():
                        if dataset.lower() == k.lower():
                            score = v
                            break
                
                if score is not None:
                    # Format: 2 decimal places if it looks like percentage, or raw
                    # Assuming input is like 0.616 -> display 61.6 or 0.616? 
                    # The prompt table implies 0,616 (European format) or just raw. 
                    # Let's keep it standard float for now, maybe round to 4 decimals.
                    row.append(f"{score:.4f}")
                    valid_scores.append(score)
                else:
                    row.append("-")
            
            # Calculate Average
            if valid_scores:
                avg = sum(valid_scores) / len(valid_scores)
                row.append(f"{avg:.4f}")
            else:
                row.append("-")
                
            writer.writerow(row)
            
    logger.info(f"Successfully wrote results to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Parse eval results to CSV in the folder.')
    parser.add_argument('--input', type=str, required=True, help='Path to the task folder (e.g., eval_outputs/CLS)')
    
    args = parser.parse_args()
    input_dir = Path(args.input)
    
    if not input_dir.is_dir():
        logger.error(f"Error: {input_dir} is not a directory.")
        return

    logger.info(f"Scanning folder: {input_dir}")
    
    # 1. Find Experiments (Methods)
    experiments = find_experiments(input_dir)
    if not experiments:
        logger.warning("No subdirectories with *_score.json files found.")
        return

    # 2. Collect Data & Detect Task
    results, task_type = collect_results(experiments)
    
    # 3. Generate Output Filename inside the input directory
    # e.g., eval_outputs/CLS/CLS_results.csv
    output_file = input_dir / f"{task_type}_results.csv"
    
    # 4. Write CSV
    write_csv(results, task_type, output_file)

if __name__ == '__main__':
    main()