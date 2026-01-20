#!/usr/bin/env python3
"""
Parse evaluation results from eval_outputs folder and generate a CSV file.

This script reads all JSON score files from each experiment directory and
creates a CSV file where each row represents an experiment and each column
represents a dataset's accuracy score.

Usage:
    python parse_eval_results.py --input eval_outputs --output results.csv
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_experiments(eval_outputs_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all experiment directories in the eval_outputs folder.
    
    An experiment is identified as a directory containing *_score.json files.
    
    Args:
        eval_outputs_dir: Path to the eval_outputs directory
        
    Returns:
        List of tuples containing (experiment_name, experiment_path)
    """
    experiments = []
    
    # Recursively search for directories containing score files
    for path in eval_outputs_dir.rglob('*_score.json'):
        experiment_dir = path.parent
        
        # Create experiment name from relative path
        relative_path = experiment_dir.relative_to(eval_outputs_dir)
        experiment_name = str(relative_path).replace('/', '_')
        
        # Add to experiments list if not already present
        if (experiment_name, experiment_dir) not in experiments:
            experiments.append((experiment_name, experiment_dir))
    
    # Remove duplicates and sort
    experiments = sorted(list(set(experiments)))
    
    logger.info(f"Found {len(experiments)} experiments")
    for exp_name, exp_path in experiments:
        logger.info(f"  - {exp_name}: {exp_path}")
    
    return experiments


def parse_score_file(score_file: Path) -> float:
    """
    Parse a score JSON file and extract the accuracy value.
    
    Args:
        score_file: Path to the *_score.json file
        
    Returns:
        Accuracy value as a float, or None if parsing fails
    """
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
            return data.get('acc', None)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to parse {score_file}: {e}")
        return None


def collect_results(experiments: List[Tuple[str, Path]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Collect all results from experiments.
    
    Args:
        experiments: List of (experiment_name, experiment_path) tuples
        
    Returns:
        Tuple of (results_dict, sorted_dataset_names)
        - results_dict: {experiment_name: {dataset_name: accuracy}}
        - sorted_dataset_names: List of all unique dataset names, sorted
    """
    results = {}
    all_datasets = set()
    
    for exp_name, exp_path in experiments:
        results[exp_name] = {}
        
        # Find all score files in this experiment
        score_files = list(exp_path.glob('*_score.json'))
        logger.info(f"Processing {exp_name}: found {len(score_files)} score files")
        
        for score_file in score_files:
            # Extract dataset name from filename (remove _score.json suffix)
            dataset_name = score_file.stem.replace('_score', '')
            all_datasets.add(dataset_name)
            
            # Parse the score
            accuracy = parse_score_file(score_file)
            if accuracy is not None:
                results[exp_name][dataset_name] = accuracy
                logger.debug(f"  {dataset_name}: {accuracy}")
    
    # Sort dataset names for consistent column ordering
    sorted_datasets = sorted(list(all_datasets))
    logger.info(f"Found {len(sorted_datasets)} unique datasets")
    
    return results, sorted_datasets


def write_csv(results: Dict[str, Dict[str, float]], 
              datasets: List[str], 
              output_file: Path) -> None:
    """
    Write results to a CSV file.
    
    Args:
        results: Dictionary mapping experiment names to dataset results
        datasets: List of dataset names (column headers)
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        header = ['experiment'] + datasets
        writer.writerow(header)
        
        # Write data rows
        for exp_name in sorted(results.keys()):
            row = [exp_name]
            for dataset in datasets:
                # Get accuracy or empty string if not available
                accuracy = results[exp_name].get(dataset, '')
                row.append(accuracy)
            writer.writerow(row)
    
    logger.info(f"Results written to {output_file}")


def main():
    """Main function to parse evaluation results and generate CSV."""
    parser = argparse.ArgumentParser(
        description='Parse evaluation results and generate CSV file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='eval_outputs',
        help='Path to eval_outputs directory (default: eval_outputs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='eval_results.csv',
        help='Path to output CSV file (default: eval_results.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Convert paths to Path objects
    eval_outputs_dir = Path(args.input)
    output_file = Path(args.output)
    
    # Validate input directory
    if not eval_outputs_dir.exists():
        logger.error(f"Input directory does not exist: {eval_outputs_dir}")
        return 1
    
    if not eval_outputs_dir.is_dir():
        logger.error(f"Input path is not a directory: {eval_outputs_dir}")
        return 1
    
    # Find experiments
    logger.info(f"Scanning {eval_outputs_dir} for experiments...")
    experiments = find_experiments(eval_outputs_dir)
    
    if not experiments:
        logger.warning("No experiments found!")
        return 1
    
    # Collect results
    logger.info("Collecting results...")
    results, datasets = collect_results(experiments)
    
    # Write CSV
    logger.info("Writing CSV file...")
    write_csv(results, datasets, output_file)
    
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
