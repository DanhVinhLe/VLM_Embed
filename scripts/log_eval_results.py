import wandb
import os
import json
import sys
import glob

def log_eval(run_id, project_name, output_dir):
    print(f"Logging evaluation results to Run ID: {run_id}")
    
    # Re-init wandb with the SAME ID as training, in resume mode
    run = wandb.init(project=project_name, id=run_id, resume="allow")
    
    # Find all score files
    score_files = glob.glob(os.path.join(output_dir, "*_score.json"))
    
    data = []
    columns = ["Subset", "Accuracy", "Num Correct", "Total"]
    
    print(f"Found {len(score_files)} score files in {output_dir}")
    
    for s_file in score_files:
        try:
            subset_name = os.path.basename(s_file).replace("_score.json", "")
            with open(s_file, 'r') as f:
                content = json.load(f)
                
            # Extract metrics
            acc = content.get("acc", 0.0)
            n_corr = content.get("num_correct", 0)
            total = content.get("num_data", 0)
            
            data.append([subset_name, acc, n_corr, total])
        except Exception as e:
            print(f"Error reading {s_file}: {e}")

    # Create and Log Table
    if data:
        # Sort by Subset name
        data.sort(key=lambda x: x[0])
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"Evaluation_Results": table})
        print("Table logged successfully.")
    else:
        print("No data found to log.")

    wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python log_eval_results.py <run_id> <project> <output_dir>")
        sys.exit(1)
    
    log_eval(sys.argv[1], sys.argv[2], sys.argv[3])