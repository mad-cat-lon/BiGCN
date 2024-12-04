import wandb

PROJECT_NAME = "RFBoostedBiGCN"  # 
ENTITY_NAME = "mulchmaxxer"    

api = wandb.Api()

runs = api.runs(f"{ENTITY_NAME}/{PROJECT_NAME}")

best_run = None
best_accuracy = float('-inf')

for run in runs:
    final_test_accuracy = run.summary.get('final_test_accuracy', None)
    
    if final_test_accuracy is not None and final_test_accuracy > best_accuracy:
        best_accuracy = final_test_accuracy
        best_run = run

if best_run:
    print(f"Best run ID: {best_run.id}")
    print(f"Run URL: {best_run.url}")
    print(best_run.summary)
    print("="*40)
    print(best_run.config)
else:
    print("No runs found with a final_test_accuracy metric.")