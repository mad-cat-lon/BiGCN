import wandb
import sys

# Replace with your project name and entity
ENTITY = "mulchmaxxer"  # e.g., "my-username" or your team name
PROJECT = "RFBoostedBiGCN"  # e.g., "my-project"

def edit(old, new):
    # Old and new variable names
    OLD_KEY = old
    NEW_KEY = new

    # Initialize API
    api = wandb.Api()

    # Fetch all runs for the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    for run in runs:
        print(f"Processing run: {run.id}")

        # Get the logged history for the run
        history = run.history(pandas=False)

        # Check if the old key exists and update it
        if OLD_KEY in run.summary:
            # Update the summary key
            run.summary[NEW_KEY] = run.summary[OLD_KEY]
            del run.summary[OLD_KEY]
            run.summary.update()

            print(f"Updated key in summary for run {run.id}")

        # Optionally: update logged history (careful as this can be large)
        for row in history:
            if OLD_KEY in row:
                row[NEW_KEY] = row[OLD_KEY]
                del row[OLD_KEY]

        # Save
        run.update()
        print(f"Finished updating run {run.id}")
    print("All runs updated!")


def add_new(key, val):
    NEW_KEY = key
    NEW_VAL = val
    
    # Initialize API
    api = wandb.Api()

    # Fetch all runs for the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    for run in runs:
        print(f"Processing run: {run.id}")

        # Get the logged history for the run
        if NEW_KEY not in run.config:
            run.config[NEW_KEY] = int(NEW_VAL)
            run.config.update()
            print(f"Added key '{NEW_KEY}' with value {NEW_KEY} for run {run.id}")
    
        run.update()
        print(f"Finished updating run {run.id}")
    print("All runs updated!")


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == "-e":
        edit(sys.argv[2], sys.argv[3])
    elif opt == "-a":
        add_new(sys.argv[2], sys.argv[3])
    else:
        print("Invalid option")