# Use this script inside docker to run sequentially:
# - process data
# - train
# - visualize

import subprocess


def run_script(script_name):
    try:
        # Run the script
        subprocess.run(["python", script_name], check=True)
        print(f"Successfully executed {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_name}: {e}")


if __name__ == "__main__":
    run_script("mlops_project/data/make_dataset.py")
    run_script("mlops_project/train_model.py")
    run_script("mlops_project/visualizations/visualize.py")
