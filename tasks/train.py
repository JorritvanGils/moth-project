# /media/jorrit/ssd/phd/moth-project/tasks/train.py
import os
import yaml
from datetime import datetime
from models.yolo_model import YOLOModel
from models.vit_model import ViTModel

OUTPUT_ROOT = "/media/jorrit/ssd/phd/moth-project/outputs"

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_run_dir(exp_type: str, model_type: str, model_size: str) -> str:
    exp_dir = os.path.join(OUTPUT_ROOT, exp_type)
    os.makedirs(exp_dir, exist_ok=True)

    # Determine next run number
    existing = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    run_number = len(existing) + 1
    date_str = datetime.now().strftime("%Y%m%d")
    run_name = f"{run_number:03d}_{date_str}_{model_type}_{model_size}"
    run_path = os.path.join(exp_dir, run_name)
    os.makedirs(run_path, exist_ok=False)
    return run_path

def run_experiment(config_path: str):
    config = load_config(config_path)
    exp_type = config["experiment"]  # detection or classification
    exp_config = config[exp_type]

    model_type = exp_config.get("model_type")
    exp_config['experiment'] = exp_type

    # Determine model size
    if model_type == "yolo":
        model_size = exp_config.get("yolo", {}).get("model_size", "nano")
    elif model_type == "vit":
        model_size = exp_config.get("vit", {}).get("model_size", "base")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create unified output dir for this run
    run_dir = make_run_dir(exp_type, model_type, model_size)
    exp_config["output_path"] = run_dir  # override output_path for model

    # Initialize and train
    if model_type == "yolo":
        model = YOLOModel(exp_config)
    elif model_type == "vit":
        model = ViTModel(exp_config)

    model.train()
    print(f"Training complete. All results saved in {run_dir}")

if __name__ == "__main__":
    run_experiment("config.yaml")