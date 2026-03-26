import os
import yaml
import argparse
from datetime import datetime
from typing import Type

from models.yolo_model import YOLOModel
from models.cls.vit_model import ViTModel
from models.det.sam_model import SAMModel

# python -m tasks.run_experiment --mode train --config configs/config.yaml
# python -m tasks.run_experiment --mode predict --path ~/moths/outputs/det/001_20260324_yolo_n
# python -m tasks.run_experiment --mode predict --path /media/jorrit/ssd/phd/outputs/det/001_20260324_yolo_n


OUTPUT_ROOT = os.path.expanduser("~/moths/outputs")
INFERENCE_SOURCE = "/media/jorrit/ssd/phd/moths/inference_img"

class ExperimentManager:
    def __init__(self, config_path: str = None, run_dir: str = None):
        self.run_dir = run_dir
        self.model = None
        
        if config_path:
            self.config = self._load_config(config_path)
            self.exp_type = self.config["experiment"]
            self.exp_config = self.config[self.exp_type]
        else:
            # If no config is provided, we are likely in standalone inference mode
            self.config = None
            self.exp_type = None
            self.exp_config = None

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def setup_run_directory(self):
        """Used only during training to create a new folder."""
        model_type = self.exp_config.get("model_type")
        m_size = self.exp_config.get("yolo", {}).get("model_size", "n") # Simplified for brevity
        
        base_path = os.path.join(OUTPUT_ROOT, self.exp_type)
        os.makedirs(base_path, exist_ok=True)

        existing = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        run_name = f"{len(existing)+1:03d}_{datetime.now().strftime('%Y%m%d')}_{model_type}_{m_size}"
        self.run_dir = os.path.join(base_path, run_name)
        
        os.makedirs(self.run_dir, exist_ok=False)
        self.exp_config["output_path"] = self.run_dir

    def initialize_model(self, model_type: str = "yolo"):
        """Initializes model. If self.exp_config exists, use it; else use defaults for inference."""
        model_map = {"yolo": YOLOModel, "vit": ViTModel, "sam": SAMModel}
        
        # If we are inferencing an existing run, we need a skeleton config
        if not self.exp_config:
            self.exp_config = {
                "output_path": self.run_dir,
                "experiment": "detection", # Default assumption
                "yolo": {"number": 8, "model_size": "n"}
            }

        self.model = model_map[model_type](self.exp_config)

    def run_train_flow(self):
        self.setup_run_directory()
        self.initialize_model(self.exp_config.get("model_type"))
        self.model.train()
        print(f"Training complete: {self.run_dir}")
        self.run_inference_flow()

    def run_inference_flow(self):
        if not self.model:
            # Logic to guess model type from folder name if possible, else default yolo
            self.initialize_model("yolo")
        
        print(f"Running inference on: {INFERENCE_SOURCE}")
        self.model.predict(source=INFERENCE_SOURCE, save_dir=self.run_dir)
        print(f"Results at: {os.path.join(self.run_dir, 'inference')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--path", help="Path to existing run dir (for predict mode only)")
    args = parser.parse_args()

    if args.mode == "train":
        manager = ExperimentManager(config_path=args.config)
        manager.run_train_flow()
    else:
        if not args.path:
            print("Error: --path is required for predict mode.")
        else:
            manager = ExperimentManager(run_dir=args.path)
            manager.run_inference_flow()