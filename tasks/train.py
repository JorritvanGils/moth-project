import os
import yaml
from datetime import datetime
from typing import Type

# Import your custom model wrappers
from models.yolo_model import YOLOModel
from models.cls.vit_model import ViTModel
from models.det.sam_model import SAMModel

# Constants
OUTPUT_ROOT = os.path.expanduser("~/moths/outputs")
INFERENCE_SOURCE = "/media/jorrit/ssd/phd/moths/inference_img"

class ExperimentManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.exp_type = self.config["experiment"]  # 'det' or 'cls'
        self.exp_config = self.config[self.exp_type]
        self.run_dir = None
        self.model = None

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _get_model_size(self) -> str:
        """Extracts model size based on type and experiment."""
        m_type = self.exp_config.get("model_type")
        if m_type == "yolo":
            return self.exp_config.get("yolo", {}).get("model_size", "n")
        elif m_type == "vit":
            return self.exp_config.get("vit", {}).get("model_size", "base")
        elif m_type == "sam":
            return "default"
        return "unknown"

    def setup_run_directory(self):
        """Creates a unique timestamped folder for this specific run."""
        model_type = self.exp_config.get("model_type")
        model_size = self._get_model_size()
        
        base_path = os.path.join(OUTPUT_ROOT, self.exp_type)
        os.makedirs(base_path, exist_ok=True)

        existing = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        run_number = len(existing) + 1
        date_str = datetime.now().strftime("%Y%m%d")
        
        run_name = f"{run_number:03d}_{date_str}_{model_type}_{model_size}"
        self.run_dir = os.path.join(base_path, run_name)
        
        os.makedirs(self.run_dir, exist_ok=False)
        self.exp_config["output_path"] = self.run_dir
        print(f"Initialized Experiment Directory: {self.run_dir}")

    def initialize_model(self):
        """Maps config strings to Model Classes."""
        model_type = self.exp_config.get("model_type")
        
        model_map = {
            "yolo": YOLOModel,
            "vit": ViTModel,
            "sam": SAMModel
        }

        if model_type not in model_map:
            raise ValueError(f"Model type '{model_type}' not supported in {self.exp_type}")

        self.exp_config["experiment_type"] = self.exp_type
        self.model = model_map[model_type](self.exp_config)

    def execute(self):
        """Main workflow: Setup -> Init -> Train -> Optional Inference."""
        self.setup_run_directory()
        self.initialize_model()

        # 1. Training
        print(f"Starting Training for {self.exp_type}...")
        self.model.train()
        print(f"Training complete. Results saved in {self.run_dir}")

        # 2. Optional Inference
        ans = input("\nWould you like to run inference on test images now? (y/n): ")
        if ans.lower() == 'y':
            if not os.path.exists(INFERENCE_SOURCE):
                print(f"Error: Inference source {INFERENCE_SOURCE} not found.")
                return

            print(f"Inference started on: {INFERENCE_SOURCE}")
            self.model.predict(source=INFERENCE_SOURCE, save_dir=self.run_dir)
            print(f"Inference saved to {os.path.join(self.run_dir, 'inference')}")

if __name__ == "__main__":
    # You could eventually use argparse here to pass the config path via CLI
    manager = ExperimentManager("configs/config.yaml")
    manager.execute()