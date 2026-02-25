# src/models/yolo_model.py
from ultralytics import YOLO
from models.base_model import BaseModel
import os

class YOLOModel(BaseModel):
    def __init__(self, config: dict):
        """
        config should contain:
        - experiment: "detection" or "classification"
        - yolo:
            - number: YOLO version (8)
            - model_size: "nano", "small", "medium", "large", "xlarge"
        - data_path: path to yaml or dataset folder
        - output_path: unified run directory
        - run_name: optional, used internally by YOLO
        - params: dict of training hyperparams
        """
        self.config = config
        self.model = YOLO(self._get_pretrained_path())

    def _get_pretrained_path(self) -> str:
        number = self.config.get("yolo", {}).get("number", 8)
        model_size = self.config.get("yolo", {}).get("model_size", "nano")
        suffix = "-cls.pt" if self.config.get("experiment") == "classification" else ".pt"
        return f"yolov{number}{model_size}{suffix}"

    def train(self):
        train_params = self.config.get("params", {})

        # Ensure all outputs go into the unified run folder
        run_dir = self.config["output_path"]
        os.makedirs(run_dir, exist_ok=True)

        # YOLO will write results inside this folder
        self.model.train(
            data=self.config["data_path"],
            project=run_dir,
            name="run",  # YOLO will create run/ folder inside project
            **train_params
        )

        # Move everything from YOLO's internal run/ folder into run_dir
        internal_run_dir = os.path.join(run_dir, "run")
        for fname in os.listdir(internal_run_dir):
            os.rename(
                os.path.join(internal_run_dir, fname),
                os.path.join(run_dir, fname)
            )
        os.rmdir(internal_run_dir)  # clean up

    def evaluate(self):
        return self.model.val()

    def predict(self, source, **kwargs):
        return self.model.predict(source, **kwargs)