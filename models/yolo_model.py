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

        if self.config.get("experiment") == "classification":
            suffix = "-cls.pt" 
        else:
            suffix = ".pt"
            
        return f"yolov{number}{model_size}{suffix}"

    def train(self):
        train_params = self.config.get("params", {})

        run_dir = self.config["output_path"]
        os.makedirs(run_dir, exist_ok=True)

        self.model.train(
            data=self.config["data_path"],
            project=run_dir,
            name="run",
            **train_params
        )

        internal_run_dir = os.path.join(run_dir, "run")
        for fname in os.listdir(internal_run_dir):
            os.rename(
                os.path.join(internal_run_dir, fname),
                os.path.join(run_dir, fname)
            )
        os.rmdir(internal_run_dir)  # clean up

    def evaluate(self):
        # How can this be used? 
        # Should we in tasks/train.py then also later run method .evaluate()
        # do we then get access to all evaluation data stored in variables?
        # can we get these metrics programatically?
        # accuracy, 
        # precision, 
        # recall, 
        # F1-score, 
        # Receiver Operating Characteristic (ROC) curve
        # confusion matrix    
        # And then later for ViT also get similar results. 
        return self.model.val()

    def predict(self, source, save_dir=None, **kwargs):
            # After training, the best model is usually at 'weights/best.pt' 
            # inside the output_path we defined.
            best_model_path = os.path.join(self.config["output_path"], "weights", "best.pt")
            
            # Load the trained weights if they exist, otherwise use the current state
            if os.path.exists(best_model_path):
                inference_model = YOLO(best_model_path)
            else:
                inference_model = self.model

            return inference_model.predict(
                source=source,
                project=save_dir, # This ensures it saves to your specific run folder
                name="inference",
                save=True,        # Saves the annotated image
                **kwargs
            )
