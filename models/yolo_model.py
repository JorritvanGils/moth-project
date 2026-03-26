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
            
            self.model.train(
                data=self.config["data_path"],
                project=run_dir,
                name="run",
                **train_params
            )

            internal_run_dir = os.path.join(run_dir, "run")
            plot_dir = os.path.join(run_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            for fname in os.listdir(internal_run_dir):
                src = os.path.join(internal_run_dir, fname)
                if fname.endswith(('.png', '.jpg', '.csv')):
                    os.rename(src, os.path.join(plot_dir, fname))
                elif fname == "weights":
                    dest_weights = os.path.join(run_dir, "weights")
                    os.makedirs(dest_weights, exist_ok=True)
                    for w in os.listdir(src):
                        os.rename(os.path.join(src, w), os.path.join(dest_weights, w))
                else:
                    os.rename(src, os.path.join(run_dir, fname))
            
            shutil.rmtree(internal_run_dir)

        def evaluate(self):
            """
            Runs validation and returns a concise dictionary of metrics.
            """
            best_path = os.path.join(self.config["output_path"], "weights", "best.pt")
            eval_model = YOLO(best_path) if os.path.exists(best_path) else self.model
            
            results = eval_model.val()

            print(f"results: {results}")

            metrics = {
                "precision": results.results_dict['metrics/precision(B)'],
                "recall": results.results_dict['metrics/recall(B)'],
                "f1": results.box.f1.mean(),
                "mAP50": results.results_dict['metrics/mAP50(B)'],
                "mAP50-95": results.results_dict['metrics/mAP50-95(B)'],
                "fitness": results.fitness
            }
            
            return metrics

    def predict(self, source, save_dir=None, conf=0.02, iou=0.3, **kwargs):
            """
            Runs inference. 
            Lower 'conf' if no bbox predicted (0.02 worked)
            """
            best_model_path = os.path.join(self.config["output_path"], "weights", "best.pt")
            print(f"Looking for weights at: {best_model_path}")
            if os.path.exists(best_model_path):
                inference_model = YOLO(best_model_path)
            else:
                inference_model = self.model

            return inference_model.predict(
                source=source,
                project=save_dir,
                name="inference",
                save=True,
                conf=conf,
                iou=iou,
                exist_ok=True,
                **kwargs
            )
