# src/models/yolo_model.py

from ultralytics import YOLO
# from models.base_model import BaseModel 
from models import BaseModel 

class YOLOModel(BaseModel):

    def __init__(self, config):
        self.config = config
        self.model = YOLO(config["pretrained"])

    def train(self):
        self.model.train(
            data=self.config["data_path"],
            project=self.config["output_path"],
            name=self.config["run_name"]
        )