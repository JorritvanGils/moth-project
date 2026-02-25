# src/models/vit_model.py

from transformers import ViTForImageClassification
from models import BaseModel

class ViTModel(BaseModel):

    def __init__(self, config):
        self.config = config
        self.model = ViTForImageClassification.from_pretrained(
            config["pretrained"],
            num_labels=config["num_labels"]
        )

    def train(self):
        # Trainer logic here
        pass