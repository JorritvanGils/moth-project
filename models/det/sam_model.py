# src/models/sam_model.py
from models.base_model import BaseModel
import os
from segment_anything import sam_model_registry, SamPredictor

# Localize objects with SAM3. 

# Dataset paper:
# - annotat image data. 
#   - 1) use SAM3 (https://github.com/facebookresearch/sam3) to get masks to select the moths
#   - 2) convert those masks into BoundingBoxes
#   - 3) use labelbox (licenced via Uni) to (experts) label the species
# - publish dataset and annotations
# - add (sub) species classification. 
# - Train some baseline latest yolo (Start with yolov8). 

class SAMModel(BaseModel):
    def __init__(self, config: dict):
        """
        config should contain:
        - sam:
        - data_path: path to images
        - output_path: unified run directory
        - params: dict for training/inference (optional)
        """
        self.config = config
        sam_config = config.get("sam", {})
        sam_model_type = sam_config.get("model_type", "vit_h")
        
        # SAM provides a registry of pretrained weights
        self.model = sam_model_registry[sam_model_type](checkpoint=None)
        self.predictor = SamPredictor(self.model)

    def train(self):
        # SAM is mostly used for zero-shot segmentation or fine-tuning
        run_dir = self.config["output_path"]
        os.makedirs(run_dir, exist_ok=True)
        print(f"SAM run folder: {run_dir}")
        # Optional: implement fine-tuning logic here
        pass

    def evaluate(self):
        # Implement evaluation logic if needed
        pass

    def predict(self, image, **kwargs):
        """
        Predict segmentation masks for a single image or batch.
        Returns masks and scores.
        """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(**kwargs)
        return {"masks": masks, "scores": scores, "logits": logits}