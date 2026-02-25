# src/models/vit_model.py
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from models.base_model import BaseModel
import torch
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

VIT_PRETRAINED_MAP = {
    "base": "google/vit-base-patch16-224",
    "large": "google/vit-large-patch16-224",
    "huge": "google/vit-huge-patch32-384"
}

class ViTModel(BaseModel):
    def __init__(self, config: dict):
        """
        config should contain:
        - vit:
            - model_size: base | large | huge
            - num_labels: number of classes
        - data_path: dataset folder
        - output_path: unified run directory
        - params: training hyperparams (epochs, batch, learning_rate, etc.)
        """
        self.config = config
        vit_config = config.get("vit", {})
        model_size = vit_config.get("model_size", "base")
        num_labels = vit_config.get("num_labels", 100)

        pretrained_name = VIT_PRETRAINED_MAP.get(model_size)
        if pretrained_name is None:
            raise ValueError(f"Unknown ViT model_size: {model_size}")

        self.model = ViTForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def train(self):
        params = self.config.get("params", {})
        run_dir = self.config["output_path"]
        os.makedirs(run_dir, exist_ok=True)

        dataset_path = self.config["data_path"]
        dataset = load_dataset("imagefolder", data_dir=dataset_path)

        batch_size = params.get("batch", 32)
        epochs = params.get("epochs", 20)
        learning_rate = params.get("learning_rate", 2e-5)

        training_args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=params.get("weight_decay", 0.01),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,  # keep only last checkpoint
            push_to_hub=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else dataset["train"],
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.evaluate()

    def evaluate(self):
        # Can add logic to reload last checkpoint if needed
        pass

    def predict(self, inputs, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(**inputs)