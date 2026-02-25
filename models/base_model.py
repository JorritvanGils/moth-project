# src/models/base_model.py

class BaseModel:

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save_outputs(self):
        raise NotImplementedError