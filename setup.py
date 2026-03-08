from setuptools import setup, find_packages

setup(
    name="moth-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # YOLO
        "ultralytics",
        "python-dotenv",
        "vastai",
        "python-dateutil>=2.7",
        # VIT:
        "transformers", # Added for ViT support
        "torch",        # Core dependency for both models
        "torchvision",
        "datasets",   
        "scikit-learn",
    ],
)