<!-- # Step 0) Optionally: Rent GPU
git clone git@github.com:JorritvanGils/moth-project.git
Go to vast.ai and copy the accounts api key into .env at project root
python gpu/vast.py
rent gpu & ssh into

# Step 1)
git clone git@github.com:JorritvanGils/moth-project.git
cd moth-project
mkdir data/classification data/detection
mkdir outputs/classification outputs/detection

# Step 2) create venv
sudo apt update && sudo apt upgrade -y
apt install python3-venv -y && \
apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt install python3.10 python3.10-venv -y && \
python3.10 -m venv .venv
source .venv/bin/activate

# Step 3) Install dependencies
cd /media/jorrit/ssd/phd/moth-project
pip install -e .
# pip install -r requirements.txt
# pip install ultralytics

# Step 4) download datasets:
cd data/detection
git clone git@github.com:cvjena/nid-dataset.git nid
# python scripts/nid_to_yolo.py

cd ../../classification
git clone git@github.com:cvjena/eu-moths-dataset.git eu

# Step 5) Yolo inference


python -m tasks.train

 -->


# Moth Project

This repository contains the pipeline for moth detection and classification using YOLO.

## 🚀 Getting Started

### Step 0: GPU Provisioning (Optional)
If you require external compute, follow these steps to rent a GPU via Vast.ai:

1. Clone the repository:
   git clone git@github.com:JorritvanGils/moth-project.git
   cd moth-project
2. Setup API Key: Add your Vast.ai API key into a .env file at the project root.
3. Provision the instance:
   python gpu/vast.py
4. Access: SSH into the rented instance to continue.

---

## 🛠️ Installation & Setup

### 1. Project Structure
Clone the repository and initialize the necessary directories for data and outputs:

git clone git@github.com:JorritvanGils/moth-project.git
cd moth-project
mkdir -p data/classification data/detection
mkdir -p outputs/classification outputs/detection

### 2. Environment Setup
Update the system and install Python 3.10:

sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-venv -y

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

### 3. Install Dependencies
Install the project in editable mode to ensure all modules are accessible:

pip install -e .
# Optional: pip install -r requirements.txt
# Optional: pip install ultralytics

---

## 📊 Data Preparation

### Detection Dataset
Download the NID Dataset for object detection:

cd data/detection
git clone git@github.com:cvjena/nid-dataset.git nid
# python scripts/nid_to_yolo.py
cd ../..

### Classification Dataset
Download the EU-Moths Dataset for classification:

cd data/classification
git clone git@github.com:cvjena/eu-moths-dataset.git eu
cd ../..

---

## 🚂 Training & Inference

To initiate the training pipeline, run the following command:

python -m tasks.train