# Step 0) Optionally: Rent GPU
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
git clone git@github.com:cvjena/eu-moths-dataset.git

# Step 5) Yolo inference


python -m tasks.train


