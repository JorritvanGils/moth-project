Step 0)
git clone git@github.com:JorritvanGils/moth-project.git
cd moth-project
mkdir data/classification data/detection

Step 1) create venv
sudo apt update && sudo apt upgrade -y
apt install python3-venv -y && \
apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt install python3.10 python3.10-venv -y && \
python3.10 -m venv .venv
source .venv/bin/activate


cd /media/jorrit/ssd/phd/moth-project
pip install -e .
# pip install -r requirements.txt

# pip install ultralytics

Step 2) download datasets:
cd data/detection
git clone git@github.com:cvjena/nid-dataset.git nid
# python scripts/nid_to_yolo.py

cd ../../classification
git clone git@github.com:cvjena/eu-moths-dataset.git


