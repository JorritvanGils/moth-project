# Moth Project Setup

## 0. (Optional) Rent a GPU

```bash
git clone git@github.com:JorritvanGils/moth-project.git
```

1. Go to Vast.ai  
2. Copy your account API key into `.env` at the project root  
3. Run:

```bash 
# TODO FIX THAT IT AUTO SSHS INTO IT. WNOW IT SAID SSH WASNT AVAILABLE WHILE IT WORKED WITH THE SECOND LINK
python gpu/vast.py
ssh-add -l
# use Proxy ssh connect and add -A, like:
ssh -A -p 37018 root@ssh7.vast.ai -L 8080:localhost:8080
```

Rent a GPU and SSH into it.

---

## 1. Project Setup

```bash
git clone git@github.com:JorritvanGils/moth-project.git moths
cd moths
mkdir -p outputs/cls outputs/det # should be handled by the script



```

---

## 2. Create Python Environment

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install python3-venv -y
sudo apt install software-properties-common -y

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-venv -y

python3.10 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
# from project root run:
pip install -e .
```

---

## 4. Download Datasets

### Detection

```bash
cd ~ # nav out of 'moths' project
mkdir -p datasets/det && cd datasets/det
git clone git@github.com:cvjena/nid-dataset.git nid

cd ~/moths
python tasks/nid_to_yolo.py
```

### Classification

```bash
mkdir -p ~/datasets/cls && cd ~/datasets/cls
git clone git@github.com:cvjena/eu-moths-dataset.git eu
```

---

## 5. Configure your config.yml
```bash
nano configs/config.yaml
# det/cls
# model_type
# params
```

## 6. Train

```bash
python -m tasks.train
# /root/moths/outputs/detection/003_20260318_yolo_n
```

## 7. Store output on local machine

```bash
cd /media/jorrit/ssd/phd/outputs && scp -P 37018 root@ssh7.vast.ai:/root/moths/outputs/detection/003_20260318_yolo_n/weights/best.pt .
cd /media/jorrit/ssd/phd/outputs && scp -r -P 37018 root@ssh7.vast.ai:/root/moths/outputs/detection/003_20260318_yolo_n .
```

ssh -A -p 37018 root@ssh7.vast.ai -L 8080:localhost:8080