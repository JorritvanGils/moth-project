# Moth Project Setup

## 0. (Optional) Rent a GPU

```bash
git clone git@github.com:JorritvanGils/moth-project.git
```

1. Go to Vast.ai  
2. Copy your account API key into `.env` at the project root  
3. Run:

```bash
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
cd /media/jorrit/ssd/phd/moths

pip install -e .
# pip install -r requirements.txt
# pip install ultralytics
```

---

## 4. Download Datasets

### Detection

```bash
cd moths & cd .. # go out of the moths project
mkdir -p datasets/cls datasets/det

cd datasets/det
git clone git@github.com:cvjena/nid-dataset.git nid

# Optional conversion
# python scripts/nid_to_yolo.py
```

### Classification

```bash
cd ../../cls
git clone git@github.com:cvjena/eu-moths-dataset.git eu
```

---

## 5. Train YOLO

```bash
python -m tasks.train
```
