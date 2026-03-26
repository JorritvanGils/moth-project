# 🦋 Moth Project Setup


This project supports **object detection** and **classification** for moth species, with an expanding set of models and datasets.

---

### 🔍 Detection

| Model | Dataset | Status |
|------|--------|--------|
| YOLO | NID Dataset | ✅ Implemented |
| Segment Anything | NID Dataset | 🚧 Planned |

---

### 🧠 Classification

| Model | Dataset | Status |
|------|--------|--------|
| YOLO | EU Moths Dataset | ✅ Implemented |
| DETR | EU Moths Dataset | 🚧 Planned |
| ViT | EU Moths Dataset | 🚧 Planned |
| TIMM models | EU Moths Dataset | 🚧 Planned |
| YOLO | NID Dataset | 🔄 Future |

---

### 🧠 Legend

- ✅ Implemented → ready to use  
- 🚧 Planned → not yet implemented  
- 🔄 Future → intended but not started  

---

### 📦 Datasets

- **NID Dataset** → currently used for detection  
- **EU Moths Dataset** → currently used for classification  

---

This project supports **two ways to get started**:

- **⚡ Automated Setup (Recommended)** → rents a GPU + installs everything via Ansible  
- **🛠 Manual Setup** → for users who already have access to a GPU

---

## 🚀 Option A — Automated GPU + Setup (Fastest)

Use this if you **don’t already have a GPU environment**.

### 1. Clone the project
```bash
git clone git@github.com:JorritvanGils/moth-project.git
cd moth-project
```

### 2. Configure Vast.ai
- Go to Vast.ai  
- Copy your API key into `.env`

### 3. Setup SSH
```bash
ssh-add -l
```

### 4. Launch GPU + auto-install
```bash
python gpu/vast.py
```

Select:
```
run install_and_train.yml
```

This will:
- Rent a GPU
- Install dependencies
- Download datasets
- Configure the project

### 5. Connect to your instance
```bash
ssh -A -p <PORT> root@<HOST> -L 8080:localhost:8080
```

### 6. Train
```bash
python -m tasks.train
```

---

## 🛠 Option B — Manual Setup (Existing GPU)

Use this if you already have a machine with a GPU.

---

### 1. Clone project
```bash
git clone git@github.com:JorritvanGils/moth-project.git moths
cd moths
mkdir -p outputs/cls outputs/det
```

---

### 2. Create Python environment
```bash
sudo apt update && sudo apt upgrade -y

sudo apt install python3.10 python3.10-venv -y

python3.10 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies
```bash
pip install -e .
```

---

### 4. Download datasets

#### Detection
```bash
cd ~
mkdir -p datasets/det && cd datasets/det
git clone git@github.com:cvjena/nid-dataset.git nid

cd ~/moths
python tasks/nid_to_yolo.py
```

#### Classification
```bash
mkdir -p ~/datasets/cls && cd ~/datasets/cls
git clone git@github.com:cvjena/eu-moths-dataset.git eu
```

---

### 5. Configure project
```bash
nano configs/config.yaml
```

---

### 6. Train
```bash
python -m tasks.train
```

Output example:
```
/root/moths/outputs/detection/003_20260318_yolo_n
```

---

### 7. Download results
```bash
scp -P <PORT> root@<HOST>:/root/moths/outputs/.../best.pt .
scp -r -P <PORT> root@<HOST>:/root/moths/outputs/... .
```

---

## 🧠 Summary

| Option | When to use | Pros |
|--------|------------|------|
| Automated | No GPU yet | Fast, zero setup |
| Manual | Already have GPU | Full control |

---

## 🔌 Notes

- Use SSH agent forwarding (`-A`) when connecting
- Port forwarding (`-L 8080:localhost:8080`) enables local access to services
- Outputs are stored in:

```
outputs/detection/
outputs/classification/
```