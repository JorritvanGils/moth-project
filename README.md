<!-- # Moth Project Setup

## 0. (Optional) Rent a GPU and automate installation Ansible

```bash
git clone git@github.com:JorritvanGils/moth-project.git
```

1. Go to Vast.ai  
2. Copy your account API key into `.env` at the project root  
3. set-up ssh key and verify with 
```bash
ssh-add -l
```
4. Run:
```bash 
python gpu/vast.py



For automated Ansible installation (doing steps 1 - 7 below) choose:
run setup_moth_project.yml playbook

For manual installation run the commands below: 

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

# mkdir -p ~/.venvs
# python3.10 -m venv ~/.venvs/.moth
# source ~/.venvs/.moth/bin/activate
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

ssh -A -p 37018 root@ssh7.vast.ai -L 8080:localhost:8080 -->

# 🦋 Moth Project Setup

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
run setup_moth_project.yml
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