# openvino-examples

Setup for Ubuntu 24.04

```bash
sudo apt update

sudo apt install -y intel-opencl-icd libze-intel-gpu1 libze1 libtbb12

sudo usermod -aG render,video $USER
```

```bash
mkdir -p ~/intel_npu_drivers

cd ~/intel_npu_drivers

wget https://github.com/intel/linux-npu-driver/releases/download/v1.28.0/linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz

tar -xf linux-npu-driver-*.tar.gz

sudo dpkg -i *.deb

wget https://github.com/oneapi-src/level-zero/releases/download/v1.28.0/level-zero_1.28.0+u24.04_amd64.deb

sudo dpkg -i level-zero*.deb

sudo apt --fix-broken install -y
```
