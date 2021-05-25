# odcl
UAV Forge's ODCL repository.

This repository contains all code necessary for recognizing AUVSI targets.

### Installation / Setup
---
Requires: 
+ python 3.8
+ opencv-python
+ CUDA libs for GPU support
---
1. Clone this repo: 
```
git clone https://github.com/uci-uav-forge/odcl.git
``` 

2. Create & activate virtual env: 
```
python3 -m venv venv
```

3. Activate virtual env
+ in *nix: 
```
. venv/bin/activate
```
+ in Windows: 
```
source ./venv/Scripts/activate
```

4. Install required packages: 
```
pip3 install -r ./requirements.txt
```

5. Install `yolov5` into the same directory as this `README.md`:
```
git clone https://github.com/ultralytics/yolov5.git
```
---

