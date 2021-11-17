# odcl
UAV Forge's ODCL repository.

This repository contains all code necessary for recognizing AUVSI targets.

## Documentation
Documentation is available at: 

https://uci-uav-forge.github.io/odcl/

## Release History

v1.0 - Oct 4, 2021

+ Add inference on COCO targets with mobilenet SSD
+ Support for TPU
+ Basic display with openCV

## Hardware Requirements

+ Google Coral TPU

+ opencv

+ Your device must have a camera 

Tested with Python 3.7.10

## How to run the example

1. Clone this repository and install requirements

2. Install TPU libraries:

https://coral.ai/docs/accelerator/get-started/#requirements

3. Download mobilenet SSD model into this folder:

```
wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
```

4. Plug in your TPU

5. run pipeline.py
```
python odcl/pipeline.py
```
