# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: nntrain
#     language: python
#     name: nntrain
# ---

GOOGLE_COLAB = False
if GOOGLE_COLAB:
    # !pip install -q tflite-model-maker 
    # !pip install -q pycocotools 
    # !pip install -q tensorflow 

# Imports. The output of `tf.__version__` should read something like 2.#

import os
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import pycocotools
import tensorflow as tf
from absl import flags
from pathlib import Path
print(tf.__version__)
import logging
logging.basicConfig(level=logging.INFO)
from tensorflow.python.client import device_lib
for d in device_lib.list_local_devices():
    print(d)
    print('---')

# Set directories.

# +
DATA_DIR = Path("./data/")
TRAIN_FILE = DATA_DIR / "10k_train.tfrecord"
TEST_FILE = DATA_DIR / "10k_test.tfrecord"
CLASS_LABELS_FILE = DATA_DIR / "classlabels.txt"
CKPT_DIR = Path("./checkpoints") / "ckpts01"
EXPORT_DIR = Path("./export/") / "01"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

print("Reading data from ", DATA_DIR.resolve())
print("Saving Checkpoints to ", CKPT_DIR.resolve())
print("Exporting to ", EXPORT_DIR.resolve())
# -

# Read class labels.

# +
with open(CLASS_LABELS_FILE, "r") as f:
    label_map = [s.strip() for s in f.readlines()]
    
label_map_dict = {}
for i, l in enumerate(label_map):
    label_map_dict[i+1] = l

for k, v in label_map_dict.items():
    print(k, ": ", v)

# +
train_data = object_detector.DataLoader(
    tfrecord_file_patten = str(TRAIN_FILE.resolve()),
    size = 9000,
    label_map = label_map_dict
)

test_data = object_detector.DataLoader(
    tfrecord_file_patten = str(TEST_FILE.resolve()),
    size = 1000,
    label_map = label_map_dict
)
# -

# Create spec with hyperparameters.

hparams = {
    "map_freq" : 2,
    "learning_rate" : 0.2,
}
spec = object_detector.EfficientDetLite1Spec(
    hparams=hparams, model_dir=CKPT_DIR, epochs=2, batch_size=64,
    var_freeze_expr='(efficientnet)',
    tflite_max_detections = 10,
    verbose = 1,
)

# Train model

model = object_detector.create(
    train_data = train_data,
    model_spec = spec,
    validation_data = test_data,
    train_whole_model = False,
    do_train = True,
)
model.export(
    export_dir = EXPORT_DIR,
    export_format = [ExportFormat.TFLITE, ExportFormat.LABEL, ExportFormat.SAVED_MODEL]
)


