# UCI's data generator and training pipeline for Target ID (ODCL) system

Repository for training target ID system on AUVSI SUAS targets.

This was developed by Mike Sutherland from 2020-2021 as part of UC Irvine's UAV Forge project, which is attempting to field a vehicle in the 2022 AUVSI competition.

## Installation

This repository requires tensorflow and tflite. I recommend creating a virtual environment, but you don't necessarily need to.

Just run 

```
pip install -r requirements.txt
```

to get the packages needed.

It's very common to use a GPU for training. To use tensorflow with an NVIDIA GPU, follow this guide:

https://www.tensorflow.org/install/gpu

It's common also to run training notebooks in Google Colab. To do this:

1. Open Google Colab (https://colab.research.google.com/)

2. Go to File > Open Notebook

3. Go to the "Github" tab

4. In another tab, open `train.ipynb` from this repository.

5. Copy/Paste the URL into the box that says `Enter a GitHub URL or search by organization or user`

6. The notebook will open in Google Colab.

## Training

This repository uses tflite-model-maker to create the object detectors. 

For documentation of that API, see here: https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector

Currently, only efficientdet models are supported by Model Maker. I'm not fully satisfied with the model-maker API (you cannot stop and continue training from a `.ckpt`, for example, and a lot of customizations to the model are unclear.)

Training is done with the jupyter notebook `train.ipynb` in this directory. Run

```jupyter notebook```

and navigate to it to run training.

## Datasets

You need `.tfrecord` files created by the generator to train. To get those, you can generate them (see the `data_gen` folder) or you can download pre-generated ones. If you are on UCI UAV Forge, those will be made available on the drive shortly. If you are using this repository from outside UCI, send me an email and I can send you those datasets. Unless you're on a team that is competing with us, then... I'll probably still send them :-)

## Contact

For questions, e-mail Mike at msutherl@uci.edu.
