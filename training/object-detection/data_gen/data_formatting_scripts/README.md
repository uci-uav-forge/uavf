# Data Formatting Scripts

By default, the data generator outputs the training images into an `/images` folder, and the annotations into a `/labels` folder. There are a few scripts in this folder that help to re-format the data into something that is usable by the training pipeline.

NOTE: it may be useful to have an SSD (not spinning metal drive) while working with large datasets, because they are large datasets so io is the big constraint with these scripts.

## Example usage

We assume that we start from the output of the data generation script, in a folder `sample` outputted to this directory.

First, we make the YOLO formatted train/test split from `sample` and copy it to `sample_tts`:

```
python trainval_yolo.py -i sample -o sample_tts -t 0.1
```

Here, we make 10% of the images validation images; they are stored in the `test` subfolder, while the training images are stored in `train` folder.

Next, we take the split YOLO set and preprocess it so that the tfrecord generator script can read it:

```
python preprocess.py -i sample_tts/ -o sample_pre
```

This will copy from `sample_tts` into a new folder, `sample_pre`. Rather than having separate `labels` folders with `.txt` files in it, we now have two files, `testlist.txt` and `traininglist.txt` that contain images and training labels from the `images` and `labels` folder that contains all of our data.

---

### Pruning

Sometimes, we may want to generate a smaller dataset than the original, e.g. for prototyping purposes. We can run `prune.py` to produce such a set. First, do the above steps, until we have a preprocessed dataset.

We call `prune.py` to cut out every `n` items in the dataset. For example, calling prune with `n=2` will produce a dataset that is 50% of the size of the original:

```
python prune.py -i sample_pre/ -o sample_tts_pruned/ -n 2
```

---

Finally, once we have our `sample_pre` folder, we can make the `.tfrecord` files. We need to do this twice for one dataset, once for the train set and once for the test set.

This script requires a `classlabels.txt` file, containing the label of each class on a single line. The default `classlabels.txt` corresponding to the targets in the data generation script in the parent folder can be found in this folder.

For the train set:

```
python to_tfrec.py -t ./sample_pre/traininglist.txt -o ./sample_tfrecs/sample_train.tfrecord -c ./classlabels.txt -l ./sample_tfrecs/label_map.pbtxt
```

For the test set:

```
python to_tfrec.py -t ./sample_pre/testlist.txt -o ./sample_tfrecs/sample_test.tfrecord -c ./classlabels.txt -l ./sample_tfrecs/label_map.pbtxt
```

Now, your `.tfrecord` files are in the `sample_tfrecrs/` directory.

## Notes:

Pass `-h` argument to any script to see a description of each argument.

If you don't have enough disk space to copy datasets, you can change the `shutil.copy` directives to `shutil.move` directives within the scripts to save a copy step.

Intermediate steps may prove useful for loading into other object detectors, like `yolov5` or something. 