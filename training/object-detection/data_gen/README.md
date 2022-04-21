# AUVSI Data Gen
Every year, AUVSI SUAS sponsors a competition to build and fly an unmanned aerial vehicle. To win the competition, students have to create a system that can recognize targets on the ground from the air. This problem naturally lends itself to neural networks which can perform object detection, localization, and classification (ODCL). These networks require training data in the form of images and bounding box or segmentation annotations.

This script is designed to generate synthetic training data for such a network.

## How to install 

Install blender:

```
sudo apt-get install blender
```

Make sure that the blender executable is accessible from the terminal. Then, install bpy. You can try

```
pip install bpy
```
But that did not work for me. If it doesn't, you can install from a precompiled wheel, from https://github.com/TylerGubala/blenderpy/releases. Follow the instructions for your platform.

You also need opencv to run the script.

## Running
For example:

```blender --background -P 'datagen.py' -- --dir ./sample --bg_dir ~/Pictures/wallpaper/ --shape_dir target_objs/shapes/ --alpha_dir target_objs/alphanumerics/ --n 100```

This will generate a dataset containing 100 images, using background images from my wallpaper folder (in `~/Pictures/wallpaper/`), using blender `.objs` from the directory `target_objs`, containing shapes in `target_objcs/shapes` and alphanumeric characters in `target_objs/alphanumerics`.

There are a wide variety of options for camera placement, no. of objects, etc. in the script; pass `-h` after `--` to see a description of them.