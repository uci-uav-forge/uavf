***************
Getting Started
***************

This page is a guide on how get started with the ``uavf`` API.

Prerequisites
=============

We have tested the API under linux and MacOS. Development of ``uavf`` is possible under Windows also.

Our release targets Python 3.8.

.. note:: 
    
    ``odcl`` uses the tflite runtime for inference. You can perform inference on the CPU, but this can be very slow. The vehicle uses the `Coral Edge TPU <https://www.coral.ai/docs/>`_ for on-board acceleration of inferencing.

    The Coral Edge TPU is an ASIC developed by Google specifically designed for accelerating deep learning. If you do not have access to an Edge TPU, you can use the CPU for inference. 

    To use the Coral Edge TPU, you need to first install the `Edge TPU Runtime. <https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime>`_ Then, you can continue these steps.

Installation
------------

Install via ``pip``.

.. code-block:: bash

    pip install git+https://github.com/uci-uav-forge/uavf.git

Object Detection, Classification, and Localization (ODCL)
=========================================================

Downloading Input Data and Models
---------------------------------

The tflite models and example data are not included in this repository, so you will need first to download them. You can download example data by running the ``example_models.sh`` script.

From the root directory:

.. code-block:: bash

    bash ./example_models.sh

This will make a directory called "examples" and download a couple of high-resolution example images and pre-trained TPU and CPU models into it.

Once we have the example data, we are ready to create a pipeline and run inference.

First, we import necessary modules:

.. code-block:: python

    # import classes
    from odcl.inference import TargetInterpreter, Tiler
    from odcl.utils.drawer import TargetDrawer
    from odcl.color import Color
    from odcl.pipeline import Pipeline
    import logging, cv2

Then, we set paths to the example data and the models we downloaded. We also want to display logs.

.. code-block:: python

    # set model directory
    MODEL_PATH = "./example/efficientdet_lite0_320_ptq.tflite"
    LABEL_PATH = "./example/coco_labels.txt"
    IMG_PATH = "./example/plaza.jpg"

    # set logger to print info
    logging.basicConfig(
        format="%(levelname)s:%(processName)s@%(module)s\t%(message)s", level=logging.INFO
    )

The :py:class:`odcl.inference.TargetInterpreter` class handles inputs and outputs to the neural network for object detection. We give it paths to the model and labels, tell it whether to run on CPU or TPU, and set the threshold for detection. 

Instantiating a :py:class:`odcl.inference.TargetInterpreter` object takes a while, so this object should be created outside of a loop if latency is at issue. 

.. code-block:: python

    # create the interpreter
    interpreter = TargetInterpreter(
        MODEL_PATH,
        LABEL_PATH,
        "cpu",
        thresh=0.4,
        order_key="efficientdetd0",
    )

Next, we create the :py:class:`odcl.inference.Tiler`, which handles the tiling of the input image. We are dealing with inputs that are very large compared to the inputs of the neural network; the tiler will decompose the image into overlapping tiles, feed the NN, and then parse NN outputs from the respective tiles back into the raw image.

:py:class:`odcl.color.Color` is a class used to extract color information from found targets. For now, it does not take any arguments. 

:py:class:`odcl.utils.drawer.TargetDrawer` is a utility class used to draw bounding boxes. Passing it as an argument will draw bounding boxes on the raw image and store the result into the :py:class:`Pipeline`'s :py:attr:`drawn` attribute. Passing it will also open a window to display targets that were found, along with the shape color-mask. Therefore, it is useful for evaluating the performance of the pipeline in real time.

If a :py:class:`TargetDrawer` is not passed to the :py:class:`odcl.pipeline.Pipeline` constructor, the :py:class:`Pipeline` will not draw bounding boxes on the image, nor will found targets be displayed.

.. code-block:: python

    # create the tiler
    tiler = Tiler(320, 50)

    # create a drawer
    drawer = TargetDrawer(interpreter.labels)

    # color
    color = Color()

    # create the pipeline object
    pipeline = Pipeline(interpreter, tiler, color, drawer)

The :py:meth:`odcl.pipeline.Pipeline.run` method takes an image and returns a list of found targets.

.. code-block:: python

    # parse the raw image
    image_raw = cv2.imread(IMG_PATH)

    # run the pipeline
    pipeline.run(image_raw, None)

The full script in this example is shown below:

.. code-block:: python

    # import classes
    from odcl.inference import TargetInterpreter, Tiler
    from odcl.utils.drawer import TargetDrawer
    from odcl.color import Color
    from odcl.pipeline import Pipeline
    import logging, cv2

    # set model directory
    MODEL_PATH = "../example/efficientdet_lite0_320_ptq.tflite"
    LABEL_PATH = "../example/coco_labels.txt"
    IMG_PATH = "../example/plaza.jpg"

    # set logger to print info
    logging.basicConfig(
        format="%(levelname)s:%(processName)s@%(module)s\t%(message)s", level=logging.INFO
    )

    # create the interpreter
    interpreter = TargetInterpreter(
        MODEL_PATH,
        LABEL_PATH,
        "cpu",
        thresh=0.4,
        order_key="efficientdetd0",
    )

    # create the tiler
    tiler = Tiler(320, 50)

    # create a drawer
    drawer = TargetDrawer(interpreter.labels)

    # color
    color = Color()

    # create the pipeline object
    pipeline = Pipeline(interpreter, tiler, color, drawer)

    # parse the raw image
    image_raw = cv2.imread(IMG_PATH)

    # run the pipeline
    pipeline.run(image_raw, None)
