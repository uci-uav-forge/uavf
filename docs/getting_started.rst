***************
Getting Started
***************

This page is a guide on how to get started with the ``core`` software.

Prerequisites
=============

We have tested the API under linux and MacOS. Development of ``odcl`` should be possible under Windows, but is untested. 

Our release targets Python 3.8 or later.

Dependencies
------------

To get started on core development, clone the repo and check out the `core` branch.:

.. code-block:: bash

    git clone https://github.com/uci-uav-forge/odcl
    cd odcl
    git checkout core

Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

.. note:: 
    
    ``odcl`` uses the tflite runtime for inference. You can perform inference on the CPU, but this can be very slow. The vehicle uses the `Coral Edge TPU <https://www.coral.ai/docs/>`_ for on-board acceleration of inferencing.

    To use the Coral Edge TPU, you need to first install the `Edge TPU Runtime. <https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime>`_ Then, you can continue these steps.

Downloading Input Data and Models
---------------------------------

When you have installed all the python dependencies, you can run a test of the model pipeline by running the example script in ``pipeline.py`` in ``odcl`` module. The tflite models and example data are not included in this repository, so you will need first to download them. 

You can download example data by running ``example_images.sh``. You can download example models (and category labels) by running ``example_models.sh``.

From the root of the ``core`` branch:

.. code-block:: bash

    bash ./example_images.sh

This will download a couple of high-resolution example images into ``example_images`` directory.

.. code-block:: bash

    bash ./example_models.sh

This will download a pre-trained TPU and CPU compatible model into ``models`` directory.

Run The Pipeline
================

Then, you can run the script in ``pipeline.py``:

.. code-block:: bash

    python ./odcl/pipeline.py
