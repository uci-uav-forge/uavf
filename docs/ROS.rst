********************
The UAVF ROS Package
********************

The ``uavf`` package in ``main`` is a python package that does not depend on ROS. We do use ROS to orchestrate the mission, so we have a ROS package as well.

``uavf`` is the name of the ROS package. Its development shares an issue tracker and repository with the main python package, but its development happens on the ``ROS`` and ``ROS-dev`` branches of the repository.

Installation
============

Prerequisites
`````````````
In order to develop packages on ROS, you need a PC equipped with Linux. Any desktop linux platform is suitable, but the easiest by far is Ubuntu. I prefer Ubuntu MATE on the desktop, but you can use a standard Ubuntu, KDE, or whichever flavor you like.

``uavf`` is a ROS package. To install it, you need to have ROS installed and configured. That will not be covered in this documentation; if you are brand new to ROS, I recommend that you go through the ROS tutorial [1]_ before continuing to the next section.

This release targets ROS 1 Noetic, for compatibility reasons. We are not developing for ROS 2.

We assume that you have set up a catkin workspace in your home directory:

.. code-block::

    ~/catkin_ws

Installation
````````````

.. warning::

    Because we are using this package from ROS, we need to ensure that we are NOT in any python virtual environment. You can verify this by typing ``which python`` into a terminal window. Make sure that the output is ``/usr/bin/python``.

Set up ROS:

.. code-block:: bash 

    source /opt/ros/noetic/setup.bash

Clone the git repository into ``~/catkin_ws/src`` and checkout the ``ROS`` branch. If you are developing ROS functionality, checkout ``ROS-dev``.

.. code-block:: bash

    cd ~/catkin_ws/src
    git clone https://github.com/uci-uav-forge/uavf
    git checkout ROS

Install the ``auvsi_suas`` interop client package to your system python. There is a script that will do this.

.. code-block:: bash

    cd ~/catkin_ws/src/uavf/
    bash ./install_auvsi_client.sh

.. note::

    If, for some reason, you need to uninstall auvsi_suas from your system python, you can do so by running ``pip uninstall auvsi_suas``.

Run ``catkin_make`` and source your ``devel/setup.bash`` file:

.. code-block:: bash

    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash

Start a ``roscore`` instance in a separate terminal window.

The interop client is a ros node written in Python. We start it with ``rosrun``.

.. code-block::

    rosrun uavf interop.py

.. [1] http://wiki.ros.org/ROS/Tutorials