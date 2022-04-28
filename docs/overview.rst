****************
Project Overview
****************

This page is a summary of the project's goals and how it is structured to meet those goals.

Development Goals
=================

Purpose of This Software
------------------------

This package is a collection of tools for UCI's competition team at the `AUVSI SUAS <https://www.auvsi-suas.org/>`_. The AUVSI SUAS is a student competition in which an Autonomous Aerial System navigates through waypoints, avoids other vehicles and static obstacles, identifies and submits objects on the ground, and performs mapping tasks.

This :py:package:`uavfpy` contains python modules for:

    * Autonomous Navigation (:py:mod:`uavfpy.planner`)
    * Object Detection, Classification, and Localization (:py:mod:`uavfpy.odcl`)
    * Interoperability with the AUVSI SUAS 

This package is intended to be deployed both on the vehicle and on the ground station. To orchestrate the mission and manage communications between the vehicle and the ground, we use `ROS Noetic <http://wiki.ros.org/noetic>`_. 

For more about how we manage simultaneous development of the package and the ROS components, see :ref:`software-architecture`. 


Development Ethos
-----------------

Because this software has some unique deployment characteristics, development workflow is a little bit unusual when compared to typical software. We have developed the architecture with the following goals:

* *Make it easy for new developers to work on the software*. Because we have a lot of novice programmers making contributions, we want to make it as easy as possible to get started. The entire pipeline has weird dependencies (for training, there's CUDA stuff; for inference, there's TPU stuff; for system integration, there's ROS... and so on). So we want to divide the software as much as possible so that small parts can be developed and tested independently, but the whole can still be integrated.

* *Make it easy to test* It's difficult to test a computer vision system on an aerial vehicle -- can't debug while flying in the air!

* *Make it easy to deploy* When we are in the field, we are probably have the sun glaring on our computer screens. We'd like to avoid dealing with dependencies and complex build/integration steps in the field. So we aim for a 0-to-functional deployment in just a couple of minutes.

* *Make it easy to manage development* We want to keep development and documentation of the software as simple as possible, so that new issues can be identified, assigned, and completed with minimal friction.

With these goals in mind, we can talk about how the software is structured.


.. _software-architecture:

Software Architecture
=====================

Why Do We Have to Do This?
--------------------------

Running Python programs with ROS requires making them available to system python. There are several ways to do this:

    1. Configure python files in the catkin workspace to be available to system python.

    2. Install python packages into the system python.

For dependencies (such as ``numpy`` or ``scipy``), developers usually opt for (2). For simple ROS programs or small scripts, (1) is fine. But developing complex software the (1) strategy introduces significant development overhead: modules and submodules need to be manually marked in catkin; when they are used by ROS, they are not imported from the workspace, but rather copied to system python and imported from there.

To get around these issues with catkin, and for some other reasons, we are opting for (2): we release the python software as a standalone package, manage its deployment with ``pip`` instead of ROS, and call its APIs from within ROS program, as we would any other package, like ``numpy`` or ``scipy``.


Repository Structure
--------------------

Therefore, there are two branches in the repository:

    * ``main`` -- contains the ``uavf`` python package
    * ``ROS`` -- contains the ``rosuavf`` ROS package

The development of these two branches are kept *entirely separate*.

To install the package to system python (e.g., on board the UAV or on the Ground Station), we run:

.. code-block:: bash

    pip install git+https://github.com/uci-uav-forge/uavf.git

outside of a virtual environment. This will install the :py:package:`uavfpy` package into the system python. Then, from ``catkin_ws/src``, we run:

.. code-block:: bash

    git clone https://github.com/uci-uav-forge/uavf.git

    git checkout ROS

    catkin make

to checkout the ROS package and build it. Inside of the ROS software, we can import :py:package:`uavfpy` and use its APIs. 