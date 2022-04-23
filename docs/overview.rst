****************
Project Overview
****************

This page is a summary of the project's goals and how it is structured to meet those goals.

Development Goals
=================

Purpose of This Software
------------------------

This is the documentation for UCI's ODCL (Object Detection, Classification, and Localization) pipeline. This software is intended to be used on-board a UAV to find and recognize "targets" on the ground. Targets are flat, colored shapes (hexagon, triangle, square, etc.) with visible alphanumeric characters pasted on them (ABCD... 789). The pipeline takes a single raw image from the on-board camera and produces a list (possibly empty!) containing information about the targets that are visible in the raw image. To recognize targets, it uses a custom deep-learning based object detector and a host of other computer vision and computational geometry algorithms to find and extract target information from the raw image.

Development Ethos
-----------------

Because this software has some unique deployment characteristics, development workflow is a little bit unusual when compared to typical software. We have developed the architecture with the following goals:

* *Make it easy for new developers to work on the software*. Because we have a lot of novice programmers making contributions, we want to make it as easy as possible to get started. The entire pipeline has weird dependencies (for training, there's CUDA stuff; for inference, there's TPU stuff; for system integration, there's ROS... and so on). So we want to divide the software as much as possible so that small parts can be developed and tested independently, but the whole can still be integrated.

* *Make it easy to test* It's difficult to test a computer vision system on an aerial vehicle -- can't debug while flying in the air!

* *Make it easy to deploy* When we are in the field, we are probably have the sun glaring on our computer screens. We'd like to avoid dealing with dependencies and complex build/integration steps in the field. So we aim for a 0-to-functional deployment in just a couple of minutes.

* *Make it easy to manage development* We want to keep development and documentation of the software as simple as possible, so that new issues can be identified, assigned, and completed with minimal friction.

With these goals in mind, we can talk about how the software is structured.

Software Architecture
=====================

You may have noticed that the `main` branch is completely empty. This is intentional!

There are two main branches:

* ``Core``

* ``ROS``

The main distinguishing factor between the two branches are the dependencies and integration/testing requirements. 