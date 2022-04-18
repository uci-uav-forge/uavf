# odcl
UAV Forge's ODCL repository.

This repository contains all code necessary for recognizing AUVSI targets.

⚠️ The core odcl code is in the [`core`](https://github.com/uci-uav-forge/odcl/tree/core) branch!

⚠️ The ROS code is in the `ros` branch!

## Documentation
Documentation is automatically generated & hosted when commits are made to `core`. It can be found here: https://uci-uav-forge.github.io/odcl/

## Developer Guide

Recognizing targets and submitting them to interop takes place on multiple devices and over radio using ROS. To simplify development, we have split the repository into two main branches: `core` and `ros`. These branches are separately maintained to keep the ROS-based components necessary for communication and orchestration separate from the core functionality of the image pipeline.

This allows developers who don't have a ROS environment handy to develop the main functionality of the imaging system, and it also maintains a clear separation of core model code from ROS code.

## Installation

See "Installation" here: https://uci-uav-forge.github.io/odcl/
