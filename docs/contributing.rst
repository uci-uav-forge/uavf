************
Contributing
************

This page is a guide to making contributions to the codebase.


Core Developer Workflow
=======================

Useful to Know
--------------

The core deveopment workflow presumes some knowledge of ``git``. If you are brand new to ``git``, you may have these questions:

* *What is git? Branches, merging, commits? How do I install git?* → `Read the git-scm guide <https://git-scm.com/book/en/v2>`_

* *What is GitHub? What are issues and pull requests?* → `Read the github flow guide <https://docs.github.com/en/get-started/quickstart/github-flow>`_


Our Workflow
------------

1. An issue, bug, or new feature is identified on the project's `issue tracker <https://github.com/uci-uav-forge/odcl/issues>`_.

2. You create a new branch from ``core-dev``.

3. You implement and test your changes in that branch.

4. When you are satisfied with your changes: You make a new pull request with *base* ``core-dev`` and *compare* ``feature-branch``

    * Go to https://github.com/uci-uav-forge/odcl/pulls

    * Click the green "New Pull Request" button

    * choose ``core-dev`` as the base, and your branch as the compare

    * You can do this in the main codebase, or in your own fork of the project.

5. When you have created a new PR that fixes some issue in the issue tracker, you can link to the PR in the issue thread. 

.. note:: 
    
    Try to keep as much discussion as possible on the issue tracker so that the development process is well documented. Good things to include are:

    * A summary of the changes

    * Rationale for the changes

    * What other parts of the codebase do the changes impact?

.. note::

    In general, you want to test your changes to make sure they work before merging them into ``core-dev.`` Unit tests are ideal! Remember that changes to parts of the codebase may break other parts.


Try to keep new features and bug fixes contained rationally in a single branch.

What should go into the repository?
===================================

* Code (``.py``)
* Scripts (``.py``, ``.sh``)
* documentation (``.rst``, ``.md``)

.. warning::

    What should *not* go into the repository?

    * Models (``.tflite``, ``.pb``, etc.)
    * Data (``.tfrecord``, ``.jpg``, etc.)
    * Example Files (``.jpg``, ``.mp4``, etc.)
    * Scripts that don't work
    * Failed Experiments

Rather than dumping non-code files into the repository, instruct the user how to acquire them. A good way to do this is to put the file in a publically-accessible link and include a script to download it. For example, I can upload a model to ``https://drive.google.com/some-public-link``, and then make a script called ``download-model.sh`` that has 

.. code:: bash 

    wget https://drive.google.com/some-public-link

Somewhere inside. 

We want to do this for any file that isn't code, like binary files, images, models, and so on. This ensures that the repository is as clean and readable as possible, which helps new maintaners understand the codebase.

Software Structure
==================

The software is organized into two main components:

* ``core``, which is a python package that contains all functionality of the imaging pipeline
* ``ROS``, which is a ROS package that has the ROS functionality. ``core`` is a dependency of ``ROS``.


ROS
---

Working with ROS has some requirements that make it somewhat more difficult to work with than you may be used to. In particular, using ROS *requires* a Linux operating system. We have a few laptops with linux installed already, but we expect that most people aren't using linux natively and may not be used to working with the OS. So rather than making the entire piece of software dependent on linux to even run, we have separated the ROS-dependent code and the standard `core` code.

We also keep ROS code separate because `it's good practice to do so anyway <http://www.artificialhumancompanions.com/structure-python-based-ros-package/>`_. This is for several reasons, but it mostly has to do with how ROS is integrated with Python. In a nutshell, ROS always needs to use the system Python; even though standard Python development usually uses virtual environments to manage dependencies:

.. image:: https://imgs.xkcd.com/comics/python_environment_2x.png
    :width: 60%
    :align: center

So, to avoid development hell, we put the bulk of the functionality into the ``core`` branch, install ``core`` (and all of its dependencies) onto the vehicle's system python, and then we can just import the core package and use its funcationality in our ROS scripts.

The Golden Rule of ROS Development
``````````````````````````````````

So we have a golden rule about ROS development:

⚠️⚠️⚠️ ENCAPSULATE ⚠️⚠️⚠️

Always Always Always Encapsulate!

What do we mean?

An Example of What Not To Do
````````````````````````````

Let's say I want to add some feature to the imaging pipeline. It's something simple: it just reports the number of pixels in the image. I want to publish this data to a special ROS topic, so I'll do something easy, just put the function into the ros node:

.. code-block:: python

    import rospy
    from std_msgs.msg import Int32

    import numpy as np
    ...

    def count_pixels(image):
        return np.sum(image)

    ...

    def publish_pixels(image):
        pixels = count_pixels(image)
        rospy.loginfo(pixels)
        pub.publish(pixels)

    ...

    def main():

    ...

        rospy.init_node('pixels_counter')
        pub = rospy.Publisher('pixels', Int32, queue_size=1)
        rospy.Subscriber('image', Image, publish_pixels)
        rospy.spin()

Great! Let's just push to the ``ROS`` branch and commit. Sounds good, right?

⚠️⚠️⚠️ DO NOT DO THIS! ⚠️⚠️⚠️

Why not?

* Nobody can run, debug, or test this code if they don't have access to a ROS system.
* These changes will not be included in ``core``, so documentation will not be automatically generated for this method
* Someone working on ``core`` might never see this piece of code, so they might write their own ``count_pixels`` function
* Someone running ``pytest`` on the ``core`` branch will not be able to run the tests for this piece of code

Do This Instead
```````````````

Put this method somewhere in ``core``. Let's say in :py:mod:`pipeline`:

.. code-block:: python

    ...

    class Pipeline(object):
        
        ...
    
        def count_pixels(self, image):
            """Count the pixels of an image"""
            return np.sum(image)

Then, call it from the piece of code in the ``ROS`` branch.:

.. code-block:: python

    from Pipeline import pipeline

    def publish_pixels(pipeline, image):
        pixels = pipeline.count_pixels(image)
        rospy.loginfo(pixels)
        pub.publish(pixels)

    def main():
        pipeline = Pipeline(interpreter, .....)

        rospy.init_node('pixels_counter')
        pub = rospy.Publisher('pixels', Int32, queue_size=1)

        ...

        while True:

            ...
            [pipeline stuff]
            ...

            pub.publish(pipeline.count_pixels)

This difference is crucial to understand: the first way commingles ``core`` functionality with ROS code, making debugging and testing a nightmare. The second way keeps ``core`` functionality in ``core``, which allows everyone working on the codebase (not just the linux developers) to understand and debug it. 


Documentation
=============

We have attempted to make writing documentation as easy as possible -- and as close to the codebase as possible! This documentation contains documentation that people have written manually (such as this guide). This manual documentation is written in a format called reStructuredText, which is a commonly-used format for software documentation. To get started writing manual documentation with reStructuredText, read the `reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

The second type of documentation is the auto-generated documentation. This documentation is generated from in-line comments in the codebase. You don't need to touch anything in the `docs/` folder to write this documentation -- just comment your code, and your comments are added to the API page (:py:mod:`odcl`) automatically. The API page will rebuild itself automatically whenever pushes are made to the ``odcl/core`` branch of the repository. 

We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and a tool called `Sphinx Autoapi <https://github.com/readthedocs/sphinx-autoapi>`_ to automatically generate descriptions and API documentation for any class or method with a numpy-formatted docstring. This tool automatically parses the codebase.

.. note::

    The sphinx autodoc can only parse documentation if it is formatted with a ``numpydoc`` style:

    https://numpydoc.readthedocs.io/en/latest/format.html

    For an example of an (excessively) well documented function, see this example:

    https://numpydoc.readthedocs.io/en/latest/example.html#example

At a minimum, we try to document:

* The purpose of the function
* Function arguments and types
* Function returns and types

Building Documentation Locally
------------------------------

You can build a local copy of this documentation without making commits. That way, you can make changes and test locally before committing.

It requires a couple extra dependencies:

.. code-block:: bash

    pip install sphinx-rtd-theme sphinx-autoapi numpydoc

Then go to ``docs/`` and build HTML documentation:

.. code-block:: bash

    cd docs
    make html

Navigate to ``docs/build/html/index.html`` in your web browser to see the documentation. You will need to run ``make html`` to see your code changes reflected.