Contributing
************

This page is a guide to making contributions to the codebase.

What should go into the repository?
-----------------------------------

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

Core Developer Workflow
-----------------------

The core developer workflow looks something like this:

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
