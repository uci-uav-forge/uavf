Installation
============

Installation is, for now, done with ``pip`` using the git flag:

.. code-block:: bash
    
    pip install git+https://github.com/uci-uav-forge/pathplanner

You may run into trouble installing ``mayavi``, especially with later versions of Python (>=3.8), due to 
``vtk`` compatibility issues with Python 3.8.

You can either not use ``mayavi``, or if you still want to use ``mayavi``, use Python 3.7 and try installing ``vtk`` 8.1.2 via pip: 

.. code-block:: bash
    
    pip install vtk==8.1.2
