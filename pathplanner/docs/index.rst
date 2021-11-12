.. No Errors Test Project documentation master file, created by
   sphinx-quickstart on Fri Aug 30 17:07:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

UAV Forge Path Planner Project Documentation
############################################

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Git Repository<https://github.com/uci-uav-forge/pathplanner>
   Installation<installation>


Package Components
------------------

Boustrophedon Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^
   -  Boustrophedon Decomposition (line sweep) algorithm
   -  Reeb Graph construction
   -  (EXPERIMENTAL) feasible path generation from Reeb Graph via
      Straight Skeleton

Polygon Generation
^^^^^^^^^^^^^^^^^^
   -  non-convex polygon generation
   -  2d polygon (boundary) visualization

Path Planning on 2D Embedded Surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   -  generate surface to embed in R^3
   -  optimally embed surface to minimize height, while:

      -  clearing obstacles constraint
      -  1st, 2nd *x, y* derivative constraints
      -  minimum height constraint

   -  visualize surfaces in 2D with ``matplotlib``
   -  visualize surfaces in 3D with ``matplotlib`` (buggy)
   -  visualize surfaces in 3D with ``mayavi`` – requires ``mayavi``
      installation

Planners
^^^^^^^^

Probabilistic Road Map Planner
""""""""""""""""""""""""""""""

   -  generate PRM (Probabilistic Roadmap Planner) for embedded surface
   -  PRM uses delaunay triangulation to compute edges
   -  tunable cost function
   -  visualize PRM with ``matplotlib``

A-Star Planner
""""""""""""""

   -  generate A-Star planner for embedded surface
   -  tunable cost function
   -  visualize A-Star planner output

RRT Planner
"""""""""""

   -  generate RRT (Rapidly Exploring Random Tree) planner for embedded
      surface
   -  tunable cost function
   -  visualize RRT with ``matplotlib``



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
