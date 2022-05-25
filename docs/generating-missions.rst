***********************************************
Generating Missions for use with qGroundControl
***********************************************

We can generate waypoint missions for qGroundControl (where they can then be sent to the drone) using the offline autopilot. Download the script and an example mission: 

.. code-block:: bash

    # get the mission file
    wget https://raw.githubusercontent.com/uci-uav-forge/uavf/main/tools/missions/MarylandTest.json
    # get the script
    wget https://raw.githubusercontent.com/uci-uav-forge/uavf/main/tools/offline_autopilot.py

Run the script:

.. code-block:: bash

    # run the script
    python offline_autopilot.py MarylandTest.json MarylandTest.mission

This will create a new directory called offlineMissions, and a qGC mission file called MarylandTest.mission will be in that directory.


Run with ``-h`` flag to see options.

.. note::

    This script depends on ``uavfpy``, so make sure it is installed before running:

    .. code-block:: bash

        git clone https://github.com/uci-uav-forge/uavf.git
        cd uavf
        pip install -e .