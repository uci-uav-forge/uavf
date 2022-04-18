Documentation
*************

We have attempted to make writing documenation as easy as possible -- and as close to the codebase as possible!

The API documentation page will rebuild itself automatically whenever pushes are made to the ``odcl/core`` branch of the repository. We use sphinx and the ``autodoc`` extension to automatically generate descriptions and API documentation for any class or method with a Numpy docstring.

The sphinx autodoc can only parse documentation if it is formatted with a ``numpydoc`` style:

https://numpydoc.readthedocs.io/en/latest/format.html

For an example of an (excessively) well documented function, see this example:

https://numpydoc.readthedocs.io/en/latest/example.html#example

At a minimum, we try to document:

* The purpose of the function
* Function arguments and types
* Function returns and types

Building Documentation
----------------------

You can build a local copy of this documentation without making commits. That way, you can make changes and test locally before committing.

It requires a couple extra dependencies:

.. code-block:: bash

    pip install sphinx-rtd-theme sphinx-autoapi numpydoc

Then go to ``docs/`` and build HTML documentation:

.. code-block:: bash

    cd docs
    make html

Navigate to ``docs/build/html/index.html`` in your web browser to see the documentation. You will need to run ``make html`` to see your code changes reflected.