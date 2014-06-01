.. default-role:: py:obj

.. toctree::
   :maxdepth: 2
   :hidden:

   userguide/index
   examples/index
   api/index
   downloads


Package description
===================

`eofs` is a Python package for EOF analysis of spatial-temporal data, some of its key features are:

* **Suitable for large data sets:** computationally efficient for the large output data sets of modern climate models.
* **Transparent handling of missing values:** missing values are removed automatically during computations and placed back into output fields.
* **Automatic metadata:** metadata from input fields is used to construct metadata for output fields (requires the `cdms2` module from CDAT_ or the iris_ data analysis package).
* **No Compiler required:** a fast implementation written in pure Python using the power of numpy_, no Fortran or C dependencies.


Download and installation
-------------------------

Released versions of `eofs` can be downloaded from the :doc:`downloads` page.
You must have setuptools_ installed in order to install `eofs`.
After downloading the source code archive, unzip it and change into the unzipped archive's directory, then to install it:

.. code-block:: bash

   $ python setup.py install

`eofs` can also be installed from PyPI using pip:

.. code-block:: bash

   $ pip install eofs

You can also check out the source code for the development version from the `github repository <https://github.com/ajdawson/eofs>`_ to access features which are not yet in the released version.


Getting started
---------------

`eofs` provides three interfaces for EOF analysis: one for analysing data contained in `numpy` arrays or masked arrays, suitable for any data set; and two for meta-data aware EOF analysis, suitable for analysing data read from self-describing files, using either the `cdms2` or `iris` packages.
All the interfaces support the same set of operations.

Regardless of which interface you use, the basic usage is the same.
The EOF analysis is handled by a solver class.
The EOF solution is computed when the solver class in instantiated.
Method calls are then used to retrieve the quantities of interest from the solver class.

The following is a very simple illustrative example which computes the leading 2 EOFs of a temporal spatial field using the `eofs.cdms` interface:

.. code-block:: python

   import cdms2
   from eofs.cdms import Eof

   # read a spatial-temporal field, time must be the first dimension
   ncin = cdms2.open('sst_monthly.nc')
   sst = ncin('sst')
   ncin.close()

   # create a solver class, taking advantage of built-in weighting
   solver = Eof(sst, weights='coslat')

   # retrieve the first two EOFs from the solver class
   eofs = solver.eofs(neofs=2)

More detailed description of usage are found in the :doc:`userguide/index` or the :doc:`examples/index`.


Requirements
------------

This package requires as a minimum that you have numpy_ available, and requires setuptools_ for installation.
The `eofs.cdms` meta-data enabled interface can only be used if the `cdms2` module is available.
This module is distributed as part of the CDAT_ project.
It is also distributed as part of the cdat-lite_ package.
The `eofs.iris` meta-data enabled interface can only be used if the iris_ package is available at version 1.2 or higher.


Compatibility with **eof2**
---------------------------

See the :doc:`userguide/eof2_compatibility` section of the :doc:`userguide/index`.


Developing and contributing
---------------------------

All development is done through `Github <http://github.com/ajdawson/eofs>`_. To check out the latest sources run:

.. code-block:: bash

   $ git clone git://github.com/ajdawson/eofs.git

It is always a good idea to run the tests during development, to do so:

.. code-block:: bash

   $ cd eofs
   $ nosetests -sv

Running the tests requires nose_.

Bug reports and feature requests should be filed using the Github issues_ system.
If you have code you would like to contribute, fork the `repository <http://github.com/ajdawson/eofs>`_ on Github, do the work on a feature branch of your fork, push your feature branch to *your* Github fork, and send a pull request.


History
-------

`eofs` is the successor to eof2_.
`eof2` was always written to be part of CDAT which already had a package called `eof`, hence the name `eof2`.
When support for `iris` was added it didn't make sense to call the package `eof2` anymore, and `eofs` was created.
`eofs` is much more than just a name change.
The package is restructured to make it easier to add new meta-data interfaces, and a comprehensive set of tests has been added to ensure that the package remains stable when new features are added.


.. _CDAT: http://uv-cdat.llnl.gov

.. _iris: http://scitools.org.uk/iris

.. _numpy: http://numpy.scipy.org

.. _cdat-lite: http://proj.badc.rl.ac.uk/cedaservices/wiki/CdatLite

.. _nose: https://nose.readthedocs.org/en/latest/

.. _setuptools: https://pypi.python.org/pypi/setuptools

.. _issues: http://github.com/ajdawson/eofs/issues?state=open

.. _eof2: http://ajdawson.github.com/eof2
