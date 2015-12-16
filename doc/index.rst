.. default-role:: py:obj

.. toctree::
   :maxdepth: 2
   :hidden:

   userguide/index
   examples/index
   api/index
   downloads
   devguide/index


Package description
===================

`eofs` is a Python package for EOF analysis of spatial-temporal data.
Using EOFs (empirical orthogonal functions) is a common technique to decompose a signal varying in time and space into a form that is easier to interpret in terms of spatial and temporal variance.
Some of the key features of `eofs` are:

* **Suitable for large data sets:** computationally efficient for the large output data sets of modern climate models.
* **Transparent handling of missing values:** missing values are removed automatically during computations and placed back into output fields.
* **Automatic metadata:** metadata from input fields is used to construct metadata for output fields (requires the `cdms2` module from CDAT_ or the iris_ data analysis package).
* **No Compiler required:** a fast implementation written in pure Python using the power of numpy_, no Fortran or C dependencies.


Download and installation
-------------------------

The core of the package runs on Python 2 or 3, on Linux, Windows or Mac OS X; basically anywhere Python+NumPy are available.
The :ref:`cdms <cdms-interface>` and :ref:`iris <iris-interface>` interfaces are available on all platforms where their respective supporting packages CDAT_ and iris_ can be installed.

`eofs` can be installed for all platforms using pip:

    pip install eofs

If you want to install on Linux or OS X, you can also use the conda_ package manager:

    conda install -c ajdawson eofs

The source code for released versions of `eofs` can be downloaded from the :doc:`downloads` page.
You must have setuptools_ installed in order to install `eofs` from source.
After downloading the source code archive, unzip it and change into the unzipped archive's directory, then to install it:

    python setup.py install

You can also check out the source code for the development version from the `github repository <https://github.com/ajdawson/eofs>`_ to access features which are not yet in a release.


Getting started
---------------

`eofs` provides three interfaces for EOF analysis: one for analysing data contained in `numpy` arrays or masked arrays, suitable for any data set; and two for meta-data aware EOF analysis, suitable for analysing data read from self-describing files, using either the `cdms2` or `iris` packages.
All the interfaces support the same set of operations.

Regardless of which interface you use, the basic usage is the same.
The EOF analysis is handled by a solver class, and the EOF solution is computed when the solver class is created.
Method calls are then used to retrieve the quantities of interest from the solver class.

The following is a very simple illustrative example which computes the leading 2 EOFs of a temporal spatial field using the `eofs.iris` interface:

.. code-block:: python

   import iris
   from eofs.iris import Eof

   # read a spatial-temporal field, time must be the first dimension
   sst = iris.load_cube('sst_monthly.nc')

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

Contributions big or small are welcomed from anyone with an interest in the project.
Bug reports and feature requests can be filed using the Github issues_ system.
If you would like to contribute code or documentation please see the :doc:`devguide/index`.


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

.. _setuptools: https://pypi.python.org/pypi/setuptools

.. _issues: http://github.com/ajdawson/eofs/issues

.. _eof2: http://ajdawson.github.com/eof2

.. _conda: http://conda.pydata.org/docs/
