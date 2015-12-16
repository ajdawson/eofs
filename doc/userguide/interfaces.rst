.. default-role:: py:obj

Solver interfaces
=================

`eofs` uses different interfaces to work with different types of input data Descriptions of each interface are below, and summarised in the following table:

.. tabularcolumns:: |L|L|

========= =====================================================================
Interface Type of input/output data
========= =====================================================================
cdms      Data read from a self-describing data format stored in a `cdms2`
          variable.
iris      Data read from a self-describing data format stored in an `iris`
          cube.
standard  Other data, stored in a `numpy.ndarray` or a `numpy.ma.MaskedArray`.
========= =====================================================================


.. _cdms-interface:

cdms interface
--------------

The `eofs.cdms` interface works with cdms variables, which are the core data container used by UV-CDAT_. A cdms variable has meta-data associated with it, including dimensions, which are understood by the `eofs.cdms.Eof` solver interface. The outputs of the `eofs.cdms.Eof` solver are cdms variables with meta-data, which can be written straight to a netCDF file using cdms, or used with other parts of the UV-CDAT framework.


.. _iris-interface:

Iris interface
--------------

The iris interface works with `~iris.cube.Cube` objects, which are the data containers used by the iris_ data analysis package. The meta-data, including coordinate dimensions, associated with iris `~iris.cube.Cube` objects is understood by the `eofs.iris.Eofs` solver interface. The outputs of the `eofs.iris.Eof` solver are also contained in `~iris.cube.Cube` objects, meaning they can be used with tools in the iris package, and easily written to a file.


.. _standard-interface:

Standard interface
------------------

The standard interface works with `numpy` arrays, which makes the standard interface the general purpose interface. Any data that can be stored in a `numpy.ndarray` or `numpy.ma.MaskedArray` can be analysed with the `eofs.standard.Eof` solver interface.


.. _iris: http://scitools.org.uk/iris

.. _UV-CDAT: http://uv-cdat.llnl.gov
