.. default-role:: py:obj

Solver interfaces
=================

`eofs` uses different interfaces to work with different types of input data Descriptions of each interface are below, and summarised in the following table:

.. tabularcolumns:: |L|L|

========= =====================================================================
Interface Type of input/output data
========= =====================================================================
iris      Data read from a self-describing data format stored in an `iris`
          cube.
xarray    Data stored in an `xarray.DataArray`.
standard  Other data, stored in a `numpy.ndarray`, a `numpy.ma.MaskedArray`,
          or a `dask.array.Array`.
========= =====================================================================


.. _iris-interface:

Iris interface
--------------

The iris interface works with `~iris.cube.Cube` objects, which are the data containers used by the iris_ data analysis package. The meta-data, including coordinate dimensions, associated with iris `~iris.cube.Cube` objects is understood by the `eofs.iris.Eofs` solver interface. The outputs of the `eofs.iris.Eof` solver are also contained in `~iris.cube.Cube` objects, meaning they can be used with tools in the iris package, and easily written to a file.


.. _xarray-interface:

xarray interface
----------------

The xarray interface works with `~xarray.DataArray` objects, which are the data containers used by the xarray_ package. The meta-data, including coordinate dimensions, associated with `~xarray.DataArray` objects is understood by the `eofs.xarray.Eofs` solver interface. The outputs of the `eofs.xarray.Eof` solver are also contained in `~xarray.DataArray` objects, allowing their use within other `xarray` tools including serialization to netCDF.


.. _standard-interface:

Standard interface
------------------

The standard interface works with numpy_ arrays or dask_ arrays, which makes the standard interface the general purpose interface. Any data that can be stored in a `numpy.ndarray`, `numpy.ma.MaskedArray` or `dask.array.Array` can be analysed with the `eofs.standard.Eof` solver interface.


.. _iris: https://scitools-iris.readthedocs.io/en/stable/

.. _xarray: https://docs.xarray.dev/en/stable/

.. _dask: https://www.dask.org/

.. _numpy: https://numpy.org