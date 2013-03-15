.. default-role:: py:obj


Compatibility with **eof2**
===========================

`eofs` is an evolution of the `eof2` package. As such its usage is still largely the same as `eof2`, with a few important exceptions.


Namespace changes
-----------------

The package has been re-organized and the namespaces are slightly different. Below is a summary:

.. tabularcolumns:: |L|L|

============================================  ===========================
Old name                                      New name
============================================  ===========================
`eof2.eofsolve.EofSolver` / `eof2.EofSolver`  `eofs.standard.Eof`
`eof2.eofwrap.Eof` / `eof2.Eof`               `eofs.cdms.Eof`
`eof2.nptools`                                `eofs.tools.standard`
`eof2.tools`                                  `eofs.tools.cdms`
============================================  ===========================


Metadata changes
----------------

Some of the metadata names (e.g., long_name) for the `eofs.cdms` interface are different from those used in `eof2`.
These changes are an effort to be more descriptive about the returned variable.


New features
------------

Some new features have been added:

* `eofs.iris` and `eofs.tools.iris` provide a new interface to support data read using iris_

* `eofs.multivariate.standard`, `eofs.multivariate.cdms` and `eofs.multivariate.iris` provide support for simple multivariate EOF analysis


.. _iris: http://scitools.org.uk/iris
