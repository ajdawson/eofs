Changelog
=========

.. default-role:: py:obj


v1.3
----

:Release: v1.3.0
:Date: 12 October 2017

* Minor change for xarray deprecations resulting in a different xarray version dependency (now >= 0.9) [`#87 <https://github.com/ajdawson/eofs/pull/87>`_].
* Bug fixes for Python 3 [`#88 <https://github.com/ajdawson/eofs/pull/88>`_].


v1.2
----

:Release: v1.2.0
:Date: 25 October 2016

* Improved the detection of time axes in the iris interface [`#68 <https://github.com/ajdawson/eofs/issues/68>`_, `#69 <https://github.com/ajdawson/eofs/pull/69>`_].


v1.1
----

:Release: v1.1.0
:Date: 3 March 2016

* Added an `xarray <http://xarray.pydata.org>`_ interface allowing use of `xarray.DataArray` objects with :py:mod:`eofs`.
* Fix bug in setting `ddof` in `eofs.tools.iris.covariance_map`, previously it was always `1` regardless of the `ddof` keyword.
* Better Python 2 and 3 support without relying on 2to3.


v1.0
----

:Release: v1.0.0
:Date: 16 February 2016

* Allow arbitrary mode selection when reconstructing a field from EOFs.
* Auxiliary coordinates are now preserved in the `eofs.iris` and `eofs.multivariate.iris` interfaces.
* Improved documentation, including the addition of a developer guide.


v0.5
----

:Release: v0.5.1
:Date: 10 December 2015

* Include workaround for a regression in numpy versions 1.10.0 and 1.10.1. [`#34 <https://github.com/ajdawson/eofs/issues/34>`_]

Existing users with unaffected numpy versions (<1.10.0, >=1.10.2) don't need to upgrade.


:Release: v0.5.0
:Date: 1 June 2014

* Switched to setuptools for installation.
* Support for Python 3.
  Since neither cdms2 or `iris` are supported on Python 3 yet, only the standard interfaces have been tested on Python 3.


v0.4
----

:Release: v0.4.1
:Date: 5 April 2013

* Fixed a serious bug that meant the package could not be imported if the iris module is not available. [`#1 <https://github.com/ajdawson/eofs/issues/1>`_]


:Release: v0.4.0
:Date: 15 March 2013

* First release.
