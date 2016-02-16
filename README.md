eofs - EOF analysis in Python
=============================

[![Build Status](https://travis-ci.org/ajdawson/eofs.png)](https://travis-ci.org/ajdawson/eofs)


Overview
--------

eofs is a Python package for performing empirical orthogonal function (EOF) analysis on spatial-temporal data sets, licensed under the GNU GPLv3.

The package was created to simplify the process of EOF analysis in the Python environment.
Some of the key features are listed below:

* Suitable for large data sets: computationally efficient for the large data sets typical of modern climate model output.
* Transparent handling of missing values: missing values are removed automatically when computing EOFs and re-inserted into output fields.
* Meta-data preserving interfaces (optional): works with both the cdms2 module (from UV-CDAT) and the iris data analysis package to carry meta-data over from input fields to output.
* No Fortran dependencies: written in Python using the power of NumPy, no compilers required.


Requirements
------------

eofs only requires the NumPy package (and setuptools to install).
In order to use the meta-data preserving interfaces one (or both) of cdms2 (part of [UV-CDAT](http://uvcdat.llnl.gov/)) or [iris](http://scitools.org.uk/iris) is needed.


Documentation
-------------

Documentation is available [online](http://ajdawson.github.com/eofs).
The package docstrings are also very complete and can be used as a source of reference when working interactively.


Frequently asked questions
--------------------------

* **Do I need UV-CDAT/cdms2 or iris to use eofs?**
  No. All the computation code uses NumPy only.
  The cdms2 module or iris are only required for the meta-data preserving interfaces.


Installation
------------

eofs works on Python 2 or 3 on Linux, Windows or OSX.
The easiest way to install eofs is by using [conda](http://conda.pydata.org/docs/) or pip:

    conda install -c ajdawson eofs

or

    pip install eofs

You can also install from the source distribution:

    python setup.py install
