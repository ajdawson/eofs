eofs - EOF analysis in Python
=============================

[![Build Status](https://travis-ci.org/ajdawson/eofs.png)](https://travis-ci.org/ajdawson/eofs)


Overview
--------

eofs is a Python package for performing EOF analysis on spatial-temporal data sets, licensed under the GNU GPLv3.

The package was created to simplify the process of EOF analysis in the Python environment.
Some of the key features are listed below:

* Suitable for large data sets: computationally efficient for the large data sets typical of modern climate model output.
* Transparent handling of missing values: missing values are removed automatically when computing EOFs and re-inserted into output fields.
* Meta-data preserving interfaces (optional): works with both the cdms2 module (from CDAT) and the Iris data analysis package to carry meta-data over from input fields to output.
* No Fortran dependencies: written in Python using the power of NumPy, no compilers required.


Requirements
------------

eofs only requires the NumPy package.
In order to use the meta-data preserving interfaces one (or both) of cdms2 or iris > 1.2 is required.
cdms2 is part of the Climate Data Analysis Tools ([CDAT](http://www2-pcmdi.llnl.gov/cdat)) or can be obtained separately in the [cdat_lite](http://proj.badc.rl.ac.uk/ndg/wiki/CdatLite) package.
[Iris](http://scitools.org.uk/iris/) is a Python library for meteorology and climatology.


Documentation
-------------

Documentation is available [online](http://ajdawson.github.com/eofs).
The package docstrings are also very complete and can be used as a source of reference when working interactively.


Frequently asked questions
--------------------------

* **Do I need CDAT/cdms2 or Iris to use eofs?**
  No. All the computation code uses NumPy only.
  The cdms2 module or Iris are only required for the meta-data preserving interfaces.


Installation
------------

    sudo python setup.py install

to install system-wide, or to install in your home directory:

    python setup.py install --user


History
-------

This code is the next generation of the [eof2](http://github.com/ajdawson/eof2) package. 
It has been re-written to make it easier to add support for meta-data interfaces other than cdms2/CDAT.
The name has been changed because the old name only makes sense in the context of CDAT.
