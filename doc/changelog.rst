Changelog
=========


v0.5.x
------

:Release: v0.5.1
:Date: 10 December 2015

* Include workaround for a regression in numpy versions 1.10.0 and 1.10.1. [`#34 <https://github.com/ajdawson/eofs/issues/34>`_]

Existing users with unaffected numpy versions (<1.10.0, >=1.10.2) don't need to upgrade.


:Release: v0.5.0
:Date: 1 June 2014

* Switched to setuptools for installation.
* Support for Python 3.
  Since neither cdms2 or `iris` are supported on Python 3 yet, only the standard interfaces have been tested on Python 3.


v0.4.x
------

:Release: v0.4.1
:Date: 5 April 2013

* Fixed a serious bug that meant the package could not be imported if the iris module is not available. [`#1 <https://github.com/ajdawson/eofs/issues/1>`_]


:Release: v0.4.0
:Date: 15 March 2013

* First release.
