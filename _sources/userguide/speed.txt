.. default-role:: py:obj

Speeding up the solvers
=======================

All the solvers rely on the `numpy.linalg.svd` function to compute the EOF solution. If you obtained numpy_ from you package manager it is likely that this function uses standard LAPACK_/BLAS_ routines, which are single threaded. It is possible to get a large performance boost from the code by using a version of numpy that is built with an optimised linear algebra library. The following options all allow a performace boost for the solvers in `eofs`:

NumPy built with the Intel Math Kernel Library (MKL)
----------------------------------------------------

Intel's MKL provides highly optimised BLAS and LAPACK routines which can take advantage of multicore processors.
If you have access to Intel's MKL you can build a numpy_ library using it following the instructions from intel_.
It is actually not that hard to do and can provide a great performance boost.

Enthought Python Distribution (EPD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The version of numpy_ provided with EPD is linked against the Intel MKL, and you don't need a license for the MKL yourself. EPD is free for academics.

Numpy built with the AMD Core Math Library (ACML)
-------------------------------------------------

AMD's ACML provides optimised BLAS and LAPACK routines in both single and multi-threaded versions. The ACML itself is free to download.
You will have to build CBLAS first as ACML only includes Fortran interfaces.
The following instructions worked for me:

1. download CBLAS from the link at www.netlib.org/blas

2. edit Makefile.LINUX and change BLLIB to point to your libacml.so or libacml_mp.so

3. copy or link this make file to Makefile.in and build CBLAS

4. copy the resulting cblas library to libcblas.a in the same directory as the ACML library

5. download a stable version fo the numpy source code from http://sourceforge.net/projects/numpy/files/NumPy/ or the latest code from https://github.com/numpy/numpy

6. create a site.cfg in the numpy source tree (copy site.cfg.example) and add::

    [blas]
    blas_libs = cblas, acml
    library_dirs = /path-to-acml-and-cblas/lib
    include_dirs = /path-to-acml-and-cblas/include

    [lapack]
    language = f77
    lapack_libs = acml
    library_dirs = /path-to-acml-and-cblas/lib
    include_dirs = /path-to-acml-and-cblas/include

7. build and install numpy as normal 


.. _numpy: http://numpy.org

.. _LAPACK: http://www.netlib.org/lapack/

.. _BLAS: http://www.netlib.org/blas/

.. _intel: http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl
