Multivariate solvers
====================

In addition to the regular EOF solvers, `eofs` also provides a set of multivariate EOF solvers. These are intended to simplify combined EOF analysis of multiple fields, a good example of this type of analysis is the canonical realtime-multivariate-MJO index calculation of `Wheeler and Hendon (2004)`_.

The multivariate solvers are all named `MultivariateEof` and are kept in the sub-package `eofs.multivariate`. The structure of this package mirrors the structure of the main package, so the multivariate solver for the standard interface is accessed with::

    from eofs.multivariate.standard import MultivariateEof

and similarly for the other interfaces.

These solvers behave in the same way as the regular solvers, except that they take a list of fields as input, and return lists of spatial fields. For example::

    msolver = MultivariateEof([data1, data2, data3])
    eofs_data1, eofs_data2, eofs_data3 = msolver.eofs()

Multivariate EOF analysis only yields a single set of PCs and eigenvalues etc. some calls will be identical to the regular solvers::

    pcs = msolver.pcs()

.. _Wheeler and Hendon (2004): https://dx.doi.org/10.1175/1520-0493(2004)132<1917:AARMMI>2.0.CO;2
