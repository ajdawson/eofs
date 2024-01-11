.. default-role:: py:obj

Basic usage
===========

This section describes the basic usage of the solver classes in a generic way that is applicable to all interfaces.


Importing the solver class
--------------------------

The solver classes are all called `Eof` and are stored in interface specific modules: `eofs.<interface>.Eof`. To import the standard interface solver class::

    from eofs.standard import Eof
 
the iris interface solver class::

    from eofs.iris import Eof

and the xarray interface solver class::

    from eofs.xarray import Eof


Creating a solver
-----------------

Creating an instance of a solver calss can be as simple as::

    solver = Eof(data)

but all interfaces have options available. It is highly likely that input data will need to be weighted prior to EOF analysis, which is best done using the *weights* keyword argument. This argument accepts an array of weights in all interfaces, and also a selection of pre-defined weighting method names in the iris interface::

    solver = Eof(data, weights=weights_array)

.. note::

   It is strongly recommended that weights be applied in this way, rather than applying weighting to ``data`` before passing it to the solver class. The solver class handles weighting in an intelligent way internally, and passing the weights using the *weights* keyword is the only way to let the solver class know that weights are required.

Two more keyword arguments are accepted by all interfaces. The *center* argument, which defaults to ``True`` determines whether or not the time-mean is removed from the input data set before analysis. In general this should be set to ``True`` unless you are absolutely sure you don't want to do this! The *ddof* keyword controls the degrees of freedom used in normalizing the covariance matrix. Its default value is ``1`` meaning the covariance matrix is normalized by :math:`N - 1` where :math:`N` is the number of temporal samples in the input data set.


Retrieving results from the solver
----------------------------------

Once a solver has been constructed the results of the analysis can be obtained by calling methods of the solver class. For example, to retrieve the PCs::

    pcs = solver.pcs()

If we only wanted the first 5 PCs, and wanted them scaled to unit variance we could use::

    pcs = solver.pcs(npcs=5, pcscaling=1)

Similar calls could be made to retrieve the EOFs themselves::

    eofs = solver.eofs()

It is also possible to perform more complex actions using the solver, here we reconstruct the original input field using only 4 EOF modes::

    reconstructed_data = solver.reconstructedField(4)

We could also project another field onto the EOFs to produce a set of pseudo-PCs::

    pseudo_pcs = solver.projectField(other_field)

