"""Extended EOF analysis for data in `xarray.DataArray` arrays."""
# (c) Copyright 2010-2016 Andrew Dawson. All Rights Reserved.
# This file is part of eofs.
#
# eofs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# eofs is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with eofs.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import (absolute_import, division, print_function)  # noqa

import collections

try:
    import xarray as xr
except ImportError:
    import xray as xr

from . import standard
from ..tools.xarray import (find_time_coordinates, categorise_ndcoords,
                            weights_array)


class ExtendedEof(object):
    """Extended EOF analysis (meta-data enabled `xarray` interface)"""

    def __init__(self, dataset, lag, weights=None, center=True, ddof=1):
        """Create an ExtendedEof instance.

        The EEOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *dataset*
            A `xarray` DataArray containing the data to be analysed. Time
            must be the first dimension. Missing values are allowed
            provided that they are constant with time (e.g., values of
            an oceanographic field over land).

        *lag*
            The number of timesteps to embed in the rows of the input matrix.
            Because window = lag + 1, lag must be greater than or equal to
            zero. A value of 0 is equivalent to standard Eof analysis.

        **Optional arguments:**

        *weights*
            An array of weights whose shape is compatible with those of
            the input array *dataset*. The weights can have the same
            shape as *dataset* or a shape compatible with an array
            broadcast (i.e., the shape of the weights can can match the
            rightmost parts of the shape of the input array *dataset*).
            If the input array *dataset* does not require weighting then
            the value *None* may be used. Defaults to *None* (no
            weighting).
            Sets the weighting method. The following pre-defined
            weighting methods are available:

        *center*
            If *True*, the mean along the first axis of *dataset* (the
            time-mean) will be removed prior to analysis. If *False*,
            the mean along the first axis will not be removed. Defaults
            to *True* (mean is removed).

        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.

        **Returns:**

        *solver*
            An `ExtendedEof` instance.

        **Examples:**

        EEOF analysis with lag 5::

            from eofs.extended import ExtendedEof
            solver = ExtendedEof(dataset, 5)

        """
        # Check that dataset is a xarray.DataArray.
        if not isinstance(dataset, xr.DataArray):
            raise TypeError('the input data must be a xarray DataArray')
        # Find a time-like dimension, and check if it is the first
        time_coords = find_time_coordinates(dataset)
        if not time_coords:
            raise ValueError('cannot find a time coordinate (must be called '
                             '"time", have a numpy.datetime64 dtype, or have '
                             'an attribute named "axis" with value "T")')
        if len(time_coords) > 1:
            raise ValueError('multiple time dimensions are not allowed')
        if dataset.dims[0] != time_coords[0].name:
            raise ValueError('time must be the first dimension, '
                             'consider using the transpose() method')
        self._timeax = time_coords[0]

        self.window = lag + 1
        if (lag < 0):
            raise ValueError('lag should not be less than 0')
        elif (lag == 0):
            self._lagtimeax = self._timeax
        else:
            # genearate lag time axis
            # Remove last window length from original time axis
            lag_time_axis = self._timeax[:-self.window+1]
            self._lagtimeax = lag_time_axis

        # Verify the presence of at least one spatial dimension. The
        # instance variable channels will also be used as a partial axis
        # list when constructing meta-data. It contains the spatial
        # dimensions.
        self._coords = [dataset.coords[dim] for dim in dataset.dims[1:]]
        if len(self._coords) < 1:
            raise ValueError('one or more spatial dimensions are required')

        # Collect other non-dimension coordinates and store them categorised
        # according to the dimensions they span.
        (self._time_ndcoords,
         self._space_ndcoords,
         self._time_space_ndcoords) = categorise_ndcoords(dataset,
                                                          self._timeax.name)

        # Generate an appropriate set of weights for the input dataset.
        # Determine the required weights.
        if weights is None:
            wtarray = None
        else:
            try:
                wtarray = weights_array(dataset, scheme=weights.lower())
            except AttributeError:
                # Catches exception from applying .lower() to a non-string.
                wtarray = weights
        # Cast the wtarray to the same type as the dataset. This prevents the
        # promotion of 32-bit input to 64-bit on multiplication with the
        # weight array when not required. This will fail with a AttributeError
        # exception if the weights array is None, which it may be if no
        # weighting was requested.
        try:
            wtarray = wtarray.astype(dataset.dtype)
        except AttributeError:
            pass
        # Create an ExtendedEof Solver object using appropriate arguments for
        # this data set. The object will be used for the decomposition and for
        # returning the results.
        self._solver = standard.ExtendedEof(dataset.data,
                                            lag=lag,
                                            weights=wtarray,
                                            center=center,
                                            ddof=ddof)
        # Number of EEOFs in the solution.
        self.neeofs = self._solver.neeofs
        # name for the dataset.
        self._dataset_name = dataset.name

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs) of EEOF.

        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled principal components (default).
            * *1* : Principal components are scaled to unit variance
              (divided by the square-root of their eigenvalue).
            * *2* : Principal components are multiplied by the
              square-root of their eigenvalue.

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of requested PCs is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *pcs*
            A `xarray.DataArray` array containing the ordered PCs. The PCs are
            numbered from 0 to *npcs* - 1.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        pcs = self._solver.pcs(pcscaling, npcs)
        pcdim = xr.IndexVariable('mode', range(pcs.shape[1]),
                                 attrs={'long_name': 'extended_eof_mode_number'})
        coords = [self._lagtimeax, pcdim]
        pcs = xr.DataArray(pcs, coords=coords, name='extended_eof_principal_components')
        pcs.coords.update({coord.name: ('time', coord)
                           for coord in self._time_ndcoords})
        return pcs

    def eeofs(self, eeofscaling=0, neeofs=None):
        """Extended Emipirical orthogonal functions (EEOFs).

        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EEOFs. The following values are
            accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EEOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EEOFs are multiplied by the square-root of their
              eigenvalues.

        *neeofs*
            Number of EEOFs to return. Defaults to all EEOFs. If the
            number of EEOFs requested is more than the number that are
            available, then all available EEOFs will be returned.

        **Returns:**

        *eeofs*
           A `xarray.DataArray` array containing the ordered EEOFs. The EEOFs
           are numbered from 0 to *neeofs* - 1.

        **Examples:**

        All EEOFs with no scaling::

            eeofs = solver.eeofs()

        First 3 EEOFs with scaling applied::

            eeofs = solver.eeofs(neeofs=3, eeofscaling=1)

        """
        eeofs = self._solver.eeofs(eeofscaling, neeofs)
        eeofdim = xr.IndexVariable('mode', range(eeofs.shape[0]),
                                  attrs={'long_name': 'extended_eof_mode_number'})
        lagax = xr.IndexVariable('lag', range(self.window),
                                 attrs={'long_name': 'lag'})
        coords = [eeofdim, lagax] + self._coords
        long_name = 'extended_empirical_orthogonal_functions'
        eeofs = xr.DataArray(eeofs, coords=coords, name='eeofs',
                            attrs={'long_name': long_name})
        eeofs.coords.update({coord.name: (coord.dims, coord)
                            for coord in self._space_ndcoords})
        return eeofs

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EEOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.If the number of eigenvalues requested is more
            than the number that are available, then all available
            eigenvalues will be returned.

        **Returns:**

        *eigenvalues*
            A `xarray.DataArray` array containing the eigenvalues arranged
            largest to smallest.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        lambdas = self._solver.eigenvalues(neigs=neigs)
        eeofdim = xr.IndexVariable('mode', range(lambdas.shape[0]),
                                  attrs={'long_name': 'extended_eof_mode_number'})
        coords = [eeofdim]
        long_name = 'eigenvalues'
        lambdas = xr.DataArray(lambdas, coords=coords, name='eigenvalues',
                               attrs={'long_name': long_name})
        return lambdas

    def varianceFraction(self, neigs=None):
        """Fractional EEOF variances.

        The fraction of the total variance explained by each EOF mode,
        values between 0 and 1 inclusive.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues. If the number of eigenvalues
            requested is more than the number that are available, then
            fractional variances for all available eigenvalues will be
            returned.

        **Returns:**

        *variance_fractions*
            A `xarray.DataArray` array containing the fractional variances for
            each eigenvalue. The eigenvalues are numbered from 0 to *neigs* -
            1.

        **Examples:**

        The fractional variance represented by each eigenvalue::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first 3 eigenvalues::

            variance_fractions = solver.VarianceFraction(neigs=3)

        """
        vf = self._solver.varianceFraction(neigs=neigs)
        eeofdim = xr.IndexVariable('mode', range(len(vf)),
                                  attrs={'long_name': 'extended_eof_mode_number'})
        coords = [eeofdim]
        long_name = 'variance_fractions'
        vf = xr.DataArray(vf, coords=coords, name='variance_fractions',
                          attrs={'long_name': long_name})
        return vf

    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.

        The method of North et al. (1982) is used to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the result may be inappropriate.

        **Optional arguments:**

        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `Eof.varianceFraction`. If *False* then no
            scaling is done. Defaults to *False* (no scaling).

        **Returns:**

        *errors*
            A `xarray.DataArray` containing the typical errors for each
            eigenvalue. The eigenvalues are numbered from 0 to
            *neigs* - 1.

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Examples:**

        Typical errors for all eigenvalues::

            errors = solver.northTest()

        Typical errors for the first 3 eigenvalues scaled by the sum of
        the eigenvalues::

            errors = solver.northTest(neigs=3, vfscaled=True)

        """
        typerrs = self._solver.northTest(neigs=neigs, vfscaled=vfscaled)
        eofdim = xr.IndexVariable('mode', range(typerrs.shape[0]),
                                  attrs={'long_name': 'extended_eof_mode_number'})
        coords = [eofdim]
        long_name = 'typical_errors'
        typerrs = xr.DataArray(typerrs, coords=coords, name='typical_errors',
                               attrs={'long_name': long_name})
        return typerrs

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EEOFs.

        If weights were passed to the `ExtendedEof` instance the returned
        reconstructed field will automatically have this weighting removed.
        Otherwise the returned field will have the same weighting as the
        `ExtendedEof` input *dataset*.

        Returns the reconstructed field in a `xarray.DataArray`.

        **Argument:**

        *neofs*
            Number of EEOFs to use for the reconstruction.
            Alternatively this argument can be an iterable of mode
            numbers (where the first mode is 1) in order to facilitate
            reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            A `xarray.DataArray` with the same dimensions `ExtendedEof` input
            *dataset* containing the reconstruction using *neofs* EOFs.

        **Example:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        rfield = self._solver.reconstructedField(neofs)
        coords = [self._timeax] + self._coords
        if isinstance(neofs, collections.Iterable):
            name_part = 'EOFs_{}'.format('_'.join([str(e) for e in neofs]))
        else:
            name_part = '{}_EOFs'.format(neofs)
        long_name = '{!s}_reconstructed_with_{!s}'.format(self._dataset_name,
                                                          name_part)
        rfield = xr.DataArray(rfield, coords=coords, name=self._dataset_name,
                              attrs={'long_name': long_name})
        ndcoords = (self._time_ndcoords + self._space_ndcoords +
                    self._time_space_ndcoords)
        rfield.coords.update({coord.name: (coord.dims, coord)
                              for coord in ndcoords})
        return rfield
