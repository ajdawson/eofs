"""EOF analysis for data in `xarray.DataArray` arrays."""
# (c) Copyright 2016 Andrew Dawson. All Rights Reserved.
#
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
from .tools.xarray import (find_time_coordinates, categorise_ndcoords,
                           weights_array)


class Eof(object):
    """EOF analysis (meta-data enabled `xarray` interface)"""

    def __init__(self, array, weights=None, center=True, ddof=1):
        """Create an Eof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *dataset*
            An `xarray.DataArray` with two or more dimensions containing
            the data to be analysed. The first dimension is assumed to
            represent time. Missing values are allowed provided that
            they are constant with time (e.g., values of an
            oceanographic field over land).

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

        *center*
            If *True*, the mean along the first axis of *dataset* (the
            time-mean) will be removed prior to analysis. If *False*,
            the mean along the first axis will not be removed. Defaults
            to *True* (mean is removed).

            The covariance interpretation relies on the input data being
            anomaly data with a time-mean of 0. Therefore this option
            should usually be set to *True*. Setting this option to
            *True* has the useful side effect of propagating missing
            values along the time dimension, ensuring that a solution
            can be found even if missing values occur in different
            locations at different times.

        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.

        **Returns:**

        *solver*
            An `Eof` instance.

        **Examples:**

        EOF analysis with no weighting::

            from eofs.xarray import Eof
            solver = Eof(data_array)

        """
        if not isinstance(array, xr.DataArray):
            raise TypeError('the input must be an xarray DataArray')
        # Find a time-like dimension, and check if it is the first.
        time_coords = find_time_coordinates(array)
        if not time_coords:
            raise ValueError('cannot find a time coordinate (must be called '
                             '"time", have a numpy.datetime64 dtype, or have '
                             'an attribute named "axis" with value "T")')
        if len(time_coords) > 1:
            raise ValueError('multiple time dimensions are not allowed')
        if array.dims[0] != time_coords[0].name:
            raise ValueError('time must be the first dimension, '
                             'consider using the transpose() method')
        self._time = time_coords[0]
        # Collect the other dimension coordinates.
        self._coords = [array.coords[dim] for dim in array.dims[1:]]
        # Collect other non-dimension coordinates and store them categorised
        # them according to the dimensions they span.
        (self._time_ndcoords,
         self._space_ndcoords,
         self._time_space_ndcoords) = categorise_ndcoords(array,
                                                          self._time.name)
        # Determine the required weights.
        if weights is None:
            wtarray = None
        else:
            try:
                wtarray = weights_array(array, scheme=weights.lower())
            except AttributeError:
                # Catches exception from applying .lower() to a non-string.
                wtarray = weights
        try:
            wtarray = wtarray.astype(array.dtype)
        except AttributeError:
            pass
        # Construct the EOF solver.
        self._solver = standard.Eof(array.values,
                                    weights=wtarray,
                                    center=center,
                                    ddof=ddof)
        # Name of the input DataArray.
        self._name = array.name
        #: The number of EOFs in the solution.
        self.neofs = self._solver.neofs

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

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
            A `~xarray.DataArray` containing the ordered PCs. The PCs
            are numbered from 0 to *npcs* - 1.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        pcs = self._solver.pcs(pcscaling, npcs)
        pcdim = xr.IndexVariable('mode', range(pcs.shape[1]),
                                 attrs={'long_name': 'eof_mode_number'})
        coords = [self._time, pcdim]
        pcs = xr.DataArray(pcs, coords=coords, name='pcs')
        pcs.coords.update({coord.name: ('time', coord)
                           for coord in self._time_ndcoords})
        return pcs

    def eofs(self, eofscaling=0, neofs=None):
        """Emipirical orthogonal functions (EOFs).

        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EOFs. The following values are
            accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalues.

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs*
           A `~xarray.DataArray` containing the ordered EOFs. The EOFs
           are numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs with no scaling::

            eofs = solver.eofs()

        First 3 EOFs with scaling applied::

            eofs = solver.eofs(neofs=3, eofscaling=1)

        """
        eofs = self._solver.eofs(eofscaling, neofs)
        eofdim = xr.IndexVariable('mode', range(eofs.shape[0]),
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim] + self._coords
        long_name = 'empirical_orthogonal_functions'
        eofs = xr.DataArray(eofs, coords=coords, name='eofs',
                            attrs={'long_name': long_name})
        eofs.coords.update({coord.name: (coord.dims, coord)
                            for coord in self._space_ndcoords})
        return eofs

    def eofsAsCorrelation(self, neofs=None):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        correlation between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        .. note::

            These are not related to the EOFs computed from the
            correlation matrix.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs*
           A `~xarray.DataArray` containing the ordered EOFs. The EOFs
           are numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCorrelation()

        The leading EOF::

            eof1 = solver.eofsAsCorrelation(neofs=1)

        """
        eofs = self._solver.eofsAsCorrelation(neofs)
        eofdim = xr.IndexVariable('mode', range(eofs.shape[0]),
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim] + self._coords
        long_name = 'correlation_between_pcs_and_{!s}'.format(self._name)
        eofs = xr.DataArray(eofs, coords=coords, name='eofs',
                            attrs={'long_name': long_name})
        eofs.coords.update({coord.name: (coord.dims, coord)
                            for coord in self._space_ndcoords})
        return eofs

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        covariance between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        **Optional arguments:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        *pcscaling*
            Set the scaling of the PCs used to compute covariance. The
            following values are accepted:

            * *0* : Un-scaled PCs.
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue) (default).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

            The default is to divide PCs by the square-root of their
            eigenvalue so that the PCs are scaled to unit variance
            (option 1).

        **Returns:**

        *eofs*
           A `~xarray.DataArray` containing the ordered EOFs. The EOFs
           are numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCovariance()

        The leading EOF::

            eof1 = solver.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs::

            eof1 = solver.eofsAsCovariance(neofs=1, pcscaling=0)

        """
        eofs = self._solver.eofsAsCovariance(neofs, pcscaling)
        eofdim = xr.IndexVariable('mode', range(eofs.shape[0]),
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim] + self._coords
        long_name = 'covariance_between_pcs_and_{!s}'.format(self._name)
        eofs = xr.DataArray(eofs, coords=coords, name='eofs',
                            attrs={'long_name': long_name})
        eofs.coords.update({coord.name: (coord.dims, coord)
                            for coord in self._space_ndcoords})
        return eofs

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.If the number of eigenvalues requested is more
            than the number that are available, then all available
            eigenvalues will be returned.

        **Returns:**

        *eigenvalues*
            A `~xarray.DataArray` containing the eigenvalues arranged
            largest to smallest. The eigenvalues are numbered from 0 to
            *neigs* - 1.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        lambdas = self._solver.eigenvalues(neigs=neigs)
        eofdim = xr.IndexVariable('mode', range(lambdas.shape[0]),
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim]
        long_name = 'eigenvalues'
        lambdas = xr.DataArray(lambdas, coords=coords, name='eigenvalues',
                               attrs={'long_name': long_name})
        return lambdas

    def varianceFraction(self, neigs=None):
        """Fractional EOF mode variances.

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
            A `~xarray.DataArray` containing the fractional variances
            for each eigenvalue. The eigenvalues are numbered from 0 to
            *neigs* - 1.

        **Examples:**

        The fractional variance represented by each eigenvalue::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first 3 eigenvalues::

            variance_fractions = solver.VarianceFraction(neigs=3)

        """
        vf = self._solver.varianceFraction(neigs=neigs)
        eofdim = xr.IndexVariable('mode', range(vf.shape[0]),
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim]
        long_name = 'variance_fractions'
        vf = xr.DataArray(vf, coords=coords, name='variance_fractions',
                          attrs={'long_name': long_name})
        return vf

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).

        **Returns:**

        *total_variance*
            A scalar value (not a `~xarray.DataArray`).

        **Example:**

        Get the total variance::

            total_variance = solver.totalAnomalyVariance()

        """
        return self._solver.totalAnomalyVariance()

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
            A `~xarray.DataArray` containing the typical errors for each
            eigenvalue. The egienvalues are numbered from 0 to
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
                                  attrs={'long_name': 'eof_mode_number'})
        coords = [eofdim]
        long_name = 'typical_errors'
        typerrs = xr.DataArray(typerrs, coords=coords, name='typical_errors',
                               attrs={'long_name': long_name})
        return typerrs

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the `Eof` instance the returned
        reconstructed field will automatically have this weighting
        removed. Otherwise the returned field will have the same
        weighting as the `Eof` input *dataset*.

        Returns the reconstructed field in a `~xarray.DataArray`.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction.
            Alternatively this argument can be an iterable of mode
            numbers (where the first mode is 1) in order to facilitate
            reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            A `~xarray.DataArray` with the same dimensions `Eof` input
            *dataset* containing the reconstruction using *neofs* EOFs.

        **Example:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        rfield = self._solver.reconstructedField(neofs)
        coords = [self._time] + self._coords
        if isinstance(neofs, collections.Iterable):
            name_part = 'EOFs_{}'.format('_'.join([str(e) for e in neofs]))
        else:
            name_part = '{}_EOFs'.format(neofs)
        long_name = '{!s}_reconstructed_with_{!s}'.format(self._name,
                                                          name_part)
        rfield = xr.DataArray(rfield, coords=coords, name=self._name,
                              attrs={'long_name': long_name})
        ndcoords = (self._time_ndcoords + self._space_ndcoords +
                    self._time_space_ndcoords)
        rfield.coords.update({coord.name: (coord.dims, coord)
                              for coord in ndcoords})
        return rfield

    def projectField(self, array, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a data set, projects it onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *field*
            An `xarray.DataArray` containing the field to project onto
            the EOFs. It must have the same corresponding spatial
            dimensions (including missing values in the same places) as
            the `Eof` input *dataset*. It may have a different length
            time dimension to the `Eof` input *dataset* or no time
            dimension at all. If a time dimension exists it must be the
            first dimension.

        **Optional arguments:**

        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then the field will be projected onto all
            available EOFs.

        *eofscaling*
            Set the scaling of the EOFs that are projected
            onto. The following values are accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then the field is weighted using the same weights
            used for the EOF analysis prior to projection. If *False*
            then no weighting is applied. Defaults to *True* (weighting
            is applied). Generally only the default setting should be
            used.

        **Returns:**

        *pseudo_pcs*
            A `~xarray.DataArray` containing the pseudo-PCs. The PCs are
            numbered from 0 to *neofs* - 1.

        **Examples:**

        Project a field onto all EOFs::

            pseudo_pcs = solver.projectField(field)

        Project fields onto the three leading EOFs::

            pseudo_pcs = solver.projectField(field, neofs=3)

        """
        if not isinstance(array, xr.DataArray):
            raise TypeError('the input must be an xarray DataArray')
        array_name = array.name
        time_coords = find_time_coordinates(array)
        if len(time_coords) > 1:
            raise ValueError('multiple time dimensions are not allowed')
        if time_coords:
            has_time = True
            time_coord = time_coords[0]
            if array.dims[0] != time_coord.name:
                raise ValueError('time must be the first dimension, '
                                 'consider using the transpose() method')
            time_ndcoords, _, _ = categorise_ndcoords(array, time_coord.name)
        else:
            has_time = False
        pcs = self._solver.projectField(array.values,
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        # Create the PCs DataArray.
        if pcs.ndim == 2:
            pcdim = xr.IndexVariable('mode', range(pcs.shape[1]),
                                     attrs={'long_name': 'eof_mode_number'})
            pcs = xr.DataArray(
                pcs,
                coords=[time_coord, pcdim], name='pseudo_pcs',
                attrs={'long_name': '{}_pseudo_pcs'.format(array_name)})
        else:
            pcdim = xr.IndexVariable('mode', range(pcs.shape[0]),
                                     attrs={'long_name': 'eof_mode_number'})
            pcs = xr.DataArray(
                pcs,
                coords=[pcdim], name='pseudo_pcs',
                attrs={'long_name': '{}_pseudo_pcs'.format(array_name)})
        if has_time:
            # Add non-dimension coordinates.
            pcs.coords.update({coord.name: (coord.dims, coord)
                               for coord in time_ndcoords})
        return pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights*
            An array contaning the analysis weights (not a
            `~xarray.DataArray`).

        **Example:**

        The weights used for the analysis::

            weights = solver.getWeights()

        """
        return self._solver.getWeights()
