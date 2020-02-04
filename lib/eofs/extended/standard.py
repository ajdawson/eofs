"""Extended EOF analysis for data in `numpy.ndarray` arrays."""
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

from eofs import standard
import collections
import numpy as np
import numpy.ma as ma

try:
    import dask.array
    has_dask = True
except ImportError:
    has_dask = False

from numpy.lib.stride_tricks import as_strided


class ExtendedEof(object):
    """Extended EOF analysis (`numpy` interface)"""

    def __init__(self, dataset, lag, weights=None, center=True, ddof=1):
        """Create an ExtendedEof instance.

        The EEOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *dataset*
            A `numpy.ndarray` or `numpy.ma.MaskedArray` with two or more
            dimensions containing the data to be analysed. The first
            dimension is assumed to represent time. Missing values are
            permitted, either in the form of a masked array, or
            `numpy.nan` values. Missing values must be constant with time
            (e.g., values of an oceanographic field over land).

        *lag*
            The number of timesteps to embed in the rows of the input matrix.
            Because window = lag + 1, lag must be greater than or equal to
            zero. A value of 0 is equivalent to standard Eof analysis.

        **Optional arguments:**

        *weights*
            An array of weights whose shape is compatible with those of the
            input array *dataset* (before *lag* is applied). The weights can
            have the same shape as *dataset* or a shape compatible with an
            array broadcast (i.e., the shape of the weights can match the
            rightmost parts of the shape of the input array *dataset*). If the
            input array *dataset* does not require weighting then the value
            *None* may be used. Defaults to *None* (no weighting).

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

        EEOF analysis with no weighting and lag 5::

            from eofs.extended.standard import ExtendedEof
            solver = ExtendedEof(data, 5)

        EEOF analysis of a data array with spatial dimensions that represent
        latitude and longitude with weighting. In this example the data array
        is dimensioned (ntime, nlat, nlon) with a lag of 5, and in order for
        the latitude weights to be broadcastable to this shape, an extra
        length-1 dimension is added to the end::

            from eofs.extended.standard import ExtendedEof
            import numpy as np
            latitude = np.linspace(-90, 90, 73)
            weights_array = np.cos(np.deg2rad(latitude))[:, np.newaxis]
            solver = ExtendedEof(data, 5, weights=weight_array)

        """
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise ValueError('the input data set must be '
                             'at least two dimensional')
        data = dataset.copy()
        # Store information about the shape/size of the input data.
        self._records = data.shape[0]
        self._originalshape = data.shape[1:]
        self._window = lag + 1
        channels = np.product(self._originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                # The broadcast_arrays call returns a list, so the second index
                # is retained, but also we want to remove the time dimension
                # from the weights so the the first index from the broadcast
                # array is taken.
                self._weights = np.broadcast_arrays(
                    data[0:1], weights)[1][0]
                data = data * self._weights
            except ValueError:
                raise ValueError('weight array dimensions are incompatible')
            except TypeError:
                raise TypeError('weights are not a valid type')
        else:
            self._weights = None
        # Remove the time mean of the input data unless explicitly told
        # not to by the "center" argument.
        self._centered = center
        if center:
            data = self._center(data)
        # Reshape to two dimensions (time, space) creating the design matrix.
        data = data.reshape([self._records, channels])
        # Get covariance matrix of eeof by passing input data
        cov_matrix_eeof = self._embed_dimension(data, self._window)
        # new channels dimension
        new_channels = self._window * channels
        self._solver = standard.Eof(cov_matrix_eeof,
                                    weights=None,
                                    center=False,
                                    ddof=ddof)
        self.neeofs = self._solver.neofs

    def _center(self, in_array):
        """Remove the mean of an array along the first dimension."""
        # Compute the mean along the first dimension.
        mean = in_array.mean(axis=0)
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

    def _embed_dimension(self, array, window):
        """
        Embed a given length window from the leading dimension of an array.

        **Arguments:**

        *array*
            A 2-dimensional (nxm) `numpy.ndarray` or `numpy.ma.MaskedArray`.

        *window*
            An integer specifying the length of the embedding window.

        **Returns:**

            A 2-dimenensional ((n-window+1) x (m*window)) `numpy.ndarray` or
            `numpy.ma.MaskedArray` which is a view on the input *array*.

        **Example:**

            data = np.arange(4*3).reshape(4, 3)
            >>> data
            array([[ 0,  1,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  8],
                   [ 9, 10, 11]])

            >>> _embed_dimension(data, window=2)
            array([[ 0,  1,  2,  3,  4,  5],
                   [ 3,  4,  5,  6,  7,  8],
                   [ 6,  7,  8,  9, 10, 11]])

            >>> _embed_dimension(data, window=3)
            array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
                   [ 3,  4,  5,  6,  7,  8,  9, 10, 11]])

            If window equals 1, the input array stays unchanged.

           >>> _embed_dimension(data, window=1)
            array([[ 0,  1,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  8],
                   [ 9, 10, 11]])

        References : A.Hannachi, 2004, "A Primer for EOF Analysis of Climate
             Data", Department of Meteorology, University of Reading Reading
             RG6 6BB, U.K. (page numbers 15-28)
        Link : http://eros.eas.gatech.edu/eas-8803/lectures/EOFs/eofprimer.pdf

        Author: Dr Andrew Dawson
        Date: 18-11-2013

        """
        if array.ndim != 2:
            raise ValueError('array must have exactly 2 dimensions')
        if window >= array.shape[0]:
            raise ValueError('embedding window must be shorter than the '
                             'first dimension of the array')
        n, _ = array.shape
        nwin = n - window + 1
        shape = (nwin, window) + array.shape[1:]

        strides = (array.strides[0], array.strides[0]) + array.strides[1:]
        windowed = as_strided(array, shape=shape, strides=strides)
        if ma.isMaskedArray(array):
            if array.mask is ma.nomask:
                windowed_mask = array.mask
            else:
                strides = ((array.mask.strides[0], array.mask.strides[0]) +
                           array.mask.strides[1:])
                windowed_mask = as_strided(array.mask, shape=shape,
                                           strides=strides)
            windowed = ma.array(windowed, mask=windowed_mask)
        out_shape = (nwin, window * array.shape[1])

        ret = windowed.reshape(out_shape)
        return ret

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled PCs (default).
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of PCs requested is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *pcs*
            An array where the columns are the ordered PCs.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        return self._solver.pcs(pcscaling, npcs)

    def eeofs(self, eofscaling=0, neeofs=None):
        """Extended Empirical Orthogonal Functions (EEOFs).

        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EEOFs. The following values are
            accepted:

            * *0* : Un-scaled EEOFs (default).
            * *1* : EEOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EEOFs are multiplied by the square-root of their
              eigenvalues.

        *neeofs*
            Number of EEOFs to return. Defaults to all EEOFs. If the
            number of EEOFs requested is more than the number that are
            available, then all available EEOFs will be returned.

        **Returns:**

        *eofs*
            An array with the ordered EEOFs along the first dimension.

        **Examples:**

        All EEOFs with no scaling::

            eeofs = solver.eeofs()

        The leading EEOF with scaling applied::

            eeof1 = solver.eeofs(neeofs=1, eofscaling=1)

        """
        rval = self._solver.eofs(eofscaling, neeofs)
        neeofs = neeofs or self.neeofs
        # Reshape so the lag becomes it's own dimension
        return rval.reshape(((neeofs, self._window,) + self._originalshape))

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EEOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues. If the number of eigenvalues requested is more
            than the number that are available, then all available
            eigenvalues will be returned.

        **Returns:**

        *eigenvalues*
            An array containing the eigenvalues arranged largest to
            smallest.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        return self._solver.eigenvalues(neigs)

    def varianceFraction(self, neigs=None):
        """Fractional EEOF mode variances.

        The fraction of the total variance explained by each EEOF mode,
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
            An array containing the fractional variances.

        **Examples:**

        The fractional variance represented by each EEOF mode::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first EEOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        return self._solver.varianceFraction(neigs)

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EEOFs.

        As the reconstruction yields the extended data-matrix, the mean
        for every location is calculated.

        If weights were passed to the `ExtendedEof` instance the returned
        reconstructed field will automatically have this weighting removed.
        Otherwise the returned field will have the same weighting as the
        `ExtendedEof` input *dataset*.

        **Argument:**

        *neofs*
            Number of EEOFs to use for the reconstruction. If the
            number of EEOFs requested is more than the number that are
            available, then all available EEOFs will be used for the
            reconstruction. Alternatively this argument can be an
            iterable of mode numbers (where the first mode is 1) in
            order to facilitate reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            An array the same shape as the `ExtendedEof` input *dataset*
            contaning the reconstruction using *neofs* EEOFs.

        **Examples:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        rval = self._solver.reconstructedField(neofs)
        # Take the mean of the values for one location
        channels = np.product(self._originalshape)
        rval = rval[::-1].reshape(rval.shape[0], self._window, channels)
        rval = np.array([rval.diagonal(i).mean(axis=1)
                         for i in range(1-rval.shape[0], self._window)])
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        rval = rval.reshape((self._records,) + self._originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self._weights is not None:
            rval = rval / self._weights
        # Return the reconstructed field.
        return rval

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
            Defaults to typical errors for all eigenvalues. If the
            number of eigenvalues requested is more than the number that
            are available, then typical errors for all available
            eigenvalues will be returned.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `MultivariateEof.varianceFraction`. If *False*
            then no scaling is done. Defaults to *False* (no scaling).

        **Returns:**

        *errors*
            An array containing the typical errors.

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Examples:**

        Typical errors for all eigenvalues::

            errors = solver.northTest()

        Typical errors for the first 5 eigenvalues scaled by the sum of
        the eigenvalues::

            errors = solver.northTest(neigs=5, vfscaled=True)

        """
        return self._solver.northTest(neigs, vfscaled)
