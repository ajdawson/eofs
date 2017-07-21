"""Multivariate EOF analysis for `numpy` array data."""
# (c) Copyright 2013-2016 Andrew Dawson. All Rights Reserved.
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

import numpy as np

from eofs import standard


class MultivariateEof(object):
    """Multivariate EOF analysis (`numpy` interface)"""

    def __init__(self, datasets, weights=None, center=True, ddof=1):
        """Create a MultivariateEof instance.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *datasets*
            A list/tuple containing one or more `numpy.ndarray` or
            `numpy.ma.MaskedArray` objects, each with two or more
            dimensions, containing the data to be analysed. The first
            dimension of each array is assumed to represent time.
            Missing values are permitted, either in the
            form of masked arrays, or `numpy.nan` values. Missing values
            must be constant with time in each field (e.g., values of an
            oceanographic field over land).

        **Optional arguments:**

        *weights*
            A sequence of arrays of weights whose shapes are compatible
            with those of the input arrays in *datasets*, one array for
            each array in *datasets*. The weights can have the same
            shape as the corresponding array in *datasets* or a shape
            compatible with an array broadcast operation (i.e., the
            shape of the weights can match the rightmost parts of the
            shape of the corresponding array in *datasets*). The value
            *None* indicates that the corresponding array in *datasets*
            should not be weighted. If none of the input data sets
            require weighting then the single value *None* may be used.
            Defaults to *None* (no weighting for any data set).

        *center*
            If *True*, the mean along the first axis of each array in
            *datasets* (the time-mean) will be removed prior to
            analysis. If *False*, the mean along the first axis will not
            be removed. Defaults to *True* (mean is removed).

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
            An `MultivariateEof` instance.

        **Examples:**

        EOF analysis of three variables with no weighting::

            from eofs.multivariate.standard import MultivariateEof
            solver = MultivariateEof([olr, u200, u850])

        EOF analysis of two variables, one without weighting and one
        with. In this example the data arrays are dimensioned
        (ntime, nlat, nlon), and in order for the latitude weights used
        for the second array to be broadcastable to this shape, an extra
        length-1 dimension is added to the end::

            from eofs.multivariate.standard import MultivariateEof
            import numpy as np
            latitude = np.linspace(-90, 90, 73)
            weights_array = np.sqrt(
                np.cos(np.deg2rad(latitude)))[:, np.newaxis]
            solver = MultivariateEof([var1, var2],
                                      weights=[None, weights_array])

        """
        self._ndata = len(datasets)
        data, info = self._merge_fields(datasets)
        self._shapes = info['shapes']
        self._slicers = info['slicers']
        weight_option = self._merge_weights(datasets, weights)
        self._solver = standard.Eof(data,
                                    weights=weight_option,
                                    center=center,
                                    ddof=ddof)
        #: Number of EOFs in the solution.
        self.neofs = self._solver.neofs

    def _merge_fields(self, fields):
        """Merge multiple fields into one field.

        Flattens each field to (time, space) dimensionality and
        concatenates to form one field. Returns the merged array
        and a dictionary {'shapes': [], 'slicers': []} where the entry
        'shapes' is a list of the input array shapes minus the time
        dimension ans the entry 'slicers' is a list of `slice` objects
        that can be used to select each individual field from the merged
        array.

        """
        info = {'shapes': [], 'slicers': []}
        islice = 0
        for field in fields:
            info['shapes'].append(field.shape[1:])
            channels = np.prod(field.shape[1:])
            info['slicers'].append(slice(islice, islice + channels))
            islice += channels
        try:
            merged = np.concatenate(
                [field.reshape([field.shape[0], np.prod(field.shape[1:])])
                 for field in fields], axis=1)
        except ValueError:
            raise ValueError('all fields must have the same first dimension')
        return merged, info

    def _merge_weights(self, fields, weights):
        """Merge multiple sets of weights into a single weights array.

        """
        if weights is None:
            return None
        if len(weights) != len(fields):
            raise ValueError('number of weights is incorrect, '
                             'expecting {:d} but got {:d}'.format(
                                 self._ndata, len(weights)))
        if all([w is None for w in weights]):
            # If all the entries are None, then just use the single value
            # None to indicate no weights are required.
            return None

        def _broadcast_weights(field, weight):
            """
            Broadcasts a weights array to the spatial shape of a field,
            but with a singleton leading dimension.

            """
            shape = np.prod(field.shape[1:])
            if weight is None:
                return np.ones([1, shape])
            # Broadcast the weights to the full shape of the field.
            bcast = np.broadcast_arrays(field, weight)[1]
            # Return weights with the same shape as the field except for the
            # leading dimension which is singleton. This dimension is removed
            # by the EOF solver anyway.
            return bcast[0].reshape([1, shape])

        try:
            # If at least one weight is an array then all weights should be
            # converted to arrays (by using arrays of 1.0 for unweighted
            # fields) of full spatial shape and concatenate them.
            weight_option = np.concatenate([_broadcast_weights(f, w)
                                            for f, w in zip(fields, weights)],
                                           axis=1)
        except ValueError:
            raise ValueError('one or more weights arrays have '
                             'incompatible shapes')
        return weight_option

    def _unwrap(self, modes):
        """Split a returned mode field into component parts."""
        nmodes = modes.shape[0]
        modeset = [modes[:, slicer].reshape((nmodes,) + shape)
                   for slicer, shape in zip(self._slicers, self._shapes)]
        return modeset

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

    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).

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

        *eofs_list*
            A list of arrays containing the ordered EOFs for each
            variable along the first dimension.

        **Examples:**

        All EOFs with no scaling::

            eofs_list = solver.eofs()

        The leading EOF with scaling applied::

            eof1_list = solver.eofs(neofs=1, eofscaling=1)

        """
        modes = self._solver.eofs(eofscaling, neofs)
        return self._unwrap(modes)

    def eofsAsCorrelation(self, neofs=None):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        correlation between the principal component time series (PCs)
        and the each data set in the `MultivariateEof` input *datasets*.

        .. note::

            These are not related to the EOFs computed from the
            correlation matrix.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs_list*
            A list of arrays containing the ordered EOFs for each
            variable along the first dimension.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCorrelation()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCorrelation(neofs=1)

        """
        modes = self._solver.eofsAsCorrelation(neofs)
        return self._unwrap(modes)

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        covariance between the principal component time series (PCs)
        and the each data set in the `MultivariateEof` input *datasets*.

        **Optional argument:**

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

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of PCs requested is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *eofs_list*
            A list of arrays containing the ordered EOFs for each
            variable along the first dimension.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCovariance()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCovariance(neofs=1)

        """
        modes = self._solver.eofsAsCovariance(neofs, pcscaling)
        return self._unwrap(modes)

    def eigenvalues(self, neigs=None):
        """
        Eigenvalues (decreasing variances) associated with each EOF
        mode.

        Returns the eigenvalues in an array.

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
            An array containing the fractional variances.

        **Examples:**

        The fractional variance represented by each EOF mode::

            variance_fraction = solver.varianceFraction()

        The fractional variance represented by the first EOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        return self._solver.varianceFraction(neigs)

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).

        **Returns:**

        *total_variance*
            A scalar value.

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

    def reconstructedField(self, neofs):
        """Reconstructed data sets based on a subset of EOFs.

        If weights were passed to the `MultivariateEof` instance the
        returned reconstructed fields will automatically have this
        weighting removed. Otherwise each returned field will have the
        same weighting as the corresponding  array in the
        `MultivariateEof` input *datasets*.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be used for the
            reconstruction. Alternatively this argument can be an
            iterable of mode numbers (where the first mode is 1) in
            order to facilitate reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction_list*
            A list of arrays with the same shapes as the arrays in the
            `MultivariateEof` input *datasets* contaning the
            reconstructions using *neofs* EOFs.

        **Example:**

        Reconstruct the input data sets using 3 EOFs::

            reconstruction_list = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction_list = solver.reconstuctedField([1, 2, 5])

        """
        rf = self._solver.reconstructedField(neofs)
        return self._unwrap(rf)

    def projectField(self, fields, neofs=None, eofscaling=0, weighted=True):
        """Project a set of fields onto the EOFs.

        Given a set of data sets, projects them onto the EOFs to
        generate a corresponding set of pseudo-PCs.

        **Argument:**

        *fields*
            A list/tuple containing one or more `numpy.ndarray` or
            `numpy.ma.MaskedArray` objects, each with two or more
            dimensions, containing the fields to be projected onto the
            EOFs. Each field must have the same spatial dimensions
            (including missing values in the same places) as the
            corresponding data set in the `MultivariateEof` input
            *datasets*. The data sets may have different length time
            dimensions to the `MultivariateEof` inputs *datasets* or no
            time dimension at all, but this must be consistent for all
            fields.

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
            * *1* : EOFs are divided by the square-root of their
              eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then each field in *fields* is weighted using the
            same weights used for the EOF analysis prior to projection.
            If *False* then no weighting is applied. Defaults to *True*
            (weighting is applied). Generally only the default setting
            should be used.

        **Returns:**

        *pseudo_pcs*
            An array where the columns are the ordered pseudo-PCs.

        **Examples:**

        Project a data set onto all EOFs::

            pseudo_pcs = solver.projectField([field1, field2])

        Project a data set onto the four leading EOFs::

            pseudo_pcs = solver.projectField([field1, field2], neofs=4)

        """
        if len(fields) != self._ndata:
            raise ValueError('number of fields is incorrect, expecting {:d} '
                             'but got {:d}'.format(self._ndata, len(fields)))
        # Make sure the input fields have valid shapes/dimensionalities.
        for field, shape in zip(fields, self._shapes):
            self._solver._verify_projection_shape(field, shape)

        # Check for a time dimension in each field.
        def time_check(field, shape):
            return len(shape) + 1 == field.ndim
        has_time = [time_check(f, e) for f, e in zip(fields, self._shapes)]
        if [t for t in has_time if t != has_time[0]]:
            raise ValueError('detected a mixture of fields with and without '
                             'time dimensions, this is not allowed')
        has_time = has_time[0]
        if has_time:
            data, info = self._merge_fields(fields)
        else:
            data, info = self._merge_fields([f.reshape((1,) + f.shape)
                                            for f in fields])
        # Do the projection.
        pcs = self._solver.projectField(data,
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        if not has_time:
            # Eliminate the time dimension if it was added artificially.
            pcs = pcs[0]
        return pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights_list*
            A list of arrays containing the analysis weights for each
            variable.

        **Example:**

        The weights used for the analysis::

            weights_list = solver.getWeights()

        """
        w = self._solver.getWeights()
        if w is None:
            return None
        return self._unwrap(w.reshape((1,) + w.shape))[0]
