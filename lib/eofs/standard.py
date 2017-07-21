"""EOF analysis for data in `numpy` arrays."""
# (c) Copyright 2000 Jon Saenz, Jesus Fernandez and Juan Zubillaga.
# (c) Copyright 2010-2016 Andrew Dawson. All Rights Reserved.
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
import warnings

import numpy as np
import numpy.ma as ma

from .tools.standard import correlation_map, covariance_map


class Eof(object):
    """EOF analysis (`numpy` interface)"""

    def __init__(self, dataset, weights=None, center=True, ddof=1):
        """Create an Eof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *dataset*
            A `numpy.ndarray` or `numpy.ma.MaskedArray` with two or more
            dimensions containing the data to be analysed. The first
            dimension is assumed to represent time. Missing values are
            permitted, either in the form of a masked array, or
            `numpy.nan` values. Missing values must be constant with time
            (e.g., values of an oceanographic field over land).

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

            from eofs.standard import Eof
            solver = Eof(data)

        EOF analysis of a data array with spatial dimensions that
        represent latitude and longitude with weighting. In this example
        the data array is dimensioned (ntime, nlat, nlon), and in order
        for the latitude weights to be broadcastable to this shape, an
        extra length-1 dimension is added to the end::

            from eofs.standard import Eof
            import numpy as np
            latitude = np.linspace(-90, 90, 73)
            weights_array = np.cos(np.deg2rad(latitude))[:, np.newaxis]
            solver = Eof(data, weights=weight_array)

        """
        # Store the input data in an instance variable.
        if dataset.ndim < 2:
            raise ValueError('the input data set must be '
                             'at least two dimensional')
        self._data = dataset.copy()
        # Check if the input is a masked array. If so fill it with NaN.
        try:
            self._data = self._data.filled(fill_value=np.nan)
            self._filled = True
        except AttributeError:
            self._filled = False
        # Store information about the shape/size of the input data.
        self._records = self._data.shape[0]
        self._originalshape = self._data.shape[1:]
        channels = np.product(self._originalshape)
        # Weight the data set according to weighting argument.
        if weights is not None:
            try:
                # The broadcast_arrays call returns a list, so the second index
                # is retained, but also we want to remove the time dimension
                # from the weights so the the first index from the broadcast
                # array is taken.
                self._weights = np.broadcast_arrays(
                    self._data[0:1], weights)[1][0]
                self._data = self._data * self._weights
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
            self._data = self._center(self._data)
        # Reshape to two dimensions (time, space) creating the design matrix.
        self._data = self._data.reshape([self._records, channels])
        # Find the indices of values that are not missing in one row. All the
        # rows will have missing values in the same places provided the
        # array was centered. If it wasn't then it is possible that some
        # missing values will be missed and the singular value decomposition
        # will produce not a number for everything.
        if not self._valid_nan(self._data):
            raise ValueError('missing values detected in different '
                             'locations at different times')
        nonMissingIndex = np.where(np.logical_not(np.isnan(self._data[0])))[0]
        # Remove missing values from the design matrix.
        dataNoMissing = self._data[:, nonMissingIndex]
        # Compute the singular value decomposition of the design matrix.
        try:
            A, Lh, E = np.linalg.svd(dataNoMissing, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            raise ValueError('error encountered in SVD, check that missing '
                             'values are in the same places at each time and '
                             'that all the values are not missing')
        # Singular values are the square-root of the eigenvalues of the
        # covariance matrix. Construct the eigenvalues appropriately and
        # normalize by N-ddof where N is the number of observations. This
        # corresponds to the eigenvalues of the normalized covariance matrix.
        self._ddof = ddof
        normfactor = float(self._records - self._ddof)
        self._L = Lh * Lh / normfactor
        # Store the number of eigenvalues (and hence EOFs) that were actually
        # computed.
        self.neofs = len(self._L)
        # Re-introduce missing values into the eigenvectors in the same places
        # as they exist in the input maps. Create an array of not-a-numbers
        # and then introduce data values where required. We have to use the
        # astype method to ensure the eigenvectors are the same type as the
        # input dataset since multiplication by np.NaN will promote to 64-bit.
        self._flatE = np.ones([self.neofs, channels],
                              dtype=self._data.dtype) * np.NaN
        self._flatE = self._flatE.astype(self._data.dtype)
        self._flatE[:, nonMissingIndex] = E
        # Remove the scaling on the principal component time-series that is
        # implicitily introduced by using SVD instead of eigen-decomposition.
        # The PCs may be re-scaled later if required.
        self._P = A * Lh

    def _center(self, in_array):
        """Remove the mean of an array along the first dimension."""
        # Compute the mean along the first dimension.
        mean = in_array.mean(axis=0)
        # Return the input array with its mean along the first dimension
        # removed.
        return (in_array - mean)

    def _valid_nan(self, in_array):
        inan = np.isnan(in_array)
        return (inan.any(axis=0) == inan.all(axis=0)).all()

    def _verify_projection_shape(self, proj_field, proj_space_shape):
        """Verify that a field can be projected onto another"""
        eof_ndim = len(proj_space_shape) + 1
        if eof_ndim - proj_field.ndim not in (0, 1):
            raise ValueError('field has the wrong number of dimensions '
                             'to be projected onto EOFs')
        if proj_field.ndim == eof_ndim:
            check_shape = proj_field.shape[1:]
        else:
            check_shape = proj_field.shape
        if check_shape != proj_space_shape:
            raise ValueError('field has the wrong shape to be projected '
                             ' onto the EOFs')

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
        slicer = slice(0, npcs)
        if pcscaling == 0:
            # Do not scale.
            return self._P[:, slicer].copy()
        elif pcscaling == 1:
            # Divide by the square-root of the eigenvalue.
            return self._P[:, slicer] / np.sqrt(self._L[slicer])
        elif pcscaling == 2:
            # Multiply by the square root of the eigenvalue.
            return self._P[:, slicer] * np.sqrt(self._L[slicer])
        else:
            raise ValueError('invalid PC scaling option: '
                             '{!s}'.format(pcscaling))

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

        *eofs*
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs with no scaling::

            eofs = solver.eofs()

        The leading EOF with scaling applied::

            eof1 = solver.eofs(neofs=1, eofscaling=1)

        """
        if neofs is None or neofs > self.neofs:
            neofs = self.neofs
        slicer = slice(0, neofs)
        neofs = neofs or self.neofs
        flat_eofs = self._flatE[slicer].copy()
        if eofscaling == 0:
            # No modification. A copy needs to be returned in case it is
            # modified. If no copy is made the internally stored eigenvectors
            # could be modified unintentionally.
            rval = flat_eofs
        elif eofscaling == 1:
            # Divide by the square-root of the eigenvalues.
            rval = flat_eofs / np.sqrt(self._L[slicer])[:, np.newaxis]
        elif eofscaling == 2:
            # Multiply by the square-root of the eigenvalues.
            rval = flat_eofs * np.sqrt(self._L[slicer])[:, np.newaxis]
        else:
            raise ValueError('invalid eof scaling option: '
                             '{!s}'.format(eofscaling))
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval.reshape((neofs,) + self._originalshape)

    def eofsAsCorrelation(self, neofs=None):
        """Correlation map EOFs.

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
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCorrelation()

        The leading EOF::

            eof1 = solver.eofsAsCorrelation(neofs=1)

        """
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, pcscaling=1)
        # Compute the correlation of the PCs with the input field.
        c = correlation_map(
            pcs,
            self._data.reshape((self._records,) + self._originalshape))
        # The results of the correlation_map function will be a masked array.
        # For consistency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """Covariance map EOFs.

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
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCovariance()

        The leading EOF::

            eof1 = solver.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs::

            eof1 = solver.eofsAsCovariance(neofs=1, pcscaling=0)

        """
        pcs = self.pcs(npcs=neofs, pcscaling=pcscaling)
        # Divide the input data by the weighting (if any) before computing
        # the covariance maps.
        data = self._data.reshape((self._records,) + self._originalshape)
        if self._weights is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                data /= self._weights
        c = covariance_map(pcs, data, ddof=self._ddof)
        # The results of the covariance_map function will be a masked array.
        # For consitsency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

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
        # Create a slicer and use it on the eigenvalue array. A copy must be
        # returned in case the slicer takes all elements, in which case a
        # reference to the eigenvalue array is returned. If this is modified
        # then the internal eigenvalues array would then be modified as well.
        slicer = slice(0, neigs)
        return self._L[slicer].copy()

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

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first EOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        # Return the array of eigenvalues divided by the sum of the
        # eigenvalues.
        slicer = slice(0, neigs)
        return self._L[slicer] / self._L.sum()

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
        # Return the sum of the eigenvalues.
        return self._L.sum()

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
            returned by `Eof.varianceFraction`. If *False* then no
            scaling is done. Defaults to *False* (no scaling).

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
        slicer = slice(0, neigs)
        # Compute the factor that multiplies the eigenvalues. The number of
        # records is assumed to be the number of realizations of the field.
        factor = np.sqrt(2.0 / self._records)
        # If requested, allow for scaling of the eigenvalues by the total
        # variance (sum of the eigenvalues).
        if vfscaled:
            factor /= self._L.sum()
        # Return the typical errors.
        return self._L[slicer] * factor

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the `Eof` instance the returned
        reconstructed field will automatically have this weighting
        removed. Otherwise the returned field will have the same
        weighting as the `Eof` input *dataset*.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be used for the
            reconstruction. Alternatively this argument can be an
            iterable of mode numbers (where the first mode is 1) in
            order to facilitate reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            An array the same shape as the `Eof` input *dataset*
            contaning the reconstruction using *neofs* EOFs.

        **Examples:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        # Determine how the PCs and EOFs will be selected.
        if isinstance(neofs, collections.Iterable):
            modes = [m - 1 for m in neofs]
        else:
            modes = slice(0, neofs)
        # Project principal components onto the EOFs to compute the
        # reconstructed field.
        rval = np.dot(self._P[:, modes], self._flatE[modes])
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        rval = rval.reshape((self._records,) + self._originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self._weights is not None:
            rval = rval / self._weights
        # Return the reconstructed field.
        if self._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a data set, projects it onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *field*
            A `numpy.ndarray` or `numpy.ma.MaskedArray` with two or more
            dimensions containing the data to be projected onto the
            EOFs. It must have the same corresponding spatial dimensions
            (including missing values in the same places) as the `Eof`
            input *dataset*. *field* may have a different length time
            dimension to the `Eof` input *dataset* or no time dimension
            at all.

        **Optional arguments:**

        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then the field will be projected onto all
            available EOFs.

        *eofscaling*
            Set the scaling of the EOFs that are projected onto. The
            following values are accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then *field* is weighted using the same weights
            used for the EOF analysis prior to projection. If *False*
            then no weighting is applied. Defaults to *True* (weighting
            is applied). Generally only the default setting should be
            used.

        **Returns:**

        *pseudo_pcs*
            An array where the columns are the ordered pseudo-PCs.

        **Examples:**

        Project a data set onto all EOFs::

            pseudo_pcs = solver.projectField(data)

        Project a data set onto the four leading EOFs::

            pseudo_pcs = solver.projectField(data, neofs=4)

        """
        # Check that the shape/dimension of the data set is compatible with
        # the EOFs.
        self._verify_projection_shape(field, self._originalshape)
        input_ndim = field.ndim
        eof_ndim = len(self._originalshape) + 1
        # Create a slice object for truncating the EOFs.
        slicer = slice(0, neofs)
        # If required, weight the data set with the same weighting that was
        # used to compute the EOFs.
        field = field.copy()
        if weighted:
            wts = self.getWeights()
            if wts is not None:
                field = field * wts
        # Fill missing values with NaN if this is a masked array.
        try:
            field = field.filled(fill_value=np.nan)
        except AttributeError:
            pass
        # Flatten the data set into [time, space] dimensionality.
        if eof_ndim > input_ndim:
            field = field.reshape((1,) + field.shape)
        records = field.shape[0]
        channels = np.product(field.shape[1:])
        field_flat = field.reshape([records, channels])
        # Locate the non-missing values and isolate them.
        if not self._valid_nan(field_flat):
            raise ValueError('missing values detected in different '
                             'locations at different times')
        nonMissingIndex = np.where(np.logical_not(np.isnan(field_flat[0])))[0]
        field_flat = field_flat[:, nonMissingIndex]
        # Locate the non-missing values in the EOFs and check they match those
        # in the data set, then isolate the non-missing values.
        eofNonMissingIndex = np.where(
            np.logical_not(np.isnan(self._flatE[0])))[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise ValueError('field and EOFs have different '
                             'missing value locations')
        eofs_flat = self._flatE[slicer, eofNonMissingIndex]
        if eofscaling == 1:
            eofs_flat /= np.sqrt(self._L[slicer])[:, np.newaxis]
        elif eofscaling == 2:
            eofs_flat *= np.sqrt(self._L[slicer])[:, np.newaxis]
        # Project the data set onto the EOFs using a matrix multiplication.
        projected_pcs = np.dot(field_flat, eofs_flat.T)
        if eof_ndim > input_ndim:
            # If an extra dimension was introduced, remove it before returning
            # the projected PCs.
            projected_pcs = projected_pcs[0]
        return projected_pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights*
            An array containing the analysis weights.

        **Example:**

        The weights used for the analysis::

            weights = solver.getWeights()

        """
        return self._weights
