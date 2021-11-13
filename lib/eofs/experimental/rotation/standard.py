"""Rotation of EOFs."""
# (c) Copyright 2013 Andrew Dawson. All Rights Reserved.
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
from __future__ import absolute_import
import collections
import warnings

import numpy as np
import numpy.ma as ma

from .kernels import KERNEL_MAPPING
from ...tools.standard import correlation_map, covariance_map


class Rotator(object):
    """Rotate EOFs from the standard `numpy` interface."""

    def __init__(self, solver, neofs, method='varimax', scaled=True,
                 kernelargs=None):
        """Create an EOF rotator.

        **Arguments:**

        *solver*
            An `~eofs.standard.Eof` instance that can generate the EOFs
            to be rotated.

        *neofs*
            Number of EOFs to use for the rotation.

        **Keyword arguments:**

        *method*
            The method to use for rotation. Currently the only accepted
            value is 'varimax'.

        *scaled*
            If *True* the EOFs are multiplied by the square-root of
            their eigenvalue before the rotation. If *False* the
            orthonormal EOFs are used for the rotation. Defaults to
            *True*.

        **Examples:**

        A varimax rotator based on the first 10 scaled EOFs::

            solver = Eof(data, weights=weights_array)
            rotator = Rotator(solver, 10)

        A varimax rotator based on the first 5 un-scaled EOFs::

            solver = Eof(data, weights=weights_array)
            rotator = Rotator(solver, 5, scaled=False)

        """
        # This is just a reference, not a copy, so it should be lightweight.
        self._solver = solver
        self._scaled = scaled
        self.neofs = neofs
        # Retrieve the required quantities from the solver.
        eofscaling = 2 if scaled else 0
        eofs = self._solver.eofs(eofscaling=eofscaling, neofs=neofs)
        # Remove metadata and store it for later use.
        eofs, self._eofmetadata = self.strip_metadata(eofs)
        # Remove missing values and metadata and reshape to [neofs, nspace].
        eofs, self._eofinfo = self._to2d(eofs)
        # Do the rotation using the appropriate kernel.
        kwargs = {}
        if kernelargs is not None:
            try:
                kwargs.update(kernelargs)
            except TypeError:
                raise TypeError('kernel arguments must be a '
                                'dictionary of keyword arguments')
        try:
            self._eofs_rot, R = KERNEL_MAPPING[method.lower()](eofs, **kwargs)
            self._rotation_matrix = R
        except KeyError:
            raise ValueError("unknown rotation method: '{!s}'".format(method))
        # Compute variances of the rotated EOFs as these are used by several
        # methods.
        self._eofs_rot_var = (self._eofs_rot ** 2).sum(axis=1)
        self._var_idx = np.argsort(self._eofs_rot_var)[::-1]
        # Reorder rotated EOFs according to their variance.
        self._eofs_rot = self._eofs_rot[self._var_idx]
        self._eofs_rot_var = self._eofs_rot_var[self._var_idx]

    def eofs(self, neofs=None, renormalize=False):
        """Rotated empirical orthogonal functions (EOFs).

        **Optional arguments:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        *renormalize*
            If *True* and the rotation was based on scaled EOFs then the
            scaling will be removed and the returned rotated EOFs will
            be scaled to unit variance. If *False* the returned rotated
            EOFs will retain the scaling.

        **Returns:**

        *eofs*
            An array with the ordered rotated EOFs along the first
            dimension.

        **Examples:**

        All rotated EOFs with scaling::

            rotator = Rotator(solver, 10, scaled=True)
            rotated_eofs = rotator.eofs()

        All rotated EOFs with scaling removed::

            rotator = Rotator(solver, 10, scaled=True)
            rotated_eofs = rotator.eofs(renormalize=True)

        The leading rotated EOF with scaling::

            rotator = Rotator(solver, 10, scaled=True)
            rotated_eofs = rotator.eofs(neofs=1)

        """
        # Determine the correct slice.
        if (neofs is None) or neofs > self.neofs:
            neofs = self.neofs
        slicer = slice(0, neofs)
        # Optionally renormalize.
        if renormalize and self._scaled:
            eofs_rot = self._eofs_rot / \
                np.sqrt(self._eofs_rot_var)[:, np.newaxis]
        else:
            eofs_rot = self._eofs_rot
        eofs_rot = self._from2d(eofs_rot, self._eofinfo)[slicer]
        eofs_rot = self.apply_metadata(eofs_rot, self._eofmetadata)
        return eofs_rot

    def varianceFraction(self, neigs=None):
        """Fractional rotated EOF mode variances.

        The fraction of the total variance explained by each rotated EOF
        mode, values between 0 and 1 inclusive. Only meaningful if the
        *scaled* option was set to *True* when creating the `Rotator`.

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

        The fractional variance represented by each rotated EOF mode::

            rotator = Rotator(solver, 10, scaled=True)
            variance_fractions = rotator.varianceFraction()

        The fractional variance represented by the first rotated EOF mode::

            rotator = Rotator(solver, 10, scaled=True)
            variance_fractions = rotator.varianceFraction(neigs=1)

        """
        if (neigs is None) or (neigs > self.neofs):
            neigs = self.neofs
        slicer = slice(0, neigs)
        # Compute fractions of variance accounted for by each rotated mode.
        eigenvalues = self._solver.eigenvalues(neigs=neigs)
        variance_fractions = self._solver.varianceFraction(neigs=neigs)
        ev, ev_metadata = self.strip_metadata(eigenvalues)
        vf, vf_metadata = self.strip_metadata(variance_fractions)
        if self._scaled:
            ratio = vf[0] / ev[0]
            vf_rot = self._eofs_rot_var[slicer] * ratio
        else:
            vf_rot = np.array([1. / float(self._eofs_rot.shape[1])] * neigs)
        vf_rot = self.apply_metadata(vf_rot, vf_metadata)
        return vf_rot

    def eigenvalues(self, neigs=None):
        """Variances of the rotated EOFs."""
        if neigs > self.neofs or neigs is None:
            neigs = self.neofs
        slicer = slice(0, neigs)
        eigenvalues = self._solver.eigenvalues(neigs=neigs)
        ev, ev_metadata = self.strip_metadata(eigenvalues)
        variances = self._eofs_rot_var[slicer]
        variances = self.apply_metadata(variances, ev_metadata)
        return variances

    def pcs(self, npcs=None, normalized=False):
        """Principal component time series (PCs).

        The PC time series associated with the rotated EOFs.

        **Optional arguments:**

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of PCs requested is more than the number that are
            available, then all available PCs will be returned.

        *normalized*
            If *True* the returned PCs are scaled to unit variance. If
            *False* no scaling is done. Defaults to *False*.

        **Returns:**

        *pcs*
            An array where the columns are the ordered PCs.

        **Examples:**

        All un-scaled PCs::

            pcs = rotator.pcs()

        First 3 PCs scaled to unit variance::

            pcs = rotator.pcs(npcs=3, normalized=True)

        """
        # Determine the correct slice.
        if (npcs is None) or (npcs > self.neofs):
            npcs = self.neofs
        slicer = slice(0, npcs)
        # Compute the PCs:
        # 1. Obtain the non-rotated PCs.
        pcs = self._solver.pcs(npcs=self.neofs, pcscaling=1)
        # 2. Apply the rotation matrix to standardized PCs.
        pcs = np.dot(pcs, self._rotation_matrix)
        # 3. Reorder according to variance.
        pcs = pcs[:, self._var_idx]
        if not normalized:
            # Optionally scale by square root of variance of rotated EOFs.
            pcs *= np.sqrt(self._eofs_rot_var)
        # Select only the required PCs.
        pcs = pcs[:, slicer]
        # Collect the metadata used for PCs by the solver and apply it to
        # these PCs.
        _, pcs_metadata = self.strip_metadata(self._solver.pcs(npcs=npcs))
        pcs = self.apply_metadata(pcs, pcs_metadata)
        return pcs

    def eofsAsCorrelation(self, neofs=None):
        """Correlation map rotated EOFs.

        Rotated empirical orthogonal functions (EOFs) expressed as the
        correlation between the rotated principal component time series (PCs)
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

            eofs = rotator.eofsAsCorrelation()

        The leading EOF::

            eof1 = rotator.eofsAsCorrelation(neofs=1)

        """
        # Get original dimensions of data
        records = self._solver._records
        originalshape = self._solver._originalshape
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, normalized=True)
        # Compute the correlation of the PCs with the input field.
        c = correlation_map(
            pcs,
            self._solverdata().reshape((records,) + originalshape))
        # The results of the correlation_map function will be a masked array.
        # For consistency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._solver._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def eofsAsCovariance(self, neofs=None, normalized=True):
        """Covariance map rotated EOFs.

        Rotated empirical orthogonal functions (EOFs) expressed as the
        covariance between the rotated principal component time series (PCs)
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
        *normalized*
            If *True* the PCs used to compute covariance are scaled to
            unit variance. If *False* no scaling is done.
            Defaults to *True* which is the same as scaling option *1*
            for non-rotated covariance maps.

        **Returns:**

        *eofs*
            An array with the ordered EOFs along the first dimension.

        **Examples:**

        All EOFs::

            eofs = rotator.eofsAsCovariance()

        The leading EOF::

            eof1 = rotator.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs::

            eof1 = rotator.eofsAsCovariance(neofs=1, normalized=False)

        """
        # Get original dimensions of data
        records = self._solver._records
        originalshape = self._solver._originalshape
        # Retrieve the specified number of PCs.
        pcs = self.pcs(npcs=neofs, normalized=normalized)
        # Divide the input data by the weighting (if any) before computing
        # the covariance maps.
        data = self._solverdata().reshape((records,) + originalshape)
        if self._solver._weights is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                data /= self._solver._weights
        c = covariance_map(pcs, data, ddof=self._solver._ddof)
        # The results of the covariance_map function will be a masked array.
        # For consitsency with other return values, this is converted to a
        # numpy array filled with numpy.nan.
        if not self._solver._filled:
            c = c.filled(fill_value=np.nan)
        return c

    def reconstuctedField(self, neofs):
        """Reconstructed data field based on a subset of rotated EOFs.

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

        Reconstruct the input field using 3 rotated EOFs::

            reconstruction = rotator.reconstructedField(3)

        Reconstruct the input field using rotated EOFs 1, 2 and 5::

            reconstruction = rotator.reconstuctedField([1, 2, 5])

        """
        # Determine how the PCs and EOFs will be selected.
        if isinstance(neofs, collections.Iterable):
            modes = [m - 1 for m in neofs]
        else:
            modes = slice(0, neofs)
        # Create array containing rotated EOFs including not a number entries
        # of original input data.
        originalshape = self._solver._originalshape
        channels = np.product(originalshape)
        nan_idx = np.isnan(self._solver._flatE).all(axis=0)
        L = self._eofs_rot[modes]
        neofs = L.shape[0]
        loadings = np.zeros((neofs, channels)) * np.nan
        loadings[:, ~nan_idx] = L
        # Project principal components onto the rotated EOFs to
        # compute the reconstructed field.
        P = self.pcs(npcs=None, normalized=True)[:, modes]
        rval = np.dot(P, loadings)
        # Reshape the reconstructed field so it has the same shape as the
        # input data set.
        records = self._solver._records
        rval = rval.reshape((records,) + originalshape)
        # Un-weight the reconstructed field if weighting was performed on
        # the input data set.
        if self._solver._weights is not None:
            rval = rval / self._solver._weights
        # Return the reconstructed field.
        if self._solver._filled:
            rval = ma.array(rval, mask=np.where(np.isnan(rval), True, False))
        return rval

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the rotated EOFs.

        Given a data set, projects it onto the rotated EOFs to generate a
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
        solver = self._solver
        solver._verify_projection_shape(field, solver._originalshape)
        input_ndim = field.ndim
        eof_ndim = len(solver._originalshape) + 1
        # Create a slice object for truncating the EOFs.
        slicer = slice(0, neofs)
        # If required, weight the data set with the same weighting that was
        # used to compute the EOFs.
        field = field.copy()
        if weighted:
            wts = solver.getWeights()
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
        if not solver._valid_nan(field_flat):
            raise ValueError('missing values detected in different '
                             'locations at different times')
        nonMissingIndex = np.where(np.logical_not(np.isnan(field_flat[0])))[0]
        field_flat = field_flat[:, nonMissingIndex]
        # Locate the non-missing values in the EOFs and check they match those
        # in the data set, then isolate the non-missing values.
        eofNonMissingIndex = np.where(
            np.logical_not(np.isnan(solver._flatE[0])))[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise ValueError('field and EOFs have different '
                             'missing value locations')
        # The correct projection of a new data field on the rotated EOFs
        # follows the following equation:
        # PC_new = x_new @ E @ L^(1/2) @ R
        # where
        # PC_new : standardized pseudo-PC for new input data field
        # x_new : new input data field
        # E : non-rotated EOFs (eigenvectors)
        # L^(1/2) : Square root of diagonal matrix containing the eigenvalues
        # R : rotation matrix
        eofs_flat = solver._flatE[:self.neofs, eofNonMissingIndex]
        projected_pcs = field_flat @ eofs_flat.T
        projected_pcs /= np.sqrt(solver._L[:self.neofs])
        projected_pcs = projected_pcs @ self._rotation_matrix
        # Reorder the obtained (standardized) rotated EOFs according
        # to their variance.
        projected_pcs = projected_pcs[:, self._var_idx]
        # Select desired PCs
        projected_pcs = projected_pcs[:, slicer]
        # PCs are standardized. In order to match the correct eofscaling
        # we have to multiply the PCs with
        # the square root of the rotated variance (eofscaling == 1)
        # the rotated variance (eofscaling == 2)
        if eofscaling == 0:
            pass
        elif eofscaling == 1:
            projected_pcs *= np.sqrt(self._eofs_rot_var[slicer])
        elif eofscaling == 2:
            projected_pcs *= self._eofs_rot_var[slicer]
        else:
            raise ValueError('invalid PC scaling option: '
                             '{!s}'.format(eofscaling))
        if eof_ndim > input_ndim:
            # If an extra dimension was introduced, remove it before returning
            # the projected PCs.
            projected_pcs = projected_pcs[0]
        return projected_pcs

    def _solverdata(self):
        """Get the raw data from the EOF solver."""
        return self._solver._data

    def _to2d(self, eofs):
        """Re-shape EOFs to 2D and remove missing values."""
        # Re-shape to 2D.
        info = {}
        neofs = eofs.shape[0]
        channels = eofs.shape[1:]
        nspace = np.prod(channels)
        ev = eofs.reshape([neofs, nspace])
        # Remove missing values.
        try:
            ev = ev.filled(fill_value=np.nan)
            info['filled'] = True
        except AttributeError:
            info['filled'] = False
        nmi = np.where(np.isnan(ev[0]) == False)[0]
        ev_nm = ev[:, nmi]
        # Record information about the transform.
        info['neofs'] = neofs
        info['original_shape'] = channels
        info['dtype'] = eofs.dtype
        info['non_missing_locations'] = nmi
        return ev_nm, info

    def _from2d(self, ev, info):
        """
        Re-shape a 2D array to full shape and replace missing values.

        """
        channels = np.prod(info['original_shape'])
        eofs = np.ones([info['neofs'], channels],
                       dtype=info['dtype']) * np.nan
        eofs[:, info['non_missing_locations']] = ev
        if info['filled']:
            eofs = ma.array(eofs, mask=np.where(np.isnan(eofs), True, False))
        eofs = eofs.reshape((info['neofs'],) + info['original_shape'])
        return eofs

    def apply_metadata(self, var, metadata):
        """Convert an array to a metadata holding instance."""
        return var

    def strip_metadata(self, var):
        """Convert a metadata holding instance to an array."""
        return var, None
