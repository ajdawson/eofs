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

import numpy as np
import numpy.ma as ma

from .kernels import KERNEL_MAPPING


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
            self._eofs_rot = KERNEL_MAPPING[method.lower()](eofs, **kwargs)
        except KeyError:
            raise ValueError("unknown rotation method: '{!s}'".format(method))
        # Compute variances of the rotated EOFs as these are used by several
        # methods.
        self._eofs_rot_var = (self._eofs_rot ** 2).sum(axis=1)

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
        if neofs > self.neofs:
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
        if neigs > self.neofs or neigs is None:
            neigs = self.neofs
        # Compute fractions of variance accounted for by each rotated mode.
        eigenvalues = self._solver.eigenvalues(neigs=neigs)
        variance_fractions = self._solver.varianceFraction(neigs=neigs)
        ev, ev_metadata = self.strip_metadata(eigenvalues)
        vf, vf_metadata = self.strip_metadata(variance_fractions)
        if self._scaled:
            ratio = vf[0] / ev[0]
            vf_rot = self._eofs_rot_var * ratio
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
        variances = self._rot_eof_var[slicer]
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
        if npcs > self.neofs or npcs is None:
            npcs = self.neofs
        slicer = slice(0, npcs)
        # Extract the original field from the solver.
        field, fieldinfo = self._to2d(self._solverdata())
        # Compute the PCs.
        pcs = np.dot(field, self._eofs_rot.T)
        if normalized:
            # Optionally standardize the PCs.
            pcs /= pcs.std(axis=0, ddof=1)
        # Select only the required PCs.
        pcs = pcs[:, slicer]
        # Collect the metadata used for PCs by the solver and apply it to
        # these PCs.
        _, pcs_metadata = self.strip_metadata(self._solver.pcs(npcs=npcs))
        pcs = self.apply_metadata(pcs, pcs_metadata)
        return pcs

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
