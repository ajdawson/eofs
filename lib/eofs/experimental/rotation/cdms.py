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
from cdms2 import createVariable

from .standard import Rotator as StandardRotator


class Rotator(StandardRotator):
    """Rotate EOFs from the `cdms2` interface."""

    def __init__(self, *args, **kwargs):
        """
        Rotator(solver, neofs, method='varimax', scaled=True)

        Create an EOF rotator.

        **Arguments:**

        *solver*
            An `~eofs.cdms.Eof` instance that can generate the EOFs to
            be rotated.

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

            solver = Eof(data, weights='area')
            rotator = Rotator(solver, 10)

        A varimax rotator based on the first 5 un-scaled EOFs::

            solver = Eof(data, weights='area')
            rotator = Rotator(solver, 5, scaled=False)

        """
        super(Rotator, self).__init__(*args, **kwargs)

    def eofs(self, *args, **kwargs):
        """
        eofs(neofs=None, renormalize=True)

        Rotated empirical orthogonal functions (EOFs).

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
            A `cdms2` variable containing the ordered rotated EOFs. The
            EOFs are numbered from 0 to *neofs* - 1.

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
        return super(Rotator, self).eofs(*args, **kwargs)

    def varianceFraction(self, *args, **kwargs):
        """
        varianceFraction(neigs=None)

        Fractional rotated EOF mode variances.

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
            A `cdms2` variable containing the fractional variances for
            each rotated EOF mode. The EOF modes are numbered from 0 to
            *neigs* - 1.

        **Examples:**

        The fractional variance represented by each rotated EOF mode::

            rotator = Rotator(solver, 10, scaled=True)
            variance_fractions = rotator.varianceFraction()

        The fractional variance represented by the first rotated EOF mode::

            rotator = Rotator(solver, 10, scaled=True)
            variance_fractions = rotator.varianceFraction(neigs=1)

        """
        return super(Rotator, self).varianceFraction(*args, **kwargs)

    def pcs(self, *args, **kwargs):
        """
        pcs(npcs=None, normalized=False)

        Principal component time series (PCs).

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
            A `cdms2` variable containing the PCs. The PCs are numbered
            from 0 to *npcs* - 1.

        **Examples:**

        All un-scaled PCs::

            pcs = rotator.pcs()

        First 3 PCs scaled to unit variance::

            pcs = rotator.pcs(npcs=3, normalized=True)

        """
        return super(Rotator, self).pcs(*args, **kwargs)

    def _solverdata(self):
        """Get the raw data from the EOF solver."""
        return self._solver._solver._data

    def strip_metadata(self, var):
        """Strip basic metadata from a `cdms2` variable.

        **Argument:**

        *var*
            A `cdms2` variable to strip.

        **Returns:**

        *data*
            An array of the raw data contained in *var*.

        *metadata*
            A dictionary containing basic metadata from *var*. The
            dictionary has entries:
                'dimensions': list of dimension coordinates
                'id': variable's id
                'standard_name': if *var* has a standard name
                'long_name': if *var* has a long name

        """
        metadata = {}
        metadata['dimensions'] = var.getAxisList()
        metadata['id'] = var.id
        for name in ('standard_name', 'long_name'):
            var_name = getattr(var, name, None)
            if var_name is not None:
                metadata[name] = var_name
        return var.asma(), metadata

    def apply_metadata(self, data, metadata):
        """Construct a cube from raw data and a metadata dictionary.

        **Arguments:**

        *data*
            Raw data array.

        *metadata*
            A dictionary of metadata returned from
            `~Rotator.strip_metadata`.

        **Returns:**

        *var*
            A `cdms2` variable.

        """
        dims = metadata['dimensions']
        var = createVariable(data, axes=dims, id=metadata['id'])
        for name in ('standard_name', 'long_name'):
            if name in metadata:
                setattr(var, name, metadata[name])
        return var
