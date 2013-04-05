"""Rotation kernels for EOF rotation."""
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


def _varimax_kernel(eofs, eps=1e-10, itermax=1000, kaisernorm=True):
    """Rotation of EOFs according to the varimax criterion.

    **Arguments:**

    *eofs*
        A 2-dimensional `~numpy.ndarray` with dimensions [neofs, nspace]
        containing no missing values.

    **Optional arguments:**

    *epsilon*
        Tolerance value used to determine convergence of the rotation
        algorithm. Defaults to 1e-10.

    *itermax*
        Maximum number of iterations to allow in the rotation algorithm.

    *kaisernorm*
        If *True* uses Kaiser row normalization. If *False* no
        normalization is used in the kernel. Defaults to *True*.

    """
    try:
        neofs, nspace = eofs.shape
    except ValueError:
        raise ValueError('kernel requires a 2-D input')
    if neofs < 2:
        raise ValueError('at least 2 EOFs are required for rotation')
    if kaisernorm:
        # Apply Kaiser row normalization.
        scale = np.sqrt((eofs ** 2).sum(axis=0))
        eofs /= scale
    # Define the initial values of the rotation matrix and the convergence
    # monitor.
    rotation = np.eye(neofs, dtype=eofs.dtype)
    delta = 0.
    # Iteratively compute the rotation matrix.
    for i in xrange(itermax):
        z = np.dot(eofs.T, rotation)
        b = np.dot(eofs,
                   z ** 3 - np.dot(z, np.diag((z ** 2).sum(axis=0)) / nspace))
        u, s, v = np.linalg.svd(b)
        rotation = np.dot(u, v)
        delta_previous = delta
        delta = s.sum()
        if delta < delta_previous * (1. + eps):
            # Convergence is reached, stop the iteration.
            break
    # Apply the rotation to the input EOFs.
    reofs = np.dot(eofs.T, rotation).T
    if kaisernorm:
        # Remove the normalization.
        reofs *= scale
    return reofs


KERNEL_MAPPING = {
    'varimax': _varimax_kernel,
}
