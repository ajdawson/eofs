"""Utilities for testing the `eofs` package."""
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
try:
    from iris.cube import Cube
except ImportError:
    pass


def _close(a, b, rtol=1e-05, atol=1e-08):
    """Check if two values are close."""
    return abs(a - b) <= (atol + rtol * abs(b))


def __tomasked(*args):
    """Convert cdms2 variables or iris cubes to masked arrays.

    The conversion is safe, so if non-variables/cubes are passed they
    are just returned.

    """
    def __asma(a):
        try:
            if type(a) is Cube:
                # Retrieve the data from the cube.
                a = a.data
        except NameError:
            pass
        try:
            # Retrieve data from cdms variable.
            a = a.asma()
        except AttributeError:
            # The input is already an array or masked array, either extracted
            # from an iris cube, or was like that to begin with.
            pass
        return a
    return [__asma(a) for a in args]


def error(a, b):
    """Compute the error between two arrays.

    Computes the RMSD normalized by the range of the second input.

    """
    a, b = __tomasked(a, b)
    return np.sqrt((a - b)**2).mean() / (np.max(b) - np.min(b))


def sign_adjustments(eofset, refeofset):
    """Sign adjustments for EOFs/PCs.

    Create a matrix of sign weights used for adjusting the sign of a set
    of EOFs or PCs to the sign of a reference set.

    The first dimension is assumed to be modes.

    **Arguments:**

    *eofset*
        Set of EOFs.

    *refeofset*
        Reference set of EOFs.

    """
    if eofset.shape != refeofset.shape:
        raise ValueError('input set has different shape from reference set')
    eofset, refeofset = __tomasked(eofset, refeofset)
    nmodes = eofset.shape[0]
    signs = np.ones([nmodes])
    shape = [nmodes] + [1] * (eofset.ndim - 1)
    eofset = eofset.reshape([nmodes, np.prod(eofset.shape[1:], dtype=np.int)])
    refeofset = refeofset.reshape([nmodes,
                                   np.prod(refeofset.shape[1:],
                                           dtype=np.int)])
    for mode in range(nmodes):
        i = 0
        try:
            while _close(eofset[mode, i], 0.) or \
                  _close(refeofset[mode, i], 0.) \
                  or np.ma.is_masked(eofset[mode, i]) or \
                  np.ma.is_masked(refeofset[mode, i]):
                i += 1
        except IndexError:
            i = 0
        if np.sign(eofset[mode, i]) != np.sign(refeofset[mode, i]):
            signs[mode] = -1
    return signs.reshape(shape)


if __name__ == '__main__':
    pass
