"""Supplementary tools useful for multiple interfaces."""
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


def covcor_dimensions(pc_dims, field_dims):
    """
    Extract the appropriate dimensions from a set of PCs and a field for
    construction of covariance/correlation map dimensions.

    """
    spatial_dims = field_dims[1:]
    try:
        pc_dim = pc_dims[1]
    except IndexError:
        pc_dim = None
    covcor_dims = [d for d in [pc_dim] + spatial_dims if d is not None]
    return covcor_dims
