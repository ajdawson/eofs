"""Supplementary tools for the `iris` EOF analysis interface."""
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
from copy import copy
import warnings

import numpy as np
from iris.cube import Cube
from iris.analysis.cartography import area_weights
from iris.analysis.cartography import cosine_latitude_weights
from iris.exceptions import CoordinateMultiDimError

from . import standard
from .generic import covcor_dimensions


def weights_array(cube, scheme):
    """Weights for a data set on a grid.

    Returned weights are a `numpy.ndarray` broadcastable to the shape of
    the input cube.

    **Arguments:**

    *cube*
        An `~iris.cube.Cube` instance to generate weights for.

    *scheme*
        Weighting scheme to use. The following values are accepted:

        * *'coslat'* : Square-root of cosine of latitude.
        * *'area'* : Square-root of grid cell area normalized by total
                     grid area.

    **Returns:**

    *weights*
        An array contanining the weights (not a `~iris.cube.Cube`).

    **Examples:**

    Area weights for a `~iris.cube.Cube` on 2D grid:

        weights = weights_array(cube, scheme='area')

    Square-root of cosine of latitude weights for a `~iris.cube.Cube`
    with a latitude dimension:

        weights = weights_array(cube, scheme='coslat')

    """
    # Always use lower-case for the scheme, allowing the user to use
    # upper-case in their calling code without an error.
    scheme = scheme.lower()
    if scheme in ('area',):
        # Handle area weighting.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                weights = np.sqrt(area_weights(cube, normalize=True))
        except (ValueError, CoordinateMultiDimError):
            raise ValueError('cannot generate area weights')
    elif scheme in ('coslat',):
        # Handle square-root of cosine of latitude weighting.
        try:
            weights = np.sqrt(cosine_latitude_weights(cube))
        except (ValueError, CoordinateMultiDimError):
            raise ValueError('cannot generate latitude weights')
    else:
        raise ValueError("invalid weighting scheme: '{!s}'".format(scheme))
    return weights


def coord_and_dim(cube, coord, multiple=False):
    """
    Retrieve a coordinate dimension and its corresponding position from
    a `~iris.cube.Cube` instance.

    **Arguments:**

    *cube*
        An `~iris.cube.Cube` instance to retrieve the dimension from.

    *coord*
        Name of the coordinate dimension to retrieve.

    **Returns:**

    *coord_tuple*
        A 2-tuple of (coordinate_dimension, dimension_number).

    """
    coords = filter(lambda c: coord in c.name(), cube.dim_coords)
    if len(coords) > 1:
        raise ValueError('multiple {} coordinates are not '
                         'allowed'.format(coord))
    try:
        c = coords[0]
    except IndexError:
        raise ValueError('cannot get {!s} coordinate from '
                         'cube {!r}'.format(coord, cube))
    c_dim = cube.coord_dims(c)
    c_dim = c_dim[0] if c_dim else None
    return c, c_dim


def _time_coord_info(cube):
    time, time_dim = coord_and_dim(cube, 'time')
    coords = list(copy(cube.dim_coords))
    coords.remove(time)
    coords = [time] + coords
    return time_dim, coords


def _map_and_dims(pcs, field, mapfunc, *args, **kwargs):
    """
    Compute a set of covariance/correlation maps and the resulting
    dimensions.

    """
    info = {}
    for cube in (field, pcs):
        info[cube.name()] = _time_coord_info(cube)
    cmap_args = [np.rollaxis(cube.data, info[cube.name()][0])
                 for cube in (pcs, field)]
    cmap_args += args
    dim_args = [info[cube.name()][1] for cube in (pcs, field)]
    cmap = mapfunc(*cmap_args, **kwargs)
    dims = covcor_dimensions(*dim_args)
    return cmap, dims


def correlation_map(pcs, field):
    """Correlation between PCs and a field.

    Computes maps of the correlation between each PC and the given
    field at each grid point.

    Given a set of PCs contained in a `~iris.cube.Cube` (e.g., as output
    from `eofs.iris.Eof.pcs`) and a field with a time dimension
    contained in a `iris.cube.Cube`, one correlation map per PC is
    computed.

    The field must have the same length time dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs contained in a `~iris.cube.Cube`.

    *field*
        Field Spatial-temporal field contained in a `~iris.cube.Cube`.

    **Returns:**

    *correlation_maps*
        A `~iris.cube.Cube` containing the correlation maps.

    **Examples:**

    Assuming *solver* is an instance of `eofs.iris.Eof`, compute
    correlation maps for each PC::

        pcs = solver.pcs(pcscaling=1)
        correlation_maps = correlation_map(pcs, field)

    """
    # Compute the correlation map and retrieve appropriate Iris coordinate
    # dimensions for it.
    cor, dims = _map_and_dims(pcs, field, standard.correlation_map)
    if not dims:
        # There are no output dimensions, return a scalar.
        return cor
    # Otherwise return an Iris cube.
    cor = Cube(cor, dim_coords_and_dims=zip(dims, range(cor.ndim)))
    cor.long_name = 'pc_correlation'
    return cor


def covariance_map(pcs, field, ddof=1):
    """Covariance between PCs and a field.

    Computes maps of the covariance between each PC and the given
    field at each grid point.

    Given a set of PCs contained in a `~iris.cube.Cube` (e.g., as output
    from `eofs.iris.Eof.pcs`) and a field with a time dimension
    contained in a `iris.cube.Cube`, one covariance map per PC is
    computed.

    The field must have the same length time dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs contained in a `~iris.cube.Cube`.

    *field*
        Field Spatial-temporal field contained in a `~iris.cube.Cube`.

    **Keyword arguments:**

    *ddof*
        'Delta degrees of freedom'. The divisor used to normalize
        the covariance matrix is *N - ddof* where *N* is the
        number of samples. Defaults to *1*.

    **Returns:**

    *covariance_maps*
        A `~iris.cube.Cube` containing the covariance maps.

    **Examples:**

    Compute covariance maps for each PC::

        pcs = solver.pcs(pcscaling=1)
        covariance_maps = covariance_map(pcs, field)

    """
    # Compute the covariance map and retrieve appropriate Iris coordinate
    # dimensions for it.
    cov, dims = _map_and_dims(pcs, field, standard.covariance_map, ddof=1)
    if not dims:
        # There are no output dimensions, return a scalar.
        return cov
    # Otherwise return an Iris cube.
    cov = Cube(cov, dim_coords_and_dims=zip(dims, range(cov.ndim)))
    cov.long_name = 'pc_covariance'
    return cov
