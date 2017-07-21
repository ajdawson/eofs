"""Supplementary tools for the `xarray` EOF analysis interface."""
# (c) Copyright 2016 Andrew Dawson. All Rights Reserved.
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
    import xarray as xr
except ImportError:
    import xray as xr

from . import standard
from .generic import covcor_dimensions


def find_time_coordinates(array):
    """Find time dimension coordinates in an `xarray.DataArray`.

    Time coordinates satisfy one or more of:

    * Values have a dtype of `numpy.datetime64`.
    * Name of the coordinate is 'time'.
    * The coordinate has an attribute 'axis' with value 'T'.

    **Argument:**

    *array*
        An `xarray.DataArray`.

    **Returns:**

    *time_coords*
        A list of time dimension coordinates. The list may be empty if
        no coordinates were recognised as time.

    """
    time_coords = []
    for name in array.dims:
        coord = array.coords[name]
        is_time = (np.issubdtype(coord.dtype, np.datetime64) or
                   coord.name == 'time' or
                   coord.attrs.get('axis') == 'T')
        if is_time:
            time_coords.append(coord)
    return time_coords


def categorise_ndcoords(array, time_name):
    """
    Categorise all the non-dimension coordinates of an
    `xarray.DataArray` into those that span only time, those that span
    only space, and those that span both time and space.

    **Arguments:**

    *array*
        An `xarray.DataArray`.

    *time_name*
        Name of the time dimension coordinate in the input *array*.

    **Returns:**

    *time_coords*
        A list of coordinates that span only the time dimension.

    *space_coords*
        A list of coordinates that span only the space dimensions.

    *time_space_coords*
        A list of coordinates that span both the time and space
        coordinates.

    """
    ndcoords = [coord for name, coord in array.coords.items()
                if name not in array.dims]
    time_ndcoords = []
    space_ndcoords = []
    time_space_ndcoords = []
    for coord in ndcoords:
        if coord.dims == (time_name,):
            time_ndcoords.append(coord)
        elif coord.dims:
            if time_name in coord.dims:
                time_space_ndcoords.append(coord)
            else:
                space_ndcoords.append(coord)
    return time_ndcoords, space_ndcoords, time_space_ndcoords


def weights_array(array, scheme):
    """Generate a weights array for a given weighting scheme.

    .. note::

       Currently not implemented, will raise a ValueError().

    """
    scheme = scheme.lower()
    raise ValueError("invalid weighting scheme: '{!s}'".format(scheme))


def _coord_info(array):
    time_coords = find_time_coordinates(array)
    if len(time_coords) > 1:
        raise ValueError('multiple time coordinates are not allowed')
    if not time_coords:
        msg = 'no time coordinates found in {!s}'.format(array.name)
        raise ValueError(msg)
    time_coord = time_coords[0]
    time_dim = array.dims.index(time_coord.name)
    coords = [time_coord] + [array.coords[name] for name in array.dims
                             if name != time_coord.name]
    return time_dim, coords


def _map_and_coords(pcs, field, mapfunc, *args, **kwargs):
    info = {}
    for array in (field, pcs):
        info[array.name] = _coord_info(array)
    cmap_args = [np.rollaxis(array.values, info[array.name][0])
                 for array in (pcs, field)]
    cmap_args += args
    cmap = mapfunc(*cmap_args, **kwargs)
    dim_args = [info[array.name][1] for array in (pcs, field)]
    dims = covcor_dimensions(*dim_args)
    return cmap, dims


def correlation_map(pcs, field):
    """Correlation between PCs and a field.

    Computes maps of the correlation between each PC and the given
    field at each grid point.

    Given a set of PCs contained in a `~xarray.DataArray` (e.g., as
    output from `eofs.xarray.Eof.pcs`) and a field with a time dimension
    contained in a `xarray.DataArray`, one correlation map per PC is
    computed.

    The field must have the same length time dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs contained in a `~xarray.DataArray`.

    *field*
        Field Spatial-temporal field contained in a `~xarray.DataArray`.

    **Returns:**

    *correlation_maps*
        A `~xarray.DataArray` containing the correlation maps.

    **Examples:**

    Assuming *solver* is an instance of `eofs.xarray.Eof`, compute
    correlation maps for each PC::

        pcs = solver.pcs(pcscaling=1)
        correlation_maps = correlation_map(pcs, field)

    """
    cor, coords = _map_and_coords(pcs, field, standard.correlation_map)
    if not coords:
        return cor
    cor = xr.DataArray(cor, coords=coords, name='pc_correlation',
                       attrs={'long_name': 'pc_correlation'})
    return cor


def covariance_map(pcs, field, ddof=1):
    """Covariance between PCs and a field.

    Computes maps of the covariance between each PC and the given
    field at each grid point.

    Given a set of PCs contained in a `~xarray.DataArray` (e.g., as
    output from `eofs.xarray.Eof.pcs`) and a field with a time dimension
    contained in a `xarray.DataArray`, one covariance map per PC is
    computed.

    The field must have the same length time dimension as the PCs. Any
    number of spatial dimensions (including zero) are allowed in the
    field and there can be any number of PCs.

    **Arguments:**

    *pcs*
        PCs contained in a `~xarray.DataArray`.

    *field*
        Field Spatial-temporal field contained in a `~xarray.DataArray`.

    **Keyword arguments:**

    *ddof*
        'Delta degrees of freedom'. The divisor used to normalize
        the covariance matrix is *N - ddof* where *N* is the
        number of samples. Defaults to *1*.

    **Returns:**

    *covariance_maps*
        A `~xarray.DataArray` containing the covariance maps.

    **Examples:**

    Assuming *solver* is an instance of `eofs.xarray.Eof`, compute
    covariance maps for each PC::

        pcs = solver.pcs(pcscaling=1)
        covariance_maps = covariance_map(pcs, field)

    """
    cov, coords = _map_and_coords(pcs, field, standard.covariance_map,
                                  ddof=ddof)
    if not coords:
        return cov
    cov = xr.DataArray(cov, coords=coords, name='pc_covariance',
                       attrs={'long_name': 'pc_covariance'})
    return cov
