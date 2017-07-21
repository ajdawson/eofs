"""Reference solutions for testing the `eofs` package."""
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
import os

import numpy as np
import numpy.ma as ma


def _test_data_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def _retrieve_test_field(name):
    filename = os.path.join(_test_data_path(), '{!s}.npy'.format(name))
    try:
        field = np.load(filename)
    except IOError:
        field = None
    return field


def _tomasked(array):
    try:
        return ma.MaskedArray(array, mask=np.isnan(array))
    except TypeError:
        return array


def _read_reference_solution(weights):
    if weights not in ('equal', 'latitude', 'area', 'area_multi',
                       'area_multi_mix'):
        raise ValueError("invalid weights: '{!s}'".format(weights))
    field_names = ['time',
                   'latitude',
                   'longitude',
                   'sst',
                   'eigenvalues.{!s}'.format(weights),
                   'eofs.{!s}'.format(weights),
                   'eofscor.{!s}'.format(weights),
                   'eofscov.{!s}'.format(weights),
                   'pcs.{!s}'.format(weights),
                   'variance.{!s}'.format(weights),
                   'weights.{!s}'.format(weights),
                   'errors.{!s}'.format(weights),
                   'scaled_errors.{!s}'.format(weights),
                   'rcon.{!s}'.format(weights)]
    fields = {name.split('.')[0]: _tomasked(_retrieve_test_field(name))
              for name in field_names}
    fields['sst'] -= fields['sst'].mean(axis=0)
    return fields


def _wrap_cdms(solution, neofs, time_units):
    try:
        import cdms2
    except ImportError:
        raise ValueError("cannot use container 'cdms' without "
                         "the cdms2 module")
    time_dim = cdms2.createAxis(solution['time'], id='time')
    time_dim.designateTime()
    time_dim.units = time_units
    lat_dim = cdms2.createAxis(solution['latitude'], id='latitude')
    lat_dim.designateLatitude()
    lon_dim = cdms2.createAxis(solution['longitude'], id='longitude')
    lon_dim.designateLongitude()
    eof_dim = cdms2.createAxis(np.arange(1, neofs+1), id='eof')
    eof_dim.long_name = 'eof_number'
    solution['sst'] = cdms2.createVariable(
        solution['sst'],
        axes=[time_dim, lat_dim, lon_dim],
        id='sst')
    solution['eigenvalues'] = cdms2.createVariable(
        solution['eigenvalues'],
        axes=[eof_dim],
        id='eigenvalues')
    solution['eofs'] = cdms2.createVariable(
        solution['eofs'],
        axes=[eof_dim, lat_dim, lon_dim],
        id='eofs')
    solution['pcs'] = cdms2.createVariable(
        solution['pcs'],
        axes=[time_dim, eof_dim],
        id='pcs')
    solution['variance'] = cdms2.createVariable(
        solution['variance'],
        axes=[eof_dim],
        id='variance')
    solution['eofscor'] = cdms2.createVariable(
        solution['eofscor'],
        axes=[eof_dim, lat_dim, lon_dim],
        id='eofscor')
    solution['eofscov'] = cdms2.createVariable(
        solution['eofscov'],
        axes=[eof_dim, lat_dim, lon_dim],
        id='eofscov')
    solution['errors'] = cdms2.createVariable(
        solution['errors'],
        axes=[eof_dim],
        id='errors')
    solution['scaled_errors'] = cdms2.createVariable(
        solution['scaled_errors'],
        axes=[eof_dim],
        id='scaled_errors')
    solution['rcon'] = cdms2.createVariable(
        solution['rcon'],
        axes=[time_dim, lat_dim, lon_dim],
        id='reconstructed_sst')


def _wrap_iris(solution, neofs, time_units):
    try:
        from iris.cube import Cube
        from iris.coords import DimCoord
        from cf_units import Unit
    except ImportError:
        raise ValueError("cannot use container 'iris' without "
                         "the iris module")
    time_dim = DimCoord(solution['time'],
                        standard_name='time',
                        units=Unit(time_units, 'gregorian'))
    lat_dim = DimCoord(solution['latitude'],
                       standard_name='latitude',
                       units='degrees_north')
    lat_dim.guess_bounds()
    lon_dim = DimCoord(solution['longitude'],
                       standard_name='longitude',
                       units='degrees_east')
    lon_dim.guess_bounds()
    eof_dim = DimCoord(np.arange(1, neofs+1),
                       long_name='eof')
    solution['sst'] = Cube(
        solution['sst'],
        dim_coords_and_dims=list(zip((time_dim, lat_dim, lon_dim), range(3))),
        long_name='sst')
    solution['eigenvalues'] = Cube(
        solution['eigenvalues'],
        dim_coords_and_dims=list(zip((eof_dim,), range(1))),
        long_name='eigenvalues')
    solution['eofs'] = Cube(
        solution['eofs'],
        dim_coords_and_dims=list(zip((eof_dim, lat_dim, lon_dim), range(3))),
        long_name='eofs')
    solution['pcs'] = Cube(
        solution['pcs'],
        dim_coords_and_dims=list(zip((time_dim, eof_dim), range(2))),
        long_name='pcs')
    solution['variance'] = Cube(
        solution['variance'],
        dim_coords_and_dims=list(zip((eof_dim,), range(1))),
        long_name='variance')
    solution['eofscor'] = Cube(
        solution['eofscor'],
        dim_coords_and_dims=list(zip((eof_dim, lat_dim, lon_dim), range(3))),
        long_name='eofscor')
    solution['eofscov'] = Cube(
        solution['eofscov'],
        dim_coords_and_dims=list(zip((eof_dim, lat_dim, lon_dim), range(3))),
        long_name='eofscov')
    solution['errors'] = Cube(
        solution['errors'],
        dim_coords_and_dims=list(zip((eof_dim,), range(1))),
        long_name='errors')
    solution['scaled_errors'] = Cube(
        solution['scaled_errors'],
        dim_coords_and_dims=list(zip((eof_dim,), range(1))),
        long_name='scaled_errors')
    solution['rcon'] = Cube(
        solution['rcon'],
        dim_coords_and_dims=list(zip((time_dim, lat_dim, lon_dim), range(3))),
        long_name='reconstructed_sst')


def _wrap_xarray(solution, neofs, time_units):
    try:
        import xarray as xr
    except ImportError:
        try:
            import xray as xr
        except ImportError:
            raise ValueError("cannot use container 'xarray' without "
                             "the xarray/xray module")
    time_dim = xr.IndexVariable('time', solution['time'])
    lat_dim = xr.IndexVariable('latitude', solution['latitude'])
    lon_dim = xr.IndexVariable('longitude', solution['longitude'])
    eof_dim = xr.IndexVariable('eof', np.arange(1, neofs+1))
    solution['sst'] = xr.DataArray(solution['sst'],
                                   coords=[time_dim, lat_dim, lon_dim])
    solution['eigenvalues'] = xr.DataArray(solution['eigenvalues'],
                                           coords=[eof_dim])
    solution['eofs'] = xr.DataArray(solution['eofs'],
                                    coords=[eof_dim, lat_dim, lon_dim])


def _get_wrapper(container_type):
    return {'standard': lambda *args: None,
            'cdms': _wrap_cdms,
            'iris': _wrap_iris,
            'xarray': _wrap_xarray}[container_type]


def reference_solution(container_type, weights):
    """Obtain a reference EOF analysis solution.

    **Arguments:**

    *container_type*
        Either 'standard', 'cdms', 'iris' or 'xarray'.

    *weights*
        Weights method. One of 'equal', 'latitude', or 'area'.

    """
    container_type = container_type.lower()
    weights = weights.lower()
    if container_type not in ('standard', 'iris', 'cdms', 'xarray'):
        raise ValueError("unknown container type "
                         "'{!s}'".format(container_type))
    solution = _read_reference_solution(weights)
    time_units = 'months since 0-1-1 00:00:0.0'
    neofs = len(solution['eigenvalues'])
    _get_wrapper(container_type)(solution, neofs, time_units)
    return solution


def reference_multivariate_solution(container_type, weights):
    """Obtain a reference multivariate EOF analysis solution.

    **Arguments:**

    *container_type*
        Either 'standard', 'cdms', or 'iris'.

    *weights*
        Weights method. One of 'equal', 'latitude', 'area',
        'area_multi', or 'area_multi_mix'.

    """
    if weights.lower() == 'area':
        weights = 'area_multi'
    if weights.lower() == 'none_area':
        weights = 'area_multi_mix'
    solution = reference_solution(container_type, weights)
    nlon = len(solution['longitude'])
    slice1 = slice(0, nlon // 2)
    slice2 = slice(nlon // 2, None)
    for var in ('longitude',
                'sst',
                'eofs',
                'eofscor',
                'eofscov',
                'weights',
                'rcon',):
        try:
            solution[var] = (solution[var][..., slice1],
                             solution[var][..., slice2])
        except TypeError:
            solution[var] = None, None
    return solution
