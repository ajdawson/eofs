"""Tests for error handling in `eofs`."""
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
    import cdms2
except ImportError:
    pass
import pytest

import eofs
from eofs.tests import EofsTest

from .reference import reference_solution


# Create a mapping from interface name to solver class.
solvers = {'standard': eofs.standard.Eof}
try:
    solvers['cdms'] = eofs.cdms.Eof
except AttributeError:
    pass
try:
    solvers['iris'] = eofs.iris.Eof
except AttributeError:
    pass
try:
    solvers['xarray'] = eofs.xarray.Eof
except AttributeError:
    pass


class ErrorHandlersTest(EofsTest):
    interface = None
    weights = None

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_solution(cls.interface, cls.weights)
        except ValueError:
            pytest.skip('missing dependencies required to test '
                        'the {!s} interface'.format(cls.interface))
        cls.neofs = cls.solution['eigenvalues'].shape[0]
        try:
            cls.solver = solvers[cls.interface](
                cls.solution['sst'], weights=cls.solution['weights'])
        except KeyError:
            pytest.skip('missing dependencies required to test '
                        'the {!s} interface'.format(cls.interface))

    def test_invalid_pcscaling(self):
        # PCs with invalid scaling
        with pytest.raises(ValueError):
            pcs = self.solver.pcs(pcscaling=-1)

    def test_invalid_eofscaling(self):
        # EOFs with invalid scaling
        with pytest.raises(ValueError):
            eofs = self.solver.eofs(eofscaling=-1)

    def test_projectField_invalid_dimensions(self):
        # projecting a field with too few dimensions
        data = self.solution['sst'][:, 0, 0]
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)

    def test_projectField_invalid_shape(self):
        # projecting a field with the wrong shape
        data = self.solution['sst'][..., 0:1]
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)

    def test_projectField_different_missing_values(self):
        # projecting a field with different missing values
        solution = reference_solution(self.interface, self.weights)
        data = solution['sst']
        mask = data.mask
        mask[0] = True
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)


# ----------------------------------------------------------------------------
# Error Handler tests for the standard interface


class TestErroHandlersStandard(ErrorHandlersTest):
    interface = 'standard'
    weights = 'equal'


# ----------------------------------------------------------------------------
# Error Handler tests for the cdms interface


class TestErrorHandlersCDMS(ErrorHandlersTest):
    """Test error handling in the cdms interface."""
    interface = 'cdms'
    weights = 'equal'

    def test_projectField_invalid_type(self):
        # projecting a field of the wrong type
        solution = reference_solution('standard', 'equal')
        with pytest.raises(TypeError):
            pcs = self.solver.projectField(solution['sst'])


# ----------------------------------------------------------------------------
# Error Handler tests for the iris interface


class TestErrorHandlersIris(ErrorHandlersTest):
    """Test error handling in the iris interface."""
    interface = 'iris'
    weights = 'equal'

    def test_projectField_invalid_type(self):
        # projecting a field of the wrong type
        solution = reference_solution('standard', 'equal')
        with pytest.raises(TypeError):
            pcs = self.solver.projectField(solution['sst'])

    def test_projectField_different_missing_values(self):
        # projecting a field with different missing values
        solution = reference_solution(self.interface, self.weights)
        data = solution['sst']
        mask = data.data.mask
        mask[0] = True
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)

    def test_projectField_invalid_dimension_order(self):
        # projecting a field with the time dimension not at the front
        solution = reference_solution(self.interface, self.weights)
        data = solution['sst']
        data.transpose((2, 1, 0))
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)


# ----------------------------------------------------------------------------
# Error Handler tests for the xarray interface


class TestErrorHandlersXarray(ErrorHandlersTest):
    """Test error handling in the xarray interface."""
    interface = 'xarray'
    weights = 'equal'

    def test_projectField_invalid_type(self):
        # projecting a field of the wrong type
        solution = reference_solution('standard', 'equal')
        with pytest.raises(TypeError):
            pcs = self.solver.projectField(solution['sst'])

    def test_projectField_different_missing_values(self):
        # projecting a field with different missing values
        solution = reference_solution(self.interface, self.weights)
        data = solution['sst']
        data.values[0] = np.nan
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)

    def test_projectField_invalid_dimension_order(self):
        # projecting a field with the time dimension not at the front
        solution = reference_solution(self.interface, self.weights)
        data = solution['sst']
        data = data.transpose('longitude', 'latitude', 'time')
        with pytest.raises(ValueError):
            pcs = self.solver.projectField(data)


# ----------------------------------------------------------------------------
# Constructor tests for the standard interface


class TestConstructorStandard(EofsTest):
    """Test the error handling in the standard interface constructor."""

    @classmethod
    def setup_class(cls):
        cls.solver_class = solvers['standard']

    def test_invalid_input_dimensions(self):
        # too few input dimensions
        solution = reference_solution('standard', 'equal')
        data = solution['sst'][:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_invalid_weights_dimensions(self):
        # weights with incompatible dimensions
        solution = reference_solution('standard', 'area')
        data = solution['sst']
        weights = solution['weights'][:, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights=weights)

    def test_invalid_weights_type(self):
        # weights with an incompatible type
        solution = reference_solution('standard', 'area')
        data = solution['sst']
        weights = 'area'
        with pytest.raises(TypeError):
            solver = self.solver_class(data, weights=weights)

    def test_input_with_all_missing_values(self):
        # input with only missing values, missing values are propagated
        # because the default for the center argument is True
        solution = reference_solution('standard', 'equal')
        data = solution['sst']
        mask = data.mask
        mask[-1] = True
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_with_non_uniform_missing_values(self):
        # missing values in different places at different times leads to
        # errors in computing the SVD
        solution = reference_solution('standard', 'equal')
        data = solution['sst']
        mask = data.mask
        mask[-1] = True
        with pytest.raises(ValueError):
            solver = self.solver_class(data, center=False)


# ----------------------------------------------------------------------------
# Constructor tests for the cdms interface


class TestConstructorCDMS(EofsTest):
    """Test the error handling in the cdms interface constructor."""

    @classmethod
    def setup_class(cls):
        try:
            cls.solver_class = solvers['cdms']
        except KeyError:
            pytest.skip('missing dependencies required to test the '
                        'cdms interface')

    def test_wrong_input_type(self):
        # input of the wrong type
        solution = reference_solution('standard', 'equal')
        data = solution['sst']
        with pytest.raises(TypeError):
            solver = self.solver_class(data)

    def test_input_without_time_dimension(self):
        # no time dimension in the input
        solution = reference_solution('cdms', 'equal')
        data = solution['sst'](time=slice(0, 1), squeeze=True)
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_time_dimension_not_first(self):
        # time not the first dimension in the input
        solution = reference_solution('cdms', 'equal')
        data = solution['sst']
        data = data.reorder('xyt')
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_no_spatial_dimensions(self):
        # not enough dimensions in the input
        solution = reference_solution('cdms', 'equal')
        data = solution['sst'][:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_invalid_builtin_weights_value(self):
        # invalid weighting scheme name
        solution = reference_solution('cdms', 'equal')
        data = solution['sst']
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='invalid')

    def test_builtin_latitude_weights_with_missing_dimension(self):
        # latitude weights without latitude dimension
        solution = reference_solution('cdms', 'equal')
        data = solution['sst'](latitude=slice(0, 1), squeeze=True)
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='coslat')

    def test_builtin_area_weights_with_missing_dimension(self):
        # area weights without longitude dimension
        solution = reference_solution('cdms', 'equal')
        data = solution['sst'](longitude=slice(0, 1), squeeze=True)
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='area')

    def test_builtin_area_weights_with_non_adjacent_dimensions(self):
        # area weights with latitude and longitude not adjacent in input
        solution = reference_solution('cdms', 'equal')
        data = solution['sst']
        newdim = cdms2.createAxis([0.], id='height')
        newdim.designateLevel()
        data = cdms2.createVariable(cdms2.MV.reshape(data, data.shape + (1,)),
                                    axes=data.getAxisList() + [newdim],
                                    id=data.id)
        data = data.reorder('txzy')
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='area')


# ----------------------------------------------------------------------------
# Constructor tests for the iris interface


class TestConstructorIris(EofsTest):
    """Test the error handling in the iris interface constructor."""

    @classmethod
    def setup_class(cls):
        try:
            cls.solver_class = solvers['iris']
        except KeyError:
            pytest.skip('missing dependencies required to test the '
                        'iris interface')

    def test_wrong_input_type(self):
        # input of the wrong type
        solution = reference_solution('standard', 'equal')
        data = solution['sst']
        with pytest.raises(TypeError):
            solver = self.solver_class(data)

    def test_input_without_time_dimension(self):
        # no time dimension in the input
        solution = reference_solution('iris', 'equal')
        data = solution['sst'][0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_time_dimension_not_first(self):
        # time not the first dimension in the input
        solution = reference_solution('iris', 'equal')
        data = solution['sst']
        data.transpose((2, 1, 0))
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_no_spatial_dimensions(self):
        # not enough dimensions in the input
        solution = reference_solution('iris', 'equal')
        data = solution['sst'][:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_invalid_builtin_weights_value(self):
        # invalid weighting scheme name
        solution = reference_solution('iris', 'equal')
        data = solution['sst']
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='invalid')

    def test_builtin_latitude_weights_with_missing_dimension(self):
        # latitude weights without latitude dimension
        solution = reference_solution('iris', 'latitude')
        data = solution['sst'][:, 0, :]
        data.remove_coord('latitude')
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='coslat')

    def test_builtin_area_weights_with_missing_dimension(self):
        # latitude weights without latitude dimension
        solution = reference_solution('iris', 'area')
        data = solution['sst'][:, :, 0]
        data.remove_coord('longitude')
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='area')

    def test_multiple_time_dimensions(self):
        # multiple dimensions representing time
        solution = reference_solution('iris', 'equal')
        data = solution['sst']
        lon = data.coord('longitude')
        lon.rename('time')
        lon.units = 'days since 1999-01-01 00:00:0.0'
        with pytest.raises(ValueError):
            solver = self.solver_class(data)


# ----------------------------------------------------------------------------
# Constructor tests for the xarray interface


class TestConstructorXarray(EofsTest):
    """Test the error handling in the xarray interface constructor."""

    @classmethod
    def setup_class(cls):
        try:
            cls.solver_class = solvers['xarray']
        except KeyError:
            pytest.skip('missing dependencies required to test the '
                        'xarray interface')

    def test_wrong_input_type(self):
        # input of the wrong type
        solution = reference_solution('standard', 'equal')
        data = solution['sst']
        with pytest.raises(TypeError):
            solver = self.solver_class(data)

    def test_input_without_time_dimension(self):
        # no time dimension in the input
        solution = reference_solution('xarray', 'equal')
        data = solution['sst'][0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_time_dimension_not_first(self):
        # time not the first dimension in the input
        solution = reference_solution('xarray', 'equal')
        data = solution['sst']
        data = data.transpose('longitude', 'latitude', 'time')
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_input_no_spatial_dimensions(self):
        # not enough dimensions in the input
        solution = reference_solution('xarray', 'equal')
        data = solution['sst'][:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class(data)

    def test_invalid_builtin_weights_value(self):
        # invalid weighting scheme name
        solution = reference_solution('xarray', 'equal')
        data = solution['sst']
        with pytest.raises(ValueError):
            solver = self.solver_class(data, weights='invalid')

    def test_multiple_time_dimensions(self):
        # multiple dimensions representing time
        solution = reference_solution('xarray', 'equal')
        data = solution['sst']
        data.coords['longitude'].attrs['axis'] = 'T'
        with pytest.raises(ValueError):
            solver = self.solver_class(data)
