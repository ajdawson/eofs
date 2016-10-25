"""Tests for error handling in `eofs.multivariate`."""
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

import eofs.multivariate as multivariate
from eofs.tests import EofsTest

from .reference import reference_multivariate_solution


# Create a mapping from interface name to solver class.
solvers = {'standard': multivariate.standard.MultivariateEof}
try:
    solvers['cdms'] = multivariate.cdms.MultivariateEof
except AttributeError:
    pass
try:
    solvers['iris'] = multivariate.iris.MultivariateEof
except AttributeError:
    pass


class MVErrorHandlersTest(EofsTest):
    interface = None
    weights = None

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_multivariate_solution(cls.interface,
                                                           cls.weights)
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

    def test_projectField_wrong_number_fields(self):
        with pytest.raises(ValueError):
            pcs = self.solver.projectField([self.solution['sst'][0]])

    def testProjectField_time_dimension_mixture(self):
        sst1, sst2 = self.solution['sst']
        sst1 = sst1[0]
        sst2 = sst2[0:1]
        with pytest.raises(ValueError):
            pcs = self.solver.projectField([sst1, sst2])


# ----------------------------------------------------------------------------
# Error Handler tests for the standard interface


class TestErrorHandlersStandard(MVErrorHandlersTest):
    interface = 'standard'
    weights = 'equal'


# ----------------------------------------------------------------------------
# Error Handler tests for the cdms interface


class TestErrorHandlersCDMS(MVErrorHandlersTest):
    interface = 'cdms'
    weights = 'equal'

    def test_projectField_wrong_input_type(self):
        solution = reference_multivariate_solution('standard', self.weights)
        with pytest.raises(TypeError):
            pcs = self.solver.projectField(solution['sst'])

    def test_projectField_time_dimension_not_first(self):
        sst1, sst2 = self.solution['sst']
        sst1 = sst1.reorder('-t')
        with pytest.raises(ValueError):
            pcs = self.solver.projectField([sst1, sst2])


# ----------------------------------------------------------------------------
# Error Handler tests for the iris interface


class TestErrorHandlersIris(MVErrorHandlersTest):
    interface = 'iris'
    weights = 'equal'

    def test_projectField_wrong_input_type(self):
        solution = reference_multivariate_solution('standard', self.weights)
        with pytest.raises(TypeError):
            pcs = self.solver.projectField(solution['sst'])

    def test_projectField_time_dimension_not_first(self):
        sst1, sst2 = self.solution['sst']
        sst1.transpose([1, 2, 0])
        with pytest.raises(ValueError):
            pcs = self.solver.projectField([sst1, sst2])


# ----------------------------------------------------------------------------
# Constructor tests for the standard interface


class TestConstructorStandard(EofsTest):
    """Test the error handling in the standard interface constructor."""

    @classmethod
    def setup_class(cls):
        cls.solver_class = solvers['standard']

    def test_input_first_dimension_different(self):
        solution = reference_multivariate_solution('standard', 'equal')
        sst1, sst2 = solution['sst']
        sst1 = sst1[0:3]
        sst2 = sst2[0:4]
        with pytest.raises(ValueError):
            solver = self.solver_class([sst1, sst2])

    def test_wrong_number_weights(self):
        solution = reference_multivariate_solution('standard', 'area')
        weights1, weights2 = solution['weights']
        with pytest.raises(ValueError):
            solver = self.solver_class(solution['sst'], weights=[weights1])

    def test_incompatible_weights(self):
        solution = reference_multivariate_solution('standard', 'area')
        weights1, weights2 = solution['weights']
        weights2 = weights2[..., :-1]
        with pytest.raises(ValueError):
            solver = self.solver_class(solution['sst'],
                                       weights=[weights1, weights2])


# ----------------------------------------------------------------------------
# Constructor tests for the cdms interface


class TestConstructorCDMS(EofsTest):
    """Test the error handling in the cdms interface constructor."""

    @classmethod
    def setup_class(cls):
        try:
            cls.solver_class = solvers['cdms']
        except KeyError:
            pytest.skip('missing dependencies required to test '
                        'the cdms interface')

    def test_wrong_number_weights(self):
        solution = reference_multivariate_solution('cdms', 'area')
        weights1, weights2 = solution['weights']
        with pytest.raises(ValueError):
            solver = self.solver_class(solution['sst'], weights=[weights1])

    def test_wrong_input_type(self):
        solution = reference_multivariate_solution('standard', 'equal')
        with pytest.raises(TypeError):
            solver = self.solver_class(solution['sst'])

    def test_input_time_dimension_not_first(self):
        solution = reference_multivariate_solution('cdms', 'equal')
        sst1, sst2 = solution['sst']
        sst1 = sst1.reorder('-t')
        with pytest.raises(ValueError):
            solver = self.solver_class([sst1, sst2])

    def test_input_no_spatial_dimensions(self):
        solution = reference_multivariate_solution('cdms', 'equal')
        sst1, sst2 = solution['sst']
        sst1 = sst1[:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class([sst1, sst2])


# ----------------------------------------------------------------------------
# Constructor tests for the standard interface


class TestConstructorIris(EofsTest):
    """Test the error handling in the iris interface constructor."""

    @classmethod
    def setup_class(cls):
        try:
            cls.solver_class = solvers['iris']
        except KeyError:
            pytest.skip('missing dependencies required to test '
                        'the iris interface')

    def test_wrong_number_weights(self):
        solution = reference_multivariate_solution('iris', 'area')
        weights1, weights2 = solution['weights']
        with pytest.raises(ValueError):
            solver = self.solver_class(solution['sst'], weights=[weights1])

    def test_wrong_input_type(self):
        solution = reference_multivariate_solution('standard', 'equal')
        with pytest.raises(TypeError):
            solver = self.solver_class(solution['sst'])

    def test_input_time_dimension_not_first(self):
        solution = reference_multivariate_solution('iris', 'equal')
        sst1, sst2 = solution['sst']
        sst1.transpose([1, 2, 0])
        with pytest.raises(ValueError):
            solver = self.solver_class([sst1, sst2])

    def test_input_no_spatial_dimensions(self):
        solution = reference_multivariate_solution('iris', 'equal')
        sst1, sst2 = solution['sst']
        sst1 = sst1[:, 0, 0]
        with pytest.raises(ValueError):
            solver = self.solver_class([sst1, sst2])
