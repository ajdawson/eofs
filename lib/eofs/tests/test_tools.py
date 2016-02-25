"""Tests for the `eofs.tools` package."""
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
from __future__ import absolute_import

from nose import SkipTest
from nose.tools import raises
import numpy as np
import numpy.ma as ma
try:
    from iris.cube import Cube
except ImportError:
    pass

import eofs
from eofs.tests import EofsTest

from .reference import reference_solution
from .utils import sign_adjustments


# Create a mapping from interface name to tools module and solver class.
tools = {'standard': eofs.tools.standard}
solvers = {'standard': eofs.standard.Eof}
try:
    tools['cdms'] = eofs.tools.cdms
    solvers['cdms'] = eofs.cdms.Eof
except AttributeError:
    pass
try:
    tools['iris'] = eofs.tools.iris
    solvers['iris'] = eofs.iris.Eof
except AttributeError:
    pass
try:
    tools['xarray'] = eofs.tools.xarray
    solvers['xarray'] = eofs.xarray.Eof
except AttributeError:
    pass


class ToolsTest(EofsTest):
    """"""
    interface = None
    weights = None

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_solution(cls.interface, cls.weights)
        except ValueError:
            raise SkipTest('library component not available '
                           'for {!s} interface'.format(cls.interface))
        cls.neofs = cls.solution['eigenvalues'].shape[0]
        try:
            cls.solver = solvers[cls.interface](
                cls.solution['sst'], weights=cls.solution['weights'])
            cls.tools = {'covariance': tools[cls.interface].covariance_map,
                         'correlation': tools[cls.interface].correlation_map}
        except KeyError:
            raise SkipTest('library component not available '
                           'for {!s} interface'.format(cls.interface))

    def test_covariance_map(self):
        # covariance maps should match reference EOFs as covariance
        pcs = self.solver.pcs(npcs=self.neofs, pcscaling=1)
        cov = self.tools['covariance'](pcs, self.solution['sst'])
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs))
        reofs = self._tomasked(self.solution['eofs'])
        cov = self._tomasked(cov) * sign_adjustments(eofs, reofs)
        self.assert_array_almost_equal(cov, self.solution['eofscov'])

    def test_correlation_map(self):
        # correlation maps should match reference EOFs as correlation
        pcs = self.solver.pcs(npcs=self.neofs, pcscaling=1)
        cor = self.tools['correlation'](pcs, self.solution['sst'])
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs))
        reofs = self._tomasked(self.solution['eofs'])
        cor = self._tomasked(cor) * sign_adjustments(eofs, reofs)
        self.assert_array_almost_equal(cor, self.solution['eofscor'])

    def test_covariance_map_point(self):
        # single point covariance map should match reference EOFs as covariance
        # at the same point
        pcs = self.solver.pcs(npcs=1, pcscaling=1)[:, 0]
        cov = self.tools['covariance'](pcs, self.solution['sst'][:, 5, 5])
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs))
        reofs = self._tomasked(self.solution['eofs'])
        cov = self._tomasked(cov) * sign_adjustments(eofs, reofs)[0]
        self.assert_array_almost_equal(cov, self.solution['eofscov'][0, 5, 5])

    def test_correlation_map_point(self):
        # single point correlation map should match reference EOFs as
        # correlation at the same point
        pcs = self.solver.pcs(npcs=1, pcscaling=1)[:, 0]
        cor = self.tools['correlation'](pcs, self.solution['sst'][:, 5, 5])
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs))
        reofs = self._tomasked(self.solution['eofs'])
        cor = self._tomasked(cor) * sign_adjustments(eofs, reofs)[0]
        self.assert_array_almost_equal(cor, self.solution['eofscor'][0, 5, 5])

    def test_covcor_map_invalid_time_dimension(self):
        # generate tests for covariance/correlation maps with invalid time
        # dimensions
        for maptype in ('covariance', 'correlation'):
            yield self.check_covcor_map_invalid_time_dimension, maptype

    @raises(ValueError)
    def check_covcor_map_invalid_time_dimension(self, maptype):
        # compute a map with an invalid time dimension in the input
        pcs = self.solver.pcs(npcs=self.neofs, pcscaling=1)[:-1]
        covcor = self.tools[maptype](pcs, self.solution['sst'])

    def test_covcor_map_invalid_pc_shape(self):
        # generate tests for covariance/correlation maps with input PCs with
        # invalid shape
        for maptype in ('covariance', 'correlation'):
            yield self.check_covcor_map_invalid_pc_shape, maptype

    @raises(ValueError)
    def check_covcor_map_invalid_pc_shape(self, maptype):
        # compute a map for PCs with invalid shape
        covcor = self.tools[maptype](self.solution['sst'],
                                     self.solution['sst'])


# ----------------------------------------------------------------------------
# Tests for the standard interface


class TestToolsStandard(ToolsTest):
    """Test the standard interface tools."""
    interface = 'standard'
    weights = 'equal'

    def _tomasked(self, value):
        return value


# ----------------------------------------------------------------------------
# Tests for the cdms interface


class TestToolsCDMS(ToolsTest):
    """Test the cdms interface tools."""
    interface = 'cdms'
    weights = 'equal'

    def _tomasked(self, value):
        try:
            return value.asma()
        except AttributeError:
            return value


# ----------------------------------------------------------------------------
# Tests for the iris interface


class TestToolsIris(ToolsTest):
    """Test the iris interface tools."""
    interface = 'iris'
    weights = 'equal'

    def _tomasked(self, value):
        if type(value) is not Cube:
            return value
        return value.data


# ----------------------------------------------------------------------------
# Tests for the xarray interface


class TestToolsXarray(ToolsTest):
    """Test the xarray interface tools."""
    interface = 'xarray'
    weights = 'equal'

    def _tomasked(self, value):
        try:
            return ma.masked_invalid(value.values)
        except:
            return ma.masked_invalid(value)
