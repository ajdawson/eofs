"""Test EOF rotations against reference solutions."""
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
from nose import SkipTest
import numpy as np
try:
    from iris.cube import Cube
except:
    pass

import eofs
import eofs.experimental.rotation as rotation
from eofs.tests import EofsTest

from utils import sign_adjustments, error
from reference import reference_rotated_solution


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

# Create a mapping from interface name to rotator class.
rotators = {'standard': rotation.standard.Rotator}
try:
    rotators['cdms'] = rotation.cdms.Rotator
except AttributeError:
    pass
try:
    rotators['iris'] = rotation.iris.Rotator
except AttributeError:
    pass


class RotatorTest(EofsTest):
    """Base class for all rotation test classes."""
    interface = None
    scaled = None
    method = None
    neofs = 5

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_rotated_solution(cls.interface,
                                                      cls.scaled)
        except ValueError:
            raise SkipTest('library component not available '
                           'for {!s} interface'.format(cls.interface))
        try:
            # use default kws for solver, already well tested for weights etc.
            solver = solvers[cls.interface](cls.solution['sst_r'], ddof=0)
            cls.rotator = rotators[cls.interface](solver,
                                                  cls.neofs,
                                                  method=cls.method,
                                                  scaled=cls.scaled)
        except KeyError:
            raise SkipTest('library component not available '
                           'for {!s} interface'.format(cls.interface))

    def test_eofs(self):
        # generate EOF tests for normalized and non-normalized EOFs
        for renormalize in (False, True):
            yield self.check_eofs, renormalize

    def check_eofs(self, renormalize):
        # rotated EOFs should match the (possibly normalized) reference
        # solution
        eofs = self._tomasked(
            self.rotator.eofs(neofs=self.neofs, renormalize=renormalize))
        reofs = self._tomasked(self.solution['rotated_eofs']).copy()
        eofs *= sign_adjustments(eofs, reofs)
        if renormalize:
            variance = (reofs ** 2.).sum(axis=1).sum(axis=1)
            reofs = reofs / np.sqrt(variance)[:, np.newaxis, np.newaxis]
        self.assert_almost_equal(error(eofs, reofs), 0, places=3)

    def test_pcs(self):
        # generate PC tests for normalized and non-normalized PCs
        for normalized in (False, True):
            yield self.check_pcs, normalized

    def check_pcs(self, normalized):
        # rotated PCs should math the (possibly normalized) reference solution
        pcs = self._tomasked(
            self.rotator.pcs(npcs=self.neofs, normalized=normalized))
        rpcs = self._tomasked(self.solution['rotated_pcs']).copy()
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        if normalized:
            rpcs /= rpcs.std(axis=0, ddof=1)
        self.assert_almost_equal(error(pcs, rpcs), 0, places=3)

    def test_varianceFraction(self):
        # rotated variance fractions should match the reference solution
        if not self.scaled:
            # variance fraction is not meaningful when normalized EOFs are
            # rotated
            return
        variance = self._tomasked(
            self.rotator.varianceFraction(neigs=self.neofs)) * 100.
        rvariance = self._tomasked(self.solution['rotated_variance'])
        self.assert_array_almost_equal(variance, rvariance, decimal=3)


#-----------------------------------------------------------------------------
# Tests for the standard interface


class StandardRotatorTest(RotatorTest):
    """Base class for all standard interface solution test cases."""
    interface = 'standard'

    def _tomasked(self, value):
        return value


class TestStandardScaledVarimax(StandardRotatorTest):
    """Rotation of scaled EOFs."""
    scaled = True
    method = 'varimax'


class TestStandardUnscaledVarimax(StandardRotatorTest):
    """Rotation of un-scaled EOFs."""
    scaled = False
    method = 'varimax'


#-----------------------------------------------------------------------------
# Tests for the cdms interface


class CDMSRotatorTest(RotatorTest):
    """Base class for all cdms interface solution test cases."""
    interface = 'cdms'

    def _tomasked(self, value):
        try:
            return value.asma()
        except AttributeError:
            return value


class TestCDMSRotatorScaledVarimax(CDMSRotatorTest):
    """Rotation of scaled EOFs."""
    scaled = True
    method = 'varimax'


class TestCDMSRotatorUnscaledVarimax(CDMSRotatorTest):
    """Rotation of un-scaled EOFs."""
    scaled = False
    method = 'varimax'


#-----------------------------------------------------------------------------
# Tests for the cdms interface


class IrisRotatorTest(RotatorTest):
    """Base class for all iris interface solution test cases."""
    interface = 'iris'

    def _tomasked(self, value):
        if type(value) is not Cube:
            return value
        return value.data


class TestIrisRotatorScaledVarimax(IrisRotatorTest):
    """Rotation of scaled EOFs."""
    scaled = True
    method = 'varimax'


class TestIrisRotatorUnscaledVarimax(IrisRotatorTest):
    """Rotation of un-scaled EOFs."""
    scaled = False
    method = 'varimax'
