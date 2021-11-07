"""Test `eofs.multivariate` computations against reference solutions."""
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
import pytest

import eofs.multivariate as multivariate
from eofs.tests import EofsTest

from .utils import sign_adjustments
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


class MVSolutionTest(EofsTest):
    """Base class for all multivariate solution test classes."""
    interface = None
    weights = None
    alternate_weights_arg = None

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_multivariate_solution(cls.interface,
                                                           cls.weights)
        except ValueError:
            pytest.skip('missing dependencies required to test '
                        'the {!s} interface'.format(cls.interface))
        cls.neofs = cls.solution['eigenvalues'].shape[0]
        if cls.alternate_weights_arg is not None:
            weights = cls.alternate_weights_arg
        else:
            weights = cls.solution['weights']
        try:
            cls.solver = solvers[cls.interface](cls.solution['sst'],
                                                weights=weights)
        except KeyError:
            pytest.skip('missing dependencies required to test '
                        'the {!s} interface'.format(cls.interface))

    def test_eigenvalues(self):
        self.assert_array_almost_equal(
            self.solver.eigenvalues(neigs=self.neofs),
            self.solution['eigenvalues'])

    @pytest.mark.parametrize('eofscaling', (0, 1, 2))
    def test_eofs(self, eofscaling):
        eofs = [self._tomasked(e)
                for e in self.solver.eofs(neofs=self.neofs,
                                          eofscaling=eofscaling)]
        reofs = [self._tomasked(e).copy() for e in self.solution['eofs']]
        eofs = [e * sign_adjustments(e, r) for e, r in zip(eofs, reofs)]
        reigs = self._tomasked(self.solution['eigenvalues'])
        if eofscaling == 1:
            reofs = [r / np.sqrt(reigs)[:, np.newaxis, np.newaxis]
                     for r in reofs]
        elif eofscaling == 2:
            reofs = [r * np.sqrt(reigs)[:, np.newaxis, np.newaxis]
                     for r in reofs]
        for e, r in zip(eofs, reofs):
            self.assert_array_almost_equal(e, r)

    def test_eofsAsCovariance(self):
        # EOFs as covariance between PCs and input field should match the
        # reference solution
        eofs = [self._tomasked(e)
                for e in self.solver.eofsAsCovariance(neofs=self.neofs,
                                                      pcscaling=1)]
        reofs = [self._tomasked(e) for e in self.solution['eofscov']]
        eofs = [e * sign_adjustments(e, r) for e, r in zip(eofs, reofs)]
        for e, r in zip(eofs, reofs):
            self.assert_array_almost_equal(e, r)

    def test_eofsAsCorrelation(self):
        # EOFs as correlation between PCs and input field should match the
        # reference solution
        eofs = [self._tomasked(e)
                for e in self.solver.eofsAsCorrelation(neofs=self.neofs)]
        reofs = [self._tomasked(e) for e in self.solution['eofscor']]
        eofs = [e * sign_adjustments(e, r) for e, r in zip(eofs, reofs)]
        for e, r in zip(eofs, reofs):
            self.assert_array_almost_equal(e, r)

    def test_eofsAsCorrelation_range(self):
        # EOFs as correlation between PCs and input field should have values
        # in the range [-1, 1]
        eofs = [self._tomasked(e)
                for e in self.solver.eofsAsCorrelation(neofs=self.neofs)]
        for e in eofs:
            self.assert_true(np.abs(e).max() < 1.000000001)

    @pytest.mark.parametrize('pcscaling', (0, 1, 2))
    def test_pcs(self, pcscaling):
        # PCs should match the (possibly scaled) reference solution
        pcs = self._tomasked(self.solver.pcs(npcs=self.neofs,
                                             pcscaling=pcscaling))
        rpcs = self._tomasked(self.solution['pcs']).copy()
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        reigs = self._tomasked(self.solution['eigenvalues'])
        if pcscaling == 1:
            rpcs /= np.sqrt(reigs)
        elif pcscaling == 2:
            rpcs *= np.sqrt(reigs)
        self.assert_array_almost_equal(pcs, rpcs)

    @pytest.mark.parametrize('pcscaling', (0, 1, 2))
    def test_pcs_uncorrelated(self, pcscaling):
        # PCs should be uncorrelated in time
        pcs = self._tomasked(self.solver.pcs(npcs=self.neofs,
                                             pcscaling=pcscaling))
        correlation = np.corrcoef(pcs.transpose())
        residual = correlation - np.diag(correlation.diagonal())
        self.assert_array_almost_equal(residual, 0.)

    def test_variance(self):
        # variance explained as a percentage should match the reference
        # solution
        variance = self._tomasked(
            self.solver.varianceFraction(neigs=self.neofs)) * 100.
        self.assert_array_almost_equal(variance, self.solution['variance'])

    def test_totalAnomalyVariance(self):
        # total variance should match the sum of the reference solution
        # eigenvalues
        variance = self.solver.totalAnomalyVariance()
        rvariance = self._tomasked(self.solution['eigenvalues']).sum()
        self.assert_almost_equal(variance, rvariance)

    def test_reconstructedField(self):
        sst_fields = self.solver.reconstructedField(self.neofs)
        for sst, ref in zip(sst_fields, self.solution['sst']):
            self.assert_array_almost_equal(sst, ref)

    def test_reconstructedField_arb(self):
        sst_fields = self.solver.reconstructedField([1, 2, 5])
        for sst, ref in zip(sst_fields, self.solution['rcon']):
            self.assert_array_almost_equal(sst, ref)

    def test_getWeights(self):
        # analysis weights should match those from the reference solution
        weights = self.solver.getWeights()
        if weights is not None:
            for weight, ref in zip(weights, self.solution['weights']):
                if weight is not None:
                    self.assert_array_almost_equal(weight, ref)

    @pytest.mark.parametrize('vfscaled', (True, False))
    def test_northTest(self, vfscaled):
        # typical errors should match the reference solution
        errs = self.solver.northTest(neigs=self.neofs, vfscaled=vfscaled)
        error_name = 'scaled_errors' if vfscaled else 'errors'
        self.assert_array_almost_equal(errs, self.solution[error_name])

    @pytest.mark.parametrize('eofscaling', (0, 1, 2))
    def test_projectField(self, eofscaling):
        # original input projected onto the EOFs should match the reference
        # solution PCs
        pcs = self._tomasked(self.solver.projectField(self.solution['sst'],
                                                      neofs=self.neofs,
                                                      eofscaling=eofscaling))
        rpcs = self._tomasked(self.solution['pcs']).copy()
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        reigs = self._tomasked(self.solution['eigenvalues'])
        if eofscaling == 1:
            rpcs /= np.sqrt(reigs)
        elif eofscaling == 2:
            rpcs *= np.sqrt(reigs)
        self.assert_array_almost_equal(pcs, rpcs)

    def test_projectField_temporal_subset(self):
        # projecting a temporal subset of the original input onto the EOFs
        # should match the same subset of the reference PCs
        pcs = self._tomasked(
            self.solver.projectField([x[:5] for x in self.solution['sst']],
                                     neofs=self.neofs))
        rpcs = self._tomasked(self.solution['pcs'])[:5]
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        self.assert_array_almost_equal(pcs, rpcs)

    def test_projectField_no_time(self):
        # projecting the first time of the original input onto the EOFs should
        # match the first time of the reference PCs
        pcs = self._tomasked(
            self.solver.projectField([x[0] for x in self.solution['sst']],
                                     neofs=self.neofs))
        rpcs = self._tomasked(self.solution['pcs'])[0]
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        self.assert_array_almost_equal(pcs, rpcs)


# ----------------------------------------------------------------------------
# Tests for the standard interface


class StandardMVSolutionTest(MVSolutionTest):
    interface = 'standard'

    def _tomasked(self, value):
        return value


class TestStandardEqualWeights(StandardMVSolutionTest):
    weights = 'equal'


class TestStandardLatitudeWeights(StandardMVSolutionTest):
    weights = 'latitude'


class TestStandardAreaWeights(StandardMVSolutionTest):
    weights = 'area'


class TestStandardMixedWeights(StandardMVSolutionTest):
    weights = 'none_area'


# ----------------------------------------------------------------------------
# Tests for the cdms interface


class CDMSMVSolutionTest(MVSolutionTest):
    interface = 'cdms'

    def _tomasked(self, value):
        try:
            return value.asma()
        except AttributeError:
            return value


class TestCDMSEqualWeights(CDMSMVSolutionTest):
    weights = 'equal'


class TestCDMSLatitudeWeights(CDMSMVSolutionTest):
    weights = 'latitude'
    alternate_weights_arg = 'coslat'


class TestCDMSAreaWeights(CDMSMVSolutionTest):
    weights = 'area'
    alternate_weights_arg = 'area'


class TestCDMSMixedWeights(CDMSMVSolutionTest):
    weights = 'none_area'
    alternate_weights_arg = (None, 'area')


# ----------------------------------------------------------------------------
# Tests for the iris interface


class IrisMVSolutionTest(MVSolutionTest):
    interface = 'iris'

    def _tomasked(self, value):
        if type(value) is not Cube:
            return value
        return value.data


class TestIrisEqualWeights(IrisMVSolutionTest):
    weights = 'equal'


class TestIrisLatitudeWeights(IrisMVSolutionTest):
    weights = 'latitude'
    alternate_weights_arg = 'coslat'


class TestIrisAreaWeights(IrisMVSolutionTest):
    weights = 'area'
    alternate_weights_arg = 'area'


class TestIrisMixedWeights(IrisMVSolutionTest):
    weights = 'none_area'
    alternate_weights_arg = (None, 'area')
