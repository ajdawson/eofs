"""Test `eofs` computations against reference solutions."""
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
import numpy.ma as ma
try:
    from iris.cube import Cube
except ImportError:
    pass
import pytest

import eofs
from eofs.tests import EofsTest

from .utils import sign_adjustments
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


class SolutionTest(EofsTest):
    """Base class for all solution test classes."""
    interface = None
    weights = None
    alternate_weights_arg = None

    @classmethod
    def setup_class(cls):
        try:
            cls.solution = reference_solution(cls.interface, cls.weights)
        except ValueError:
            pytest.skip('missing dependencies required to test '
                        'the {!s} interface'.format(cls.interface))
        cls.modify_solution()
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

    @classmethod
    def modify_solution(cls):
        pass

    def test_eigenvalues(self):
        self.assert_array_almost_equal(
            self.solver.eigenvalues(neigs=self.neofs),
            self.solution['eigenvalues'])

    @pytest.mark.parametrize('eofscaling', (0, 1, 2))
    def test_eofs(self, eofscaling):
        # EOFs should match the (possibly scaled) reference solution
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs,
                                               eofscaling=eofscaling))
        reofs = self._tomasked(self.solution['eofs']).copy()
        eofs *= sign_adjustments(eofs, reofs)
        reigs = self._tomasked(self.solution['eigenvalues'])
        if eofscaling == 1:
            reofs /= np.sqrt(reigs)[:, np.newaxis, np.newaxis]
        elif eofscaling == 2:
            reofs *= np.sqrt(reigs)[:, np.newaxis, np.newaxis]
        self.assert_array_almost_equal(eofs, reofs)

    @pytest.mark.parametrize('eofscaling', (0, 1, 2))
    def test_eofs_orthogonal(self, eofscaling):
        # EOFs should be mutually orthogonal
        eofs = self._tomasked(self.solver.eofs(neofs=self.neofs,
                                               eofscaling=eofscaling))
        eofs = eofs.compressed()
        ns = eofs.shape[0] // self.neofs
        eofs = eofs.reshape([self.neofs, ns])
        dot = np.dot(eofs, eofs.T)
        residual = dot - np.diag(dot.diagonal())
        self.assert_array_almost_equal(residual, 0.)

    def test_eofsAsCovariance(self):
        # EOFs as covariance between PCs and input field should match the
        # reference solution
        eofs = self._tomasked(self.solver.eofsAsCovariance(neofs=self.neofs,
                                                           pcscaling=1))
        reofs = self._tomasked(self.solution['eofscov'])
        eofs *= sign_adjustments(eofs, reofs)
        self.assert_array_almost_equal(eofs, reofs)

    def test_eofsAsCorrelation(self):
        # EOFs as correlation between PCs and input field should match the
        # reference solution
        eofs = self._tomasked(self.solver.eofsAsCorrelation(neofs=self.neofs))
        reofs = self._tomasked(self.solution['eofscor'])
        eofs *= sign_adjustments(eofs, reofs)
        self.assert_array_almost_equal(eofs, reofs)

    def test_eofsAsCorrelation_range(self):
        # EOFs as correlation between PCs and input field should have values
        # in the range [-1, 1]
        eofs = self._tomasked(self.solver.eofsAsCorrelation(neofs=self.neofs))
        self.assert_true(np.abs(eofs).max() < 1.000000001)

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

    def test_getWeights(self):
        # analysis weights should match those from the reference solution
        weights = self.solver.getWeights()
        if weights is not None:
            self.assert_array_almost_equal(weights, self.solution['weights'])

    @pytest.mark.parametrize('vfscaled', (True, False))
    def test_northTest(self, vfscaled):
        # typical errors should match the reference solution
        errs = self.solver.northTest(neigs=self.neofs, vfscaled=vfscaled)
        error_name = 'scaled_errors' if vfscaled else 'errors'
        self.assert_array_almost_equal(errs, self.solution[error_name])

    def test_reconstructedField(self):
        # reconstructed field using all EOFs should match the original input
        sst = self.solver.reconstructedField(self.solver.neofs)
        self.assert_array_almost_equal(sst, self.solution['sst'])

    def test_reconstructedField_arb(self):
        sst = self.solver.reconstructedField([1, 2, 5])
        self.assert_array_almost_equal(sst, self.solution['rcon'])

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
            self.solver.projectField(self.solution['sst'][:5],
                                     neofs=self.neofs))
        rpcs = self._tomasked(self.solution['pcs'])[:5]
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        self.assert_array_almost_equal(pcs, rpcs)

    def test_projectField_no_time(self):
        # projecting the first time of the original input onto the EOFs should
        # match the first time of the reference PCs
        pcs = self._tomasked(
            self.solver.projectField(self.solution['sst'][0],
                                     neofs=self.neofs))
        rpcs = self._tomasked(self.solution['pcs'])[0]
        pcs *= sign_adjustments(pcs.transpose(), rpcs.transpose()).transpose()
        self.assert_array_almost_equal(pcs, rpcs)


# ----------------------------------------------------------------------------
# Tests for the standard interface


class StandardSolutionTest(SolutionTest):
    """Base class for all standard interface solution test classes."""
    interface = 'standard'

    def _tomasked(self, value):
        return value


class TestStandardEqualWeights(StandardSolutionTest):
    """Equal grid weighting."""
    weights = 'equal'


class TestStandardLatitudeWeights(StandardSolutionTest):
    """Square-root of cosine of latitude grid weighting."""
    weights = 'latitude'


class TestStandardAreaWeights(StandardSolutionTest):
    """Square-root of normalised grid cell area grid weighting."""
    weights = 'area'


class TestStandardMissingValuesAsNaN(StandardSolutionTest):
    """
    Missing values replaced with not-a-number values and equal grid
    weighting.

    """
    weights = 'equal'

    @classmethod
    def modify_solution(cls):
        for name in cls.solution:
            try:
                cls.solution[name] = cls.solution[name].filled(
                    fill_value=np.nan)
            except AttributeError:
                pass

    def _tomasked(self, value):
        return ma.MaskedArray(value, mask=np.isnan(value))


# ----------------------------------------------------------------------------
# Tests for the xarray interface


class XarraySolutionTest(SolutionTest):
    interface = 'xarray'

    def _tomasked(self, value):
        try:
            return ma.masked_invalid(value.values)
        except AttributeError:
            return ma.masked_invalid(value)


class TestXarrayEqualWeights(XarraySolutionTest):
    weights = 'equal'


# ----------------------------------------------------------------------------
# Tests for the cdms interface


class CDMSSolutionTest(SolutionTest):
    """Base class for all cdms interface solution test classes."""
    interface = 'cdms'

    def _tomasked(self, value):
        try:
            return value.asma()
        except AttributeError:
            return value


class TestCDMSEqualWeights(CDMSSolutionTest):
    """Equal grid weighting."""
    weights = 'equal'


class TestCDMSLatitudeWeights(CDMSSolutionTest):
    """
    Square-root of cosine of latitude grid weighting (automatically
    generated weights).

    """
    weights = 'latitude'
    alternate_weights_arg = 'coslat'


class TestCDMSAreaWeights(CDMSSolutionTest):
    """
    Square-root of normalised grid cell area grid weighting
    (automatically generated weights).

    """
    weights = 'area'
    alternate_weights_arg = 'area'


class TestCDMSAreaWeightsTransposedGrid(CDMSSolutionTest):
    """
    Square-root of normalised grid cell area grid weighting
    (automatically generated weights) after transposing grid variables
    to a longitude-latitude grid.

    """
    weights = 'area'
    alternate_weights_arg = 'area'

    @classmethod
    def modify_solution(cls):
        cls.solution['sst'] = cls.solution['sst'].reorder('txy')
        cls.solution['eofs'] = cls.solution['eofs'].reorder('-xy')
        cls.solution['eofscor'] = cls.solution['eofscor'].reorder('-xy')
        cls.solution['eofscov'] = cls.solution['eofscov'].reorder('-xy')
        cls.solution['weights'] = cls.solution['weights'].transpose()
        cls.solution['rcon'] = cls.solution['rcon'].reorder('-xy')


class TestCDMSLatitudeWeightsManual(CDMSSolutionTest):
    """
    Square-root of cosine of latitude grid weighting (weights from
    reference solution).

    """
    weights = 'latitude'


class TestCDMSAreaWeightsManual(CDMSSolutionTest):
    """
    Square-root of normalised grid cell area grid weighting (weights
    from reference solution).

    """
    weights = 'area'


# ----------------------------------------------------------------------------
# Tests for the iris interface


class IrisSolutionTest(SolutionTest):
    """Base class for all iris interface solution test classes."""
    interface = 'iris'

    def _tomasked(self, value):
        if type(value) is not Cube:
            return value
        return value.data


class TestIrisEqualWeights(IrisSolutionTest):
    """Equal grid weighting."""
    weights = 'equal'


class TestIrisLatitudeWeights(IrisSolutionTest):
    """
    Square-root of cosine of latitude grid weighting (automatically
    generated weights).

    """
    weights = 'latitude'
    alternate_weights_arg = 'coslat'


class TestIrisAreaWeights(IrisSolutionTest):
    """
    Square-root of normalised grid cell area grid weighting
    (automatically generated weights).

    """
    weights = 'area'
    alternate_weights_arg = 'area'


class TestIrisLatitudeWeightsManual(IrisSolutionTest):
    """
    Square-root of cosine of latitude grid weighting (weights from
    reference solution).

    """
    weights = 'latitude'


class TestIrisAreaWeightsManual(IrisSolutionTest):
    """
    Square-root of normalised grid cell area grid weighting (weights
    from reference solution).

    """
    weights = 'area'
