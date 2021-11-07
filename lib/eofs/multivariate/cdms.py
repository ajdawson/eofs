"""Multivariate EOF analysis for `cdms2` variables."""
# (c) Copyright 2013-2015 Andrew Dawson. All Rights Reserved.
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

import collections

import cdms2

from eofs.tools.cdms import weights_array, cdms2_name
from . import standard


class MultivariateEof(object):
    """Multivariate EOF analysis (meta-data enabled `cdms2` interface)"""

    def __init__(self, datasets, weights=None, center=True, ddof=1):
        """Create a MultivariateEof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *datasets*
            A list/tuple containing one or more `cdms2` variables, each
            two or more dimensions, containing the data to be analysed.
            Time must be the first dimension of each variable. Missing
            values are allowed provided that they are constant with time
            in each field (e.g., values of an oceanographic field over
            land).

        **Optional arguments:**

        *weights*
            Sets the weighting method. One method can be chosen to apply
            to all variables in *datasets* or a sequence of options can
            be given to specify a different weighting method for each
            variable in *datasets*. The following pre-defined weighting
            methods are available:

            * *'area'* : Square-root of grid cell area normalized by
              total grid area. Requires a latitude-longitude grid to be
              present in the corresponding `cdms2` variable. This is a
              fairly standard weighting strategy. If you are unsure
              which method to use and you have gridded data then this
              should be your first choice.

            * *'coslat'* : Square-root of cosine of latitude. Requires a
              latitude dimension to be present in the corresponding
              `cdms2` variable.

            * *None* : Equal weights for all grid points (*'none'* is
              also accepted).

             Alternatively a sequence of arrays of weights whose shapes
             are compatible with the corresponding `cdms2` variables in
             *datasets* may be supplied instead of specifying a
             weighting method.

        *center*
            If *True*, the mean along the first axis of each variable in
            *datasets* (the time-mean) will be removed prior to
            analysis. If *False*, the mean along the first axis will not
            be removed. Defaults to *True* (mean is removed).

            The covariance interpretation relies on the input data being
            anomalies with a time-mean of 0. Therefore this option
            should usually be set to *True*. Setting this option to
            *True* has the useful side effect of propagating missing
            values along the time dimension, ensuring that a solution
            can be found even if missing values occur in different
            locations at different times.

        *ddof*
            'Delta degrees of freedom'. The divisor used to normalize
            the covariance matrix is *N - ddof* where *N* is the
            number of samples. Defaults to *1*.

        **Returns:**

        *solver*
            An `MultivariateEof` instance.

        **Examples:**

        EOF analysis with grid-cell-area weighting using two input
        fields::

            from eofs.multivariate.cdms import MultivariateEof
            solver = MultivariateEof([var1, var2], weights='area')

        """
        # Record the number of datasets.
        self._ndata = len(datasets)
        # Ensure the weights are specified one per dataset.
        if weights in ('none', None, 'area', 'coslat'):
            weights = [weights] * self._ndata
        elif len(weights) != self._ndata:
            raise ValueError('number of weights is incorrect, '
                             'expecting {:d} but got {:d}'.format(
                                 self._ndata, len(weights)))
        # Record dimension information, missing values and compute weights.
        self._multitimeaxes = list()
        self._multichannels = list()
        self._multimissing = list()
        passweights = list()
        for dataset, weight in zip(datasets, weights):
            if not cdms2.isVariable(dataset):
                raise TypeError('the input data set must be a cdms2 variable')
            # Ensure a time dimension exists.
            timeaxis = dataset.getTime()
            if timeaxis is None:
                raise ValueError('time axis not found')
            self._multitimeaxes.append(timeaxis)
            # Ensure the time dimension is the first dimension.
            order = dataset.getOrder()
            if order[0] != "t":
                raise ValueError('time must be the first dimension, '
                                 'consider using the reorder() method')
            # Record the other dimensions.
            channels = dataset.getAxisList()
            channels.remove(timeaxis)
            if len(channels) < 1:
                raise ValueError('one or more spatial dimensions are required')
            self._multichannels.append(channels)
            # Record the missing values.
            self._multimissing.append(dataset.getMissing())
            # Compute weights as required.
            if weight in ("none", None):
                passweights.append(None)
            else:
                try:
                    wtarray = weights_array(dataset, scheme=weight.lower())
                    passweights.append(wtarray)
                except AttributeError:
                    # Weight specification is not a string. Assume it is an
                    # array of weights.
                    passweights.append(weight)
                # any other error will be raised
        # Define a time axis as the time axis of the first dataset.
        self._timeax = self._multitimeaxes[0]
        # Create a MultipleEofSolver to do the computations.
        self._solver = standard.MultivariateEof([d.asma() for d in datasets],
                                                weights=passweights,
                                                center=center,
                                                ddof=ddof)
        #: Number of EOFs in the solution.
        self.neofs = self._solver.neofs
        # Names of the input variables.
        self._dataset_names = [cdms2_name(v).replace(' ', '_')
                               for v in datasets]
        self._dataset_ids = [dataset.id for dataset in datasets]

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled PCs (default).
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of PCs requested is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *pcs*
            A `cdms2` variable containing the ordered PCs.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        pcs = self._solver.pcs(pcscaling, npcs)
        pcsax = cdms2.createAxis(list(range(pcs.shape[1])), id='pc')
        pcsax.long_name = 'pc_number'
        axlist = [self._timeax, pcsax]
        pcs = cdms2.createVariable(pcs, id='pcs', axes=axlist)
        pcs.long_name = 'principal_components'
        return pcs

    def eofs(self, eofscaling=0, neofs=None):
        """Empirical orthogonal functions (EOFs).

        **Optional arguments:**

        *eofscaling*
            Sets the scaling of the EOFs. The following values are
            accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalues.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalues.

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs_list*
            A list of `cdms2` variables containing the ordered EOFs for
            each variable.

        **Examples:**

        All EOFs with no scaling::

            eofs_list = solver.eofs()

        The leading EOF with scaling applied::

            eof1_list = solver.eofs(neofs=1, eofscaling=1)

        """
        eofset = self._solver.eofs(eofscaling, neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(list(range(neofs)), id='eof')
        eofax.long_name = 'eof_number'
        for iset in range(self._ndata):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset].fill_value = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(
                eofset[iset],
                id='eofs',
                axes=axlist,
                fill_value=self._multimissing[iset])
            eofset[iset].long_name = 'empirical_orthogonal_functions'
        return eofset

    def eofsAsCorrelation(self, neofs=None):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        correlation between the principal component time series (PCs)
        and the each data set in the `MultivariateEof` input *datasets*.

        .. note::

            These are not related to the EOFs computed from the
            correlation matrix.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs_list*
            A list of `cdms2` variables containing the ordered EOFs for
            each variable.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCorrelation()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCorrelation(neofs=1)

        """
        eofset = self._solver.eofsAsCorrelation(neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(list(range(neofs)), id='eof')
        eofax.long_name = 'eof_number'
        for iset in range(self._ndata):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset].fill_value = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(
                eofset[iset],
                id='eofs',
                axes=axlist,
                fill_value=self._multimissing[iset])
            eofset[iset].long_name = 'correlation_between_pcs_and_{:s}'.format(
                self._dataset_names[iset])
        return eofset

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        covariance between the principal component time series (PCs)
        and the each data set in the `MultivariateEof` input *datasets*.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        *pcscaling*
            Set the scaling of the PCs used to compute covariance. The
            following values are accepted:

            * *0* : Un-scaled PCs.
            * *1* : PCs are scaled to unit variance (divided by the
              square-root of their eigenvalue) (default).
            * *2* : PCs are multiplied by the square-root of their
              eigenvalue.

            The default is to divide PCs by the square-root of their
            eigenvalue so that the PCs are scaled to unit variance
            (option 1).

        **Returns:**

        *eofs_list*
            A list of `cdms2` variables containing the ordered EOFs for
            each variable.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCovariance()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCovariance(neofs=1)

        """
        eofset = self._solver.eofsAsCovariance(neofs)
        neofs = eofset[0].shape[0]
        eofax = cdms2.createAxis(list(range(neofs)), id='eof')
        eofax.long_name = 'eof_number'
        for iset in range(self._ndata):
            axlist = [eofax] + self._multichannels[iset]
            eofset[iset].fill_value = self._multimissing[iset]
            eofset[iset] = cdms2.createVariable(
                eofset[iset],
                id='eofs',
                axes=axlist,
                fill_value=self._multimissing[iset])
            eofset[iset].long_name = 'covariance_between_pcs_and_{:s}'.format(
                self._dataset_names[iset])
        return eofset

    def eigenvalues(self, neigs=None):
        """
        Eigenvalues (decreasing variances) associated with each EOF
        mode.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues. If the number of eigenvalues requested is more
            than the number that are available, then all available
            eigenvalues will be returned.

        **Returns:**

        *eigenvalues*
            A `cdms2` variable containing the eigenvalues arranged
            largest to smallest.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        lambdas = self._solver.eigenvalues(neigs=neigs)
        eofax = cdms2.createAxis(list(range(len(lambdas))), id='eigenvalue')
        eofax.long_name = 'eigenvalue_number'
        axlist = [eofax]
        lambdas = cdms2.createVariable(lambdas, id='eigenvalues', axes=axlist)
        lambdas.long_name = 'eigenvalues'
        return lambdas

    def varianceFraction(self, neigs=None):
        """Fractional EOF mode variances.

        The fraction of the total variance explained by each EOF mode,
        values between 0 and 1 inclusive.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues. If the number of eigenvalues
            requested is more than the number that are available, then
            fractional variances for all available eigenvalues will be
            returned.

        **Returns:**

        *variance_fractions*
            A `cdms2` variable containing the fractional variances.

        **Examples:**

        The fractional variance represented by each EOF mode::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first EOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        vfrac = self._solver.varianceFraction(neigs=neigs)
        eofax = cdms2.createAxis(list(range(len(vfrac))), id='eigenvalue')
        axlist = [eofax]
        vfrac = cdms2.createVariable(vfrac, id='variance_fraction',
                                     axes=axlist)
        vfrac.long_name = 'variance_fraction'
        return vfrac

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).

        **Returns:**

        *total_variance*
            A scalar value (not a `cdms2` variable).

        **Example:**

        Get the total variance::

            total_variance = solver.totalAnomalyVariance()

        """
        return self._solver.totalAnomalyVariance()

    def northTest(self, neigs=None, vfscaled=False):
        """Typical errors for eigenvalues.

        The method of North et al. (1982) is used to compute the typical
        error for each eigenvalue. It is assumed that the number of
        times in the input data set is the same as the number of
        independent realizations. If this assumption is not valid then
        the result may be inappropriate.

        **Optional arguments:**

        *neigs*
            The number of eigenvalues to return typical errors for.
            Defaults to typical errors for all eigenvalues. If the
            number of eigenvalues requested is more than the number that
            are available, then typical errors for all available
            eigenvalues will be returned.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `MultivariateEof.varianceFraction`. If *False*
            then no scaling is done. Defaults to *False* (no scaling).

        **Returns:**

        *errors*
            A `cdms2` variable containing the typical errors.

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Examples:**

        Typical errors for all eigenvalues::

            errors = solver.northTest()

        Typical errors for the first 5 eigenvalues scaled by the sum of
        the eigenvalues::

            errors = solver.northTest(neigs=5, vfscaled=True)

        """
        typerrs = self._solver.northTest(neigs=neigs, vfscaled=vfscaled)
        eofax = cdms2.createAxis(list(range(len(typerrs))), id='eigenvalue')
        eofax.long_name = 'eof_number'
        axlist = [eofax]
        typerrs = cdms2.createVariable(typerrs,
                                       id='typical_errors',
                                       axes=axlist)
        typerrs.long_name = 'typical_errors'
        return typerrs

    def reconstructedField(self, neofs):
        """Reconstructed data sets based on a subset of EOFs.

        If weights were passed to the `MultivariateEof` instance the
        returned reconstructed fields will automatically have this
        weighting removed. Otherwise each returned field will have the
        same weighting as the corresponding  array in the
        `MultivariateEof` input *datasets*.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be used for the
            reconstruction. Alternatively this argument can be an
            iterable of mode numbers (where the first mode is 1) in
            order to facilitate reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction_list*
            A list of `cdms2` variable with the same dimensions as the
            variables in the `MultivariateEof` input *datasets*
            contaning the reconstructions using *neofs* EOFs.

        **Example:**

        Reconstruct the input data sets using 3 EOFs::

            reconstruction_list = solver.reconstructedField(neofs=3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction_list = solver.reconstuctedField([1, 2, 5])

        """
        rfset = self._solver.reconstructedField(neofs)
        if isinstance(neofs, collections.Iterable):
            name_part = 'EOFs_{}'.format('_'.join([str(e) for e in neofs]))
        else:
            name_part = '{:d}_EOFs'.format(neofs)
        for iset in range(self._ndata):
            axlist = [self._multitimeaxes[iset]] + self._multichannels[iset]
            rfset[iset].fill_value = self._multimissing[iset]
            rfset[iset] = cdms2.createVariable(
                rfset[iset],
                id=self._dataset_ids[iset],
                axes=axlist,
                fill_value=self._multimissing[iset])
            rfset[iset].long_name = '{:s}_reconstructed_with_{:s}'.format(
                self._dataset_names[iset], name_part)
            rfset[iset].neofs = neofs
        return rfset

    def projectField(self, fields, neofs=None, eofscaling=0, weighted=True):
        """Project a set of fields onto the EOFs.

        Given a set of fields, projects them onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *fields*
            A list/tuple containing one or more `cdms2` variables, each
            with two or more dimensions, containing the data to be
            projected onto the EOFs. Each field must have the same
            spatial dimensions (including missing values in the same
            places) as the corresponding data set in the
            `MultivariateEof` input *datasets*. The fields may have
            different length time dimensions to the `MultivariateEof`
            inputs *datasets* or no time dimension at all, but this
            must be consistent for all fields.

        **Optional arguments:**

        *neofs*
            Number of EOFs to project onto. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then the field will be projected onto all
            available EOFs.

        *eofscaling*
            Set the scaling of the EOFs that are projected
            onto. The following values are accepted:

            * *0* : Un-scaled EOFs (default).
            * *1* : EOFs are divided by the square-root of their
              eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then each field in *fields* is weighted using the
            same weights used for the EOF analysis prior to projection.
            If *False* then no weighting is applied. Defaults to *True*
            (weighting is applied). Generally only the default setting
            should be used.

        **Returns:**

        *pseudo_pcs*
            A `cdms2` variable containing the ordered pseudo-PCs.

        **Examples:**

        Project a data set onto all EOFs::

            pseudo_pcs = solver.projectField([field1, field2])

        Project a data set onto the four leading EOFs::

            pseudo_pcs = solver.projectField([field1, field2], neofs=4)

        """
        for field in fields:
            if not cdms2.isVariable(field):
                raise TypeError('the input data set must be a cdms2 variable')
        if len(fields) != self._ndata:
            raise ValueError('number of fields is incorrect, expecting {:d} '
                             'but got {:d}'.format(self._ndata, len(fields)))
        for field in fields:
            order = field.getOrder()
            if 't' in order:
                if order[0] != 't':
                    raise ValueError('time must be the first dimension, '
                                     'consider using the reorder() method')
        pcs = self._solver.projectField([f.asma() for f in fields],
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        # Create an axis list, its contents depend on whether or not a time
        # axis was present in the input field.
        if pcs.ndim == 2:
            # Time dimension present:
            pcsax = cdms2.createAxis(list(range(pcs.shape[1])), id='pc')
            pcsax.long_name = 'pc_number'
            axlist = [fields[0].getAxis(0), pcsax]
        else:
            # A PC axis and a leading time axis.
            pcsax = cdms2.createAxis(list(range(pcs.shape[0])), id='pc')
            pcsax.long_name = 'pc_number'
            axlist = [pcsax]
        # Apply meta data to the projected PCs.
        pcs = cdms2.createVariable(pcs, id='pseudo_pcs', axes=axlist)
        pcs.long_name = 'psuedo_pcs'
        return pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights_list*
            A list of arrays containing the analysis weights for each
            variable (not `cdms2` variables).

        **Example:**

        The weights used for the analysis::

            weights_list = solver.getWeights()

        """
        return self._solver.getWeights()
