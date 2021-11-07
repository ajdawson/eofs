"""Meta-data preserving EOF analysis for `cdms2`."""
# (c) Copyright 2010-2015 Andrew Dawson. All Rights Reserved.
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

from . import standard
from .tools.cdms import weights_array, cdms2_name


class Eof(object):
    """EOF analysis (meta-data enabled `cdms2` interface)"""

    def __init__(self, dataset, weights=None, center=True, ddof=1):
        """Create an Eof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Argument:**

        *dataset*
            A `cdms2` variable containing the data to be analysed. Time
            must be the first dimension. Missing values are allowed
            provided that they are constant with time (e.g., values of
            an oceanographic field over land).

        **Optional arguments:**

        *weights*
            Sets the weighting method. The following pre-defined
            weighting methods are available:

            * *'area'* : Square-root of grid cell area normalized by
              total grid area. Requires a latitude-longitude grid to be
              present in the `cdms2` variable *dataset*. This is a
              fairly standard weighting strategy. If you are unsure
              which method to use and you have gridded data then this
              should be your first choice.

            * *'coslat'* : Square-root of cosine of latitude. Requires a
              latitude dimension to be present in the `cdms2` variable
              *dataset*.

            * *None* : Equal weights for all grid points (*'none'* is
              also accepted).

             Alternatively an array of weights whose shape is compatible
             with the `cdms2` variable *dataset* may be supplied instead
             of specifying a weighting method.

        *center*
            If *True*, the mean along the first axis of *dataset* (the
            time-mean) will be removed prior to analysis. If *False*,
            the mean along the first axis will not be removed. Defaults
            to *True* (mean is removed).

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
            An `Eof` instance.

        **Examples:**

        EOF analysis with grid-cell-area weighting for the input field::

            from eofs.cdms import Eof
            solver = Eof(dataset, weights='area')

        """
        # Check that dataset is recognised by cdms2 as a variable.
        if not cdms2.isVariable(dataset):
            raise TypeError('the input data must be a cdms2 variable')
        # Store the time axis as an instance variable.
        self._timeax = dataset.getTime()
        # Verify that a time axis was found, getTime returns None when a
        # time axis is not found.
        if self._timeax is None:
            raise ValueError('time axis not found')
        # Check the dimension order of the input, time must be the first
        # dimension.
        order = dataset.getOrder()
        if order[0] != 't':
            raise ValueError('time must be the first dimension, '
                             'consider using the reorder() method')
        # Verify the presence of at least one spatial dimension. The
        # instance variable channels will also be used as a partial axis
        # list when constructing meta-data. It contains the spatial
        # dimensions.
        self._channels = dataset.getAxisList()
        self._channels.remove(self._timeax)
        if len(self._channels) < 1:
            raise ValueError('one or more spatial dimensions are required')
        # Store the missing value attribute of the data set in an
        # instance variable so that it is recoverable later.
        self._missing_value = dataset.getMissing()
        # Generate an appropriate set of weights for the input dataset. There
        # are several weighting schemes. The 'area' weighting scheme requires
        # a latitude-longitude grid to be present, the 'cos_lat' scheme only
        # requires a latitude dimension.
        if weights is None or weights == 'none':
            # No weights requested, set the weight array to None.
            wtarray = None
        else:
            try:
                # Generate a weights array of the appropriate kind, with a
                # shape compatible with the data set.
                scheme = weights.lower()
                wtarray = weights_array(dataset, scheme=scheme)
            except AttributeError:
                # Weights is not a string, assume it is an array.
                wtarray = weights
            except ValueError as err:
                # Weights is not recognized, raise an error.
                raise ValueError(err)
        # Cast the wtarray to the same type as the dataset. This prevents the
        # promotion of 32-bit input to 64-bit on multiplication with the
        # weight array when not required. This will fail with a AttributeError
        # exception if the weights array is None, which it may be if no
        # weighting was requested.
        try:
            wtarray = wtarray.astype(dataset.dtype)
        except AttributeError:
            pass
        # Create an EofSolver object using appropriate arguments for this
        # data set. The object will be used for the decomposition and
        # for returning the results.
        self._solver = standard.Eof(dataset.asma(),
                                    weights=wtarray,
                                    center=center,
                                    ddof=ddof)
        #: Number of EOFs in the solution.
        self.neofs = self._solver.neofs
        # name for the dataset.
        self._dataset_name = cdms2_name(dataset).replace(' ', '_')
        self._dataset_id = dataset.id

    def pcs(self, pcscaling=0, npcs=None):
        """Principal component time series (PCs).

        **Optional arguments:**

        *pcscaling*
            Set the scaling of the retrieved PCs. The following
            values are accepted:

            * *0* : Un-scaled principal components (default).
            * *1* : Principal components are scaled to unit variance
              (divided by the square-root of their eigenvalue).
            * *2* : Principal components are multiplied by the
              square-root of their eigenvalue.

        *npcs*
            Number of PCs to retrieve. Defaults to all the PCs. If the
            number of requested PCs is more than the number that are
            available, then all available PCs will be returned.

        **Returns:**

        *pcs*
            A `cdms2` variable containing the ordered PCs. The PCs are
            numbered from 0 to *npcs* - 1.

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
        """Emipirical orthogonal functions (EOFs).

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

        *eofs*
           A `cdms2` variable containing the ordered EOFs. The EOFs are
           numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs with no scaling::

            eofs = solver.eofs()

        First 3 EOFs with scaling applied::

            eofs = solver.eofs(neofs=3, eofscaling=1)

        """
        eofs = self._solver.eofs(eofscaling, neofs)
        eofs.fill_value = self._missing_value
        eofax = cdms2.createAxis(list(range(len(eofs))), id='eof')
        eofax.long_name = 'eof_number'
        axlist = [eofax] + self._channels
        eofs = cdms2.createVariable(eofs,
                                    id='eofs',
                                    axes=axlist,
                                    fill_value=self._missing_value)
        eofs.long_name = 'empirical_orthogonal_functions'
        return eofs

    def eofsAsCorrelation(self, neofs=None):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        correlation between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        .. note::

            These are not related to the EOFs computed from the
            correlation matrix.

        **Optional argument:**

        *neofs*
            Number of EOFs to return. Defaults to all EOFs. If the
            number of EOFs requested is more than the number that are
            available, then all available EOFs will be returned.

        **Returns:**

        *eofs*
           A `cdms2` variable containing the ordered EOFs. The EOFs are
           numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCorrelation()

        The leading EOF::

            eof1 = solver.eofsAsCorrelation(neofs=1)

        """
        eofs = self._solver.eofsAsCorrelation(neofs)
        eofs.fill_value = self._missing_value
        eofax = cdms2.createAxis(list(range(len(eofs))), id='eof')
        eofax.long_name = 'eof_number'
        axlist = [eofax] + self._channels
        eofs = cdms2.createVariable(eofs,
                                    id='eofs',
                                    axes=axlist,
                                    fill_value=self._missing_value)
        eofs.long_name = 'correlation_between_pcs_and_{:s}'.format(
            self._dataset_name)
        return eofs

    def eofsAsCovariance(self, neofs=None, pcscaling=1):
        """
        Empirical orthogonal functions (EOFs) expressed as the
        covariance between the principal component time series (PCs)
        and the time series of the `Eof` input *dataset* at each grid
        point.

        **Optional arguments:**

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

        *eofs*
           A `cdms2` variable containing the ordered EOFs. The EOFs are
           numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCovariance()

        The leading EOF::

            eof1 = solver.eofsAsCovariance(neofs=1)

        The leading EOF using un-scaled PCs::

            eof1 = solver.eofsAsCovariance(neofs=1, pcscaling=0)

        """
        eofs = self._solver.eofsAsCovariance(neofs, pcscaling)
        eofs.fill_value = self._missing_value
        eofax = cdms2.createAxis(list(range(len(eofs))), id='eof')
        axlist = [eofax] + self._channels
        eofs = cdms2.createVariable(eofs,
                                    id='eofs_cov',
                                    axes=axlist,
                                    fill_value=self._missing_value)
        eofs.long_name = 'covariance_between_pcs_and_{:s}'.format(
            self._dataset_name)
        return eofs

    def eigenvalues(self, neigs=None):
        """Eigenvalues (decreasing variances) associated with each EOF.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return. Defaults to all
            eigenvalues.If the number of eigenvalues requested is more
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
        """Fractional EOF variances.

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
            A `cdms2` variable containing the fractional variances for
            each eigenvalue. The eigenvalues are numbered from 0 to
            *neigs* - 1.

        **Examples:**

        The fractional variance represented by each eigenvalue::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first 3 eigenvalues::

            variance_fractions = solver.VarianceFraction(neigs=3)

        """
        vfrac = self._solver.varianceFraction(neigs=neigs)
        eofax = cdms2.createAxis(list(range(len(vfrac))), id='eigenvalue')
        eofax.long_name = 'eigenvalue_number'
        axlist = [eofax]
        vfrac = cdms2.createVariable(vfrac, id='variance_fractions',
                                     axes=axlist)
        vfrac.long_name = 'variance_fractions'
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
            Defaults to typical errors for all eigenvalues.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `Eof.varianceFraction`. If *False* then no
            scaling is done. Defaults to *False* (no scaling).

        **Returns:**

        *errors*
            A `cdms2` variable containing the typical errors for each
            eigenvalue. The egienvalues are numbered from 0 to
            *neigs* - 1.

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Examples:**

        Typical errors for all eigenvalues::

            errors = solver.northTest()

        Typical errors for the first 3 eigenvalues scaled by the sum of
        the eigenvalues::

            errors = solver.northTest(neigs=3, vfscaled=True)

        """
        typerrs = self._solver.northTest(neigs=neigs, vfscaled=vfscaled)
        eofax = cdms2.createAxis(list(range(len(typerrs))), id='eigenvalue')
        eofax.long_name = 'eigenvalue_number'
        axlist = [eofax]
        typerrs = cdms2.createVariable(typerrs,
                                       id='typical_errors',
                                       axes=axlist)
        typerrs.long_name = 'typical_errors'
        return typerrs

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the `Eof` instance the returned
        reconstructed field will automatically have this weighting
        removed. Otherwise the returned field will have the same
        weighting as the `Eof` input *dataset*.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction.
            Alternatively this argument can be an iterable of mode
            numbers (where the first mode is 1) in order to facilitate
            reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            A `cdms2` variable with the same dimensions `Eof` input
            *dataset* containing the reconstruction using *neofs* EOFs.

        **Example:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        rfield = self._solver.reconstructedField(neofs)
        rfield.fill_value = self._missing_value
        axlist = [self._timeax] + self._channels
        rfield = cdms2.createVariable(rfield,
                                      id=self._dataset_id,
                                      axes=axlist,
                                      fill_value=self._missing_value)
        if isinstance(neofs, collections.Iterable):
            name_part = 'EOFs_{}'.format('_'.join([str(e) for e in neofs]))
        else:
            name_part = '{:d}_EOFs'.format(neofs)
        rfield.long_name = '{:s}_reconstructed_with_{:s}'.format(
            self._dataset_name, name_part)
        rfield.neofs = neofs
        return rfield

    def projectField(self, field, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a data set, projects it onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *field*
            A `cdms2` variable containing the field to project onto the
            EOFs. It must have the same corresponding spatial dimensions
            (including missing values in the same places) as the `Eof`
            input *dataset*. It may have a different length time
            dimension to the `Eof` input *dataset* or no time dimension
            at all. If a time dimension exists it must be the first
            dimension.

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
            * *1* : EOFs are divided by the square-root of their eigenvalue.
            * *2* : EOFs are multiplied by the square-root of their
              eigenvalue.

        *weighted*
            If *True* then the field is weighted using the same weights
            used for the EOF analysis prior to projection. If *False*
            then no weighting is applied. Defaults to *True* (weighting
            is applied). Generally only the default setting should be
            used.

        **Returns:**

        *pseudo_pcs*
            A `cdms2` variable containing the pseudo-PCs. The PCs are
            numbered from 0 to *neofs* - 1.

        **Examples:**

        Project a field onto all EOFs::

            pseudo_pcs = solver.projectField(field)

        Project fields onto the three leading EOFs::

            pseudo_pcs = solver.projectField(field, neofs=3)

        """
        # Check that field is recognised by cdms2 as a variable.
        if not cdms2.isVariable(field):
            raise TypeError('the input field must be a cdms2 variable')
        dataset_name = cdms2_name(field).replace(' ', '_')
        # Compute the projected PCs.
        pcs = self._solver.projectField(field.asma(),
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        # Construct the required axes.
        if pcs.ndim == 2:
            # 2D PCs require a time axis and a PC axis.
            pcsax = cdms2.createAxis(list(range(pcs.shape[1])), id='pc')
            pcsax.long_name = 'pc_number'
            timeax = field.getAxis(0)  # time is assumed to be first anyway
            axlist = [timeax, pcsax]
        else:
            # 1D PCs require only a PC axis.
            pcsax = cdms2.createAxis(list(range(pcs.shape[0])), id='pc')
            pcsax.long_name = 'pc_number'
            axlist = [pcsax]
        # Apply meta data to the projected PCs.
        pcs = cdms2.createVariable(pcs, id='pseudo_pcs', axes=axlist)
        pcs.long_name = '{:s}_pseudo_pcs'.format(dataset_name)
        return pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights*
            An array contaning the analysis weights (not a `cdms2`
            variable).

        **Example:**

        The weights used for the analysis::

            weights = solver.getWeights()

        """
        return self._solver.getWeights()
