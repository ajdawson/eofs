"""Meta-data preserving EOF analysis for `iris`."""
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

import collections
from copy import copy

from iris.cube import Cube
from iris.coords import DimCoord

from . import standard
from .tools.iris import get_time_coord, weights_array, classified_aux_coords


class Eof(object):
    """EOF analysis (meta-data enabled `iris` interface)"""

    def __init__(self, cube, weights=None, center=True, ddof=1):
        """Create an Eof object.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Argument:**

        *dataset*
            A `~iris.cube.Cube` instance containing the data to be
            analysed. Time must be the first dimension. Missing values
            are allowed provided that they are constant with time (e.g.,
            values of an oceanographic field over land).

        **Optional arguments:**

        *weights*
            Sets the weighting method. The following pre-defined
            weighting methods are available:

            * *'area'* : Square-root of grid cell area normalized by
              total grid area. Requires a latitude-longitude grid to be
              present in the `~iris.cube.Cube` *dataset*. This is a
              fairly standard weighting strategy. If you are unsure
              which method to use and you have gridded data then this
              should be your first choice.

            * *'coslat'* : Square-root of cosine of latitude. Requires a
              latitude dimension to be present in the `~iris.cube.Cube`
              *dataset*.

            * *None* : Equal weights for all grid points (*'none'* is
              also accepted).

             Alternatively an array of weights whose shape is compatible
             with the `~iris.cube.Cube` *dataset* may be supplied instead
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

            from eofs.iris import Eof
            solver = Eof(cube, weights='area')

        """
        # Check that the input is an Iris cube.
        if not isinstance(cube, Cube):
            raise TypeError('the input must be an iris cube')
        # Check for a time coordinate, raise an error if there isn't one.
        # The get_time_coord function will raise a ValuerError with a
        # useful message so no need to handle it explicitly here.
        _time, self._time_dim = get_time_coord(cube)
        self._time = copy(_time)
        if self._time_dim != 0:
            raise ValueError('time must be the first dimension, '
                             'consider using the transpose() method')
        # Get the cube coordinates and remove time, leaving just the other
        # dimensions.
        self._coords = [copy(coord) for coord in cube.dim_coords]
        self._coords.remove(self._time)
        if not self._coords:
            raise ValueError('one or more non-time dimensions are required')
        # Store the auxiliary coordinates from the cube, categorising them into
        # coordinates spanning time only, coordinates spanning space only, and
        # coordinates spanning both time and space. This is helpful due to the
        # natural separation of space and time in EOF analysis. The time and
        # space spanning coordinates are only useful for reconstruction, as all
        # other methods return either a temporal field or a spatial field.
        (self._time_aux_coords,
         self._space_aux_coords,
         self._time_space_aux_coords) = classified_aux_coords(cube)
        # Define the weights array for the cube.
        if weights is None:
            wtarray = None
        else:
            try:
                scheme = weights.lower()
                wtarray = weights_array(cube, scheme=scheme)
            except AttributeError:
                wtarray = weights
        try:
            # Ensure weights are the same type as the cube data.
            wtarray = wtarray.astype(cube.data.dtype)
        except AttributeError:
            pass
        # Initialize a solver.
        self._solver = standard.Eof(cube.data,
                                    weights=wtarray,
                                    center=center,
                                    ddof=ddof)
        #: Number of EOFs in the solution.
        self.neofs = self._solver.neofs
        # Get the name of the cube to refer to later.
        self._cube_name = cube.name(default='dataset').replace(' ', '_')
        self._cube_var_name = cube.var_name

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
            A `~iris.cube.Cube` containing the ordered PCs. The PCs are
            numbered from 0 to *npcs* - 1.

        **Examples:**

        All un-scaled PCs::

            pcs = solver.pcs()

        First 3 PCs scaled to unit variance::

            pcs = solver.pcs(npcs=3, pcscaling=1)

        """
        pcs = self._solver.pcs(pcscaling, npcs)
        pcdim = DimCoord(list(range(pcs.shape[1])),
                         var_name='pc',
                         long_name='pc_number')
        coords = [copy(self._time), pcdim]
        pcs = Cube(
            pcs,
            dim_coords_and_dims=list(zip(coords, list(range(pcs.ndim)))),
            var_name='pcs',
            long_name='principal_components')
        # Add any auxiliary coords spanning time back to the returned cube.
        for coord, dims in self._time_aux_coords:
            pcs.add_aux_coord(copy(coord), dims)
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
           A `~iris.cube.Cube` containing the ordered EOFs. The EOFs are
           numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs with no scaling::

            eofs = solver.eofs()

        First 3 EOFs with scaling applied::

            eofs = solver.eofs(neofs=3, eofscaling=1)

        """
        eofs = self._solver.eofs(eofscaling, neofs)
        eofdim = DimCoord(list(range(eofs.shape[0])),
                          var_name='eof',
                          long_name='eof_number')
        coords = [eofdim] + [copy(coord) for coord in self._coords]
        eofs = Cube(
            eofs,
            dim_coords_and_dims=list(zip(coords, list(range(eofs.ndim)))),
            var_name='eofs',
            long_name='empirical_orthogonal_functions')
        # Add any auxiliary coordinates spanning space to the returned cube.
        for coord, dims in self._space_aux_coords:
            eofs.add_aux_coord(copy(coord), dims)
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
           A `~iris.cube.Cube` containing the ordered EOFs. The EOFs are
           numbered from 0 to *neofs* - 1.

        **Examples:**

        All EOFs::

            eofs = solver.eofsAsCorrelation()

        The leading EOF::

            eof1 = solver.eofsAsCorrelation(neofs=1)

        """
        eofs = self._solver.eofsAsCorrelation(neofs)
        eofdim = DimCoord(list(range(eofs.shape[0])),
                          var_name='eof',
                          long_name='eof_number')
        coords = [eofdim] + [copy(coord) for coord in self._coords]
        eofs = Cube(
            eofs,
            dim_coords_and_dims=list(zip(coords, list(range(eofs.ndim)))),
            var_name='eofs',
            long_name='correlation_between_pcs_and_{:s}'.format(
                self._cube_name))
        # Add any auxiliary coordinates spanning space to the returned cube.
        for coord, dims in self._space_aux_coords:
            eofs.add_aux_coord(copy(coord), dims)
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
           A `~iris.cube.Cube` containing the ordered EOFs. The EOFs are
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
        eofdim = DimCoord(list(range(eofs.shape[0])),
                          var_name='eof',
                          long_name='eof_number')
        coords = [eofdim] + [copy(coord) for coord in self._coords]
        eofs = Cube(
            eofs,
            dim_coords_and_dims=list(zip(coords, list(range(eofs.ndim)))),
            var_name='eofs',
            long_name='covariance_between_pcs_and_{:s}'.format(
                self._cube_name))
        # Add any auxiliary coordinates spanning space to the returned cube.
        for coord, dims in self._space_aux_coords:
            eofs.add_aux_coord(copy(coord), dims)
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
            A `~iris.cube.Cube` containing the eigenvalues arranged
            largest to smallest. The eigenvalues are numbered from 0 to
            *neigs* - 1.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalue1 = solver.eigenvalues(neigs=1)

        """
        lambdas = self._solver.eigenvalues(neigs=neigs)
        eofdim = DimCoord(list(range(lambdas.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        lambdas = Cube(
            lambdas,
            dim_coords_and_dims=list(zip(coords, list(range(lambdas.ndim)))),
            var_name='eigenvalues',
            long_name='eigenvalues')
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
            A `~iris.cube.Cube` containing the fractional variances for
            each eigenvalue. The eigenvalues are numbered from 0 to
            *neigs* - 1.


        **Examples:**

        The fractional variance represented by each eigenvalue::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first 3 eigenvalues::

            variance_fractions = solver.VarianceFraction(neigs=3)

        """
        vfrac = self._solver.varianceFraction(neigs=neigs)
        eofdim = DimCoord(list(range(vfrac.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        vfrac = Cube(
            vfrac,
            dim_coords_and_dims=list(zip(coords, list(range(vfrac.ndim)))),
            var_name='variance_fractions',
            long_name='variance_fractions')
        return vfrac

    def totalAnomalyVariance(self):
        """
        Total variance associated with the field of anomalies (the sum
        of the eigenvalues).

        **Returns:**

        *total_variance*
            A scalar value (not a `~iris.cube.Cube`).

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
            A `~iris.cube.Cube` containing the typical errors for each
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
        eofdim = DimCoord(list(range(typerrs.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        typerrs = Cube(
            typerrs,
            dim_coords_and_dims=list(zip(coords, list(range(typerrs.ndim)))),
            var_name='typical_errors',
            long_name='typical_errors')
        return typerrs

    def reconstructedField(self, neofs):
        """Reconstructed data field based on a subset of EOFs.

        If weights were passed to the `Eof` instance the returned
        reconstructed field will automatically have this weighting
        removed. Otherwise the returned field will have the same
        weighting as the `Eof` input *dataset*.

        Returns the reconstructed field in a `~iris.cube.Cube`.

        **Argument:**

        *neofs*
            Number of EOFs to use for the reconstruction.
            Alternatively this argument can be an iterable of mode
            numbers (where the first mode is 1) in order to facilitate
            reconstruction with arbitrary modes.

        **Returns:**

        *reconstruction*
            A `~iris.cube.Cube` with the same dimensions `Eof` input
            *dataset* containing the reconstruction using *neofs* EOFs.

        **Example:**

        Reconstruct the input field using 3 EOFs::

            reconstruction = solver.reconstructedField(3)

        Reconstruct the input field using EOFs 1, 2 and 5::

            reconstruction = solver.reconstuctedField([1, 2, 5])

        """
        rfield = self._solver.reconstructedField(neofs)
        coords = [copy(self._time)] + [copy(coord) for coord in self._coords]
        if isinstance(neofs, collections.Iterable):
            name_part = 'EOFs_{}'.format('_'.join([str(e) for e in neofs]))
        else:
            name_part = '{}_EOFs'.format(neofs)
        rfield = Cube(
            rfield,
            dim_coords_and_dims=list(zip(coords, list(range(rfield.ndim)))),
            var_name=self._cube_var_name or 'dataset',
            long_name='{:s}_reconstructed_with_{:s}'.format(
                self._cube_name, name_part))
        rfield.attributes.update({'neofs': neofs})
        # Add any auxiliary coordinates to the returned cube.
        for coord, dims in (self._time_aux_coords +
                            self._space_aux_coords +
                            self._time_space_aux_coords):
            rfield.add_aux_coord(copy(coord), dims)
        return rfield

    def projectField(self, cube, neofs=None, eofscaling=0, weighted=True):
        """Project a field onto the EOFs.

        Given a data set, projects it onto the EOFs to generate a
        corresponding set of pseudo-PCs.

        **Argument:**

        *field*
            An `iris.cube.Cube` containing the field to project onto the
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
            A `~iris.cube.Cube` containing the pseudo-PCs. The PCs are
            numbered from 0 to *neofs* - 1.

        **Examples:**

        Project a field onto all EOFs::

            pseudo_pcs = solver.projectField(field)

        Project fields onto the three leading EOFs::

            pseudo_pcs = solver.projectField(field, neofs=3)

        """
        # Check that the input is an Iris cube.
        if not isinstance(cube, Cube):
            raise TypeError('the input must be an iris cube')
        cube_name = cube.name(default='dataset').replace(' ', '_')
        has_time = False
        try:
            # A time dimension must be first.
            time, time_dim = get_time_coord(cube)
            has_time = True
        except ValueError:
            # No time dimension is also acceptable.
            has_time = False
        if has_time:
            if time_dim != 0:
                raise ValueError('time must be the first dimension, '
                                 'consider using the transpose() method')
            _time_aux_coords, _, _ = classified_aux_coords(cube)
        pcs = self._solver.projectField(cube.data,
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        # Create the PCs cube.
        pcs = Cube(pcs,
                   long_name='{}_pseudo_pcs'.format(cube_name),
                   var_name='pseudo_pcs')
        # Construct the required dimensions.
        if pcs.ndim == 2:
            # 2D PCs require a time axis and a PC axis.
            pcdim = DimCoord(list(range(pcs.shape[1])),
                             var_name='pc',
                             long_name='pc_number')
            pcs.add_dim_coord(copy(time), 0)
            pcs.add_dim_coord(pcdim, 1)
            # Add any time-spanning auxiliary coordinates from the input cube
            # to the returned PCs.
            for coord, dims in _time_aux_coords:
                pcs.add_aux_coord(copy(coord), dims)
        else:
            # 1D PCs require only a PC axis.
            pcdim = DimCoord(list(range(pcs.shape[0])),
                             var_name='pc',
                             long_name='pc_number')
            pcs.add_dim_coord(pcdim, 0)
        return pcs

    def getWeights(self):
        """Weights used for the analysis.

        **Returns:**

        *weights*
            An array contaning the analysis weights (not a
            `~iris.cube.Cube`).

        **Example:**

        The weights used for the analysis::

            weights = solver.getWeights()

        """
        return self._solver.getWeights()
