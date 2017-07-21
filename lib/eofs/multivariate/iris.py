"""Multivariate EOF analysis for `iris` cubes."""
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

from eofs.tools.iris import (get_time_coord, weights_array,
                             classified_aux_coords, common_items)
from . import standard


class MultivariateEof(object):
    """Multivariate EOF analysis (meta-data enabled `iris` interface)"""

    def __init__(self, cubes, weights=None, center=True, ddof=1):
        """Create a MultivariateEof instance.

        The EOF solution is computed at initialization time. Method
        calls are used to retrieve computed quantities.

        **Arguments:**

        *cubes*
            A list/tuple containing one or more `~iris.cube.Cube`
            instances, each with two or more dimensions, containing the
            data to be analysed. Time must be the first dimension of
            each `~iris.cube.Cube`. Missing values are allowed provided
            that they are constant with time in each field (e.g., values
            of an oceanographic field over land).

        **Optional arguments:**

        *weights*
            Sets the weighting method. One method can be chosen to apply
            to all cubes in *datasets* or a sequence of options can be
            given to specify a different weighting method for each cube
            in *datasets*. The following pre-defined weighting methods
            are available:

            * *'area'* : Square-root of grid cell area normalized by
              total grid area. Requires a latitude-longitude grid to be
              present in the corresponding `~iris.cube.Cube`. This is a
              fairly standard weighting strategy. If you are unsure
              which method to use and you have gridded data then this
              should be your first choice.

            * *'coslat'* : Square-root of cosine of latitude. Requires a
              latitude dimension to be present in the corresponding
              `~iris.cube.Cube`.

            * *None* : Equal weights for all grid points (*'none'* is
              also accepted).

             Alternatively a sequence of arrays of weights whose shapes
             are compatible with the corresponding `~iris.cube.Cube`
             instances in *datasets* may be supplied instead of
             specifying a weighting method.

        *center*
            If *True*, the mean along the first axis of each cube in
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

        EOF analysis of two cubes with area-weighting::

            from eofs.multivariate.iris import MultivariateEof
            solver = MultivariateEof(cube1, cube2, weights='area')

        """
        # Record the number of input cubes.
        self._ncubes = len(cubes)
        # Check that the weights argument is valid and refactor it if there
        # is only one option provided.
        if weights in (None, 'area', 'coslat'):
            weights = [weights] * self._ncubes
        elif len(weights) != self._ncubes:
            raise ValueError('number of weights and cubes must match')
        # Process each input cube recording its time dimension coordinate,
        # other dimension coordinates, and defining its weight array.
        self._time = []
        self._coords = []
        self._time_aux_coords = []
        self._space_aux_coords = []
        self._time_space_aux_coords = []
        passweights = []
        for cube, weight in zip(cubes, weights):
            if not isinstance(cube, Cube):
                raise TypeError('input is not an iris cube')
            # Record the time dimension and it's position. If its position is
            # not 0 then raise an error.
            time, time_dim = get_time_coord(cube)
            if time_dim != 0:
                raise ValueError('time must be the first dimension, '
                                 'consider using the transpose() method')
            self._time.append(copy(time))
            # Make a list of the cube's other dimension coordinates.
            coords = [copy(coord) for coord in cube.dim_coords]
            coords.remove(time)
            if not coords:
                raise ValueError('one or more non-time '
                                 'dimensions are required')
            self._coords.append(coords)
            # Make a lists of the AuxCoords on the current cube and store
            # them for reapplication later.
            _t, _s, _ts = classified_aux_coords(cube)
            self._time_aux_coords.append(_t)
            self._space_aux_coords.append(_s)
            self._time_space_aux_coords.append(_ts)
            # Determine the weighting option for the cube.
            if weight is None:
                wtarray = None
            else:
                try:
                    scheme = weight.lower()
                    wtarray = weights_array(cube, scheme=scheme)
                except AttributeError:
                    wtarray = weight
            try:
                wtarray = wtarray.astype(cube.data.dtype)
            except AttributeError:
                pass
            passweights.append(wtarray)
        # Get a list of all the auxiliary coordinates that span just time
        # and are present on every input cube.
        self._common_time_aux_coords = common_items(self._time_aux_coords)
        # Create a solver.
        self._solver = standard.MultivariateEof(
            [cube.data for cube in cubes],
            weights=passweights,
            center=center,
            ddof=ddof)
        #: Number of EOFs in the solution.
        self.neofs = self._solver.neofs
        # Names of the cubes.
        self._cube_names = [c.name(default='dataset').replace(' ', '_')
                            for c in cubes]
        self._cube_var_names = [cube.var_name for cube in cubes]

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
            A `~iris.cube.Cube` containing the ordered PCs.

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
        coords = [copy(self._time[0]), pcdim]
        pcs = Cube(
            pcs,
            dim_coords_and_dims=list(zip(coords, list(range(pcs.ndim)))),
            var_name='pcs',
            long_name='principal_components')
        # Add any AuxCoords that described the time dimension of all the input
        # cubes.
        for coord, dims in self._common_time_aux_coords:
            pcs.add_aux_coord(copy(coord), dims)
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
            A list of `~iris.cube.Cube` instances containing the ordered
            EOFs for each variable.

        **Examples:**

        All EOFs with no scaling::

            eofs_list = solver.eofs()

        The leading EOF with scaling applied::

            eof1_list = solver.eofs(neofs=1, eofscaling=1)

        """
        eofset = self._solver.eofs(eofscaling=eofscaling, neofs=neofs)
        neofs = eofset[0].shape[0]
        eofdim = DimCoord(list(range(neofs)),
                          var_name='eof',
                          long_name='eof_number')
        for iset in range(self._ncubes):
            coords = [eofdim] + [copy(coord) for coord in self._coords[iset]]
            eofset[iset] = Cube(
                eofset[iset],
                dim_coords_and_dims=list(zip(coords,
                                             range(eofset[iset].ndim))),
                var_name='eofs',
                long_name='empirical_orthogonal_functions')
            for coord, dims in self._space_aux_coords[iset]:
                eofset[iset].add_aux_coord(copy(coord), dims)
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
            A list of `~iris.cube.Cube` instances containing the ordered
            EOFs for each variable.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCorrelation()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCorrelation(neofs=1)

        """
        eofset = self._solver.eofsAsCorrelation(neofs=neofs)
        neofs = eofset[0].shape[0]
        eofdim = DimCoord(list(range(neofs)),
                          var_name='eof',
                          long_name='eof_number')
        for iset in range(self._ncubes):
            coords = [eofdim] + [copy(coord) for coord in self._coords[iset]]
            eofset[iset] = Cube(
                eofset[iset],
                dim_coords_and_dims=list(zip(coords,
                                             range(eofset[iset].ndim))),
                var_name='eofs',
                long_name='correlation_between_pcs_and_{:s}'.format(
                          self._cube_names[iset]))
            for coord, dims in self._space_aux_coords[iset]:
                eofset[iset].add_aux_coord(copy(coord), dims)
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
            A list of `~iris.cube.Cube` instances containing the ordered
            EOFs for each variable.

        **Examples:**

        All EOFs of each data set::

            eofs_list = solver.eofsAsCovariance()

        The leading EOF of each data set::

            eof1_list = solver.eofsAsCovariance(neofs=1)

        """
        eofset = self._solver.eofsAsCovariance(neofs=neofs)
        neofs = eofset[0].shape[0]
        eofdim = DimCoord(list(range(neofs)),
                          var_name='eof',
                          long_name='eof_number')
        for iset in range(self._ncubes):
            coords = [eofdim] + [copy(coord) for coord in self._coords[iset]]
            eofset[iset] = Cube(
                eofset[iset],
                dim_coords_and_dims=list(zip(coords,
                                             range(eofset[iset].ndim))),
                var_name='eofs',
                long_name='covariance_between_pcs_and_{:s}'.format(
                          self._cube_names[iset]))
            for coord, dims in self._space_aux_coords[iset]:
                eofset[iset].add_aux_coord(copy(coord), dims)
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
            A `~iris.cube.Cube` containing the eigenvalues arranged
            largest to smallest.

        **Examples:**

        All eigenvalues::

            eigenvalues = solver.eigenvalues()

        The first eigenvalue::

            eigenvalues1 = solver.eigenvalues(neigs=1)

        """
        lambdas = self._solver.eigenvalues(neigs=neigs)
        eofdim = DimCoord(list(range(lambdas.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        lambdas = Cube(
            lambdas,
            dim_coords_and_dims=list(zip(coords, range(lambdas.ndim))),
            var_name='eigenvalues',
            long_name='eigenvalues')
        return lambdas

    def varianceFraction(self, neigs=None):
        """Fractional EOF mode variances.

        **Optional argument:**

        *neigs*
            Number of eigenvalues to return the fractional variance for.
            Defaults to all eigenvalues. If the number of eigenvalues
            requested is more than the number that are available, then
            fractional variances for all available eigenvalues will be
            returned.

        **Returns:**

        *variance_fractions*
            A `~iris.cube.Cube` containing the fractional variances.

        **Examples:**

        The fractional variance represented by each EOF mode::

            variance_fractions = solver.varianceFraction()

        The fractional variance represented by the first EOF mode::

            variance_fraction_mode_1 = solver.VarianceFraction(neigs=1)

        """
        vfrac = self._solver.varianceFraction(neigs=neigs)
        eofdim = DimCoord(list(range(vfrac.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        vfrac = Cube(
            vfrac,
            dim_coords_and_dims=list(zip(coords, range(vfrac.ndim))),
            var_name='variance_fraction',
            long_name='variance_fraction')
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
            Defaults to typical errors for all eigenvalues. If the
            number of eigenvalues requested is more than the number that
            are available, then typical errors for all available
            eigenvalues will be returned.

        *vfscaled*
            If *True* scale the errors by the sum of the eigenvalues.
            This yields typical errors with the same scale as the values
            returned by `MultivariateEof.varianceFraction`. If *False*
            then no scaling is done. Defaults to *False* (no scaling).

        **References**

        North G.R., T.L. Bell, R.F. Cahalan, and F.J. Moeng (1982)
        Sampling errors in the estimation of empirical orthogonal
        functions. *Mon. Weather. Rev.*, **110**, pp 669-706.

        **Returns:**

        *errors*
            A `~iris.cube.Cube` containing the typical errors.

        **Examples:**

        Typical errors for all eigenvalues::

            errs = solver.northTest()

        Typical errors for the first 5 eigenvalues scaled by the sum of
        the eigenvalues::

            errs = solver.northTest(neigs=5, vfscaled=True)

        """
        typerrs = self._solver.northTest(neigs=neigs, vfscaled=vfscaled)
        eofdim = DimCoord(list(range(typerrs.shape[0])),
                          var_name='eigenvalue',
                          long_name='eigenvalue_number')
        coords = [eofdim]
        typerrs = Cube(
            typerrs,
            dim_coords_and_dims=list(zip(coords, range(typerrs.ndim))),
            var_name='typical_errors',
            long_name='typical_errors')
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
            A list of `~iris.cube.Cube` with the same dimensions as the
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
        for iset in range(self._ncubes):
            coords = [copy(self._time[iset])] + \
                     [copy(coord) for coord in self._coords[iset]]
            rfset[iset] = Cube(
                rfset[iset],
                dim_coords_and_dims=list(zip(coords, range(rfset[iset].ndim))),
                var_name=self._cube_var_names[iset] or 'dataset_{:d}'.format(
                    iset),
                long_name='{:s}_reconstructed_with_{:s}'.format(
                    self._cube_names[iset], name_part))
            rfset[iset].attributes.update({'neofs': neofs})
            for coord, dims in (self._time_aux_coords[iset] +
                                self._space_aux_coords[iset] +
                                self._time_space_aux_coords[iset]):
                rfset[iset].add_aux_coord(copy(coord), dims)
        return rfset

    def projectField(self, cubes, neofs=None, eofscaling=0, weighted=True):
        """Project a set of fields onto the EOFs.

        Given a set of fields, projects them onto the EOFs to generate a
        corresponding set of pseudo-PCs.
        **Argument:**

        *fields*
            A list/tuple containing one or more `~iris.cube.Cube`
            instances, each with two or more dimensions, containing the
            data to be projected onto the EOFs. Each field must have the
            same spatial dimensions (including missing values in the
            same places) as the corresponding data set in the
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
            A `~iris.cube.Cube` containing the ordered pseudo-PCs.

        **Examples:**

        Project a data set onto all EOFs::

            pseudo_pcs = solver.projectField([field1, field2])

        Project a data set onto the four leading EOFs::

            pseudo_pcs = solver.projectField([field1, field2], neofs=4)

        """
        for cube in cubes:
            if not isinstance(cube, Cube):
                raise TypeError('input is not an iris cube')
        if len(cubes) != self._ncubes:
            raise ValueError('number of cubes is incorrect, expecting {:d} '
                             'but got {:d}'.format(self._ncubes, len(cubes)))
        _all_time_aux_coords = []
        for cube in cubes:
            try:
                # Time dimension must be first.
                raise_error = False
                time, time_coord = get_time_coord(cube)
                if time_coord != 0:
                    raise_error = True
            except ValueError:
                # Not having a time dimension is also acceptable.
                pass
            if raise_error:
                raise ValueError('time must be the first dimension, '
                                 'consider using the transpose() method')
            # Store any AuxCoords describing the time dimension.
            _t, _, _ = classified_aux_coords(cube)
            _all_time_aux_coords.append(_t)
        # Retain AuxCoords that describe the time dimension of *every* input
        # cube.
        _common_time_aux_coords = common_items(_all_time_aux_coords)
        # Compute the PCs.
        pcs = self._solver.projectField([cube.data for cube in cubes],
                                        neofs=neofs,
                                        eofscaling=eofscaling,
                                        weighted=weighted)
        pcs = Cube(pcs, var_name='pseudo_pcs', long_name='pseudo_pcs')
        # Construct the required dimensions.
        if pcs.ndim == 2:
            # 2D PCs require a time axis and a PC axis.
            pcdim = DimCoord(list(range(pcs.shape[1])),
                             var_name='pc',
                             long_name='pc_number')
            time, time_dim = get_time_coord(cubes[0])
            pcs.add_dim_coord(copy(time), 0)
            pcs.add_dim_coord(pcdim, 1)
            # Add any auxiliary coordinates for the time dimension.
            for coord, dims in _common_time_aux_coords:
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

        *weights_list*
            A list of arrays containing the analysis weights for each
            variable (not `~iris.cube.Cube` instances).

        **Example:**

        The weights used for the analysis::

            weights_list = solver.getWeights()

        """
        return self._solver.getWeights()
