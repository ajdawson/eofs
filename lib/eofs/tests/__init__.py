"""Test for the `eofs` package."""
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


np.seterr(all='ignore')


class EofsTest(object):
    """Base class for tests."""

    def _tomasked(self, value):
        """
        Defines the conversion of the data type the test uses to a
        :class:`numpy.ma.MaskedArray`.

        This method should be overridden by subclasses. It must take a
        single object as input and return a :class:`numpy.ma.MaskedArray`
        representation of that object. It must also be able to be called
        multiple times on the same object.

        The default implementation simply returns the input value.

        """
        return value

    def assert_array_almost_equal(self, a, b):
        """Assertion that two arrays compare almost equal.

        The arrays are converted to :class:`numpy.ma.MaskedArray` using
        the :meth:`_tomasked` method of the test class before the
        comparison is made.

        """
        assert ma.allclose(self._tomasked(a), self._tomasked(b))

    def assert_almost_equal(self, a, b):
        """Assertion that two values compare almost equal."""
        assert ma.allclose(self._tomasked(a), self._tomasked(b))

    def assert_true(self, cond):
        """Assertion that a condition is True."""
        assert cond

    def assert_equal(self, a, b, message=None):
        if message is not None:
            assert a == b, message
        else:
            assert a == b
