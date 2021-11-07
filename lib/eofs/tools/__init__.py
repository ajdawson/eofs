"""Tools related to EOF analysis."""
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
from __future__ import (absolute_import, division, print_function)  # noqa

from . import standard


# Define the objects imported by imports of the form: from eofs.tools import *.
__all__ = ['standard']


try:
    from . import cdms
    __all__.append('cdms')
except ImportError:
    pass

try:
    from . import iris
    __all__.append('iris')
except ImportError:
    pass
