"""Test coding standards compliance."""
# (c) Copyright 2016 Andrew Dawson. All Rights Reserved.
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

import os

import pycodestyle

import eofs
from eofs.tests import EofsTest


class TestCodingStandards(EofsTest):

    def test_pep8(self):
        pep8style = pycodestyle.StyleGuide(quiet=False)
        base_paths = [os.path.dirname(eofs.__file__)]
        result = pep8style.check_files(base_paths)
        self.assert_equal(result.total_errors, 0, "Found PEP8 style issues.")
