"""Run the test suite for `eofs`."""
# (c) Copyright 2013 Andrew Dawson. All Rights Reserved.
#
# This file is part of eofs.
#
# eofs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# eofs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with eofs.  If not, see <http://www.gnu.org/licenses/>.
import nose


#: Test modules to run by default.
default_test_modules = [
    'eofs.tests.test_solution',
    'eofs.tests.test_error_handling',
    'eofs.tests.test_tools',
    'eofs.tests.test_multivariate_solution',
    'eofs.tests.test_multivariate_error_handling',
    'eofs.tests.test_rotation',
]


def run():
    """Run the :mod:`eofs` tests."""
    nose.main(defaultTest=default_test_modules)


if __name__ == '__main__':
    run()
