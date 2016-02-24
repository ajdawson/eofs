"""Build and install the eofs package."""
# (c) Copyright 2013-2014 Andrew Dawson. All Rights Reserved.
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
from setuptools import setup


for line in open('lib/eofs/__init__.py').readlines():
    if line.startswith('__version__'):
        exec(line.strip())

packages = ['eofs',
            'eofs.tools',
            'eofs.multivariate',
            'eofs.examples',
            'eofs.tests']

package_data = {'eofs.examples': ['example_data/*'],
                'eofs.tests': ['data/*']}


if __name__ == '__main__':
    setup(name='eofs',
          version=__version__,
          description='EOF analysis in Python',
          author='Andrew Dawson',
          author_email='dawson@atm.ox.ac.uk',
          url='https://ajdawson.github.com/eofs',
          long_description="""
eofs is a package for performing EOF analysis on spatial-temporal
data sets, aimed at meteorology, oceanography and climate sciences
          """,
          packages=packages,
          package_dir={'': 'lib'},
          package_data=package_data,
          install_requires=['numpy'])
