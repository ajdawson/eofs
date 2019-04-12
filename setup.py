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
import os.path

from setuptools import setup
import versioneer


packages = ['eofs',
            'eofs.tools',
            'eofs.multivariate',
            'eofs.examples',
            'eofs.tests']

package_data = {'eofs.examples': ['example_data/*'],
                'eofs.tests': ['data/*']}

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(name='eofs',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          description='EOF analysis in Python',
          author='Andrew Dawson',
          author_email='dawson@atm.ox.ac.uk',
          url='https://ajdawson.github.com/eofs',
          long_description=long_description,
          long_description_content_type='text/markdown',
          packages=packages,
          package_dir={'': 'lib'},
          package_data=package_data,
          install_requires=['numpy'])
