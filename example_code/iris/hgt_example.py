"""
Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector during winter time.

This example uses the metadata-retaining iris interface.

"""
import warnings

import iris
import iris.plot as iplt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from eofs.iris import Eof
from eofs.examples import example_data_path


# Read geopotential height data using the iris module. The file contains
# December-February averages of geopotential height at 500 hPa for the
# European/Atlantic domain (80W-40E, 20-90N).
filename = example_data_path('hgt_djf.nc')
z_djf = iris.load_cube(filename)

# Compute anomalies by removing the time-mean.
with warnings.catch_warnings():
    # Iris emits a warning due to the non-contiguous time dimension.
    warnings.simplefilter('ignore', UserWarning)
    z_djf_mean = z_djf.collapsed('time', iris.analysis.MEAN)
z_djf.data = z_djf.data - z_djf_mean.data

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
solver = Eof(z_djf, weights='coslat')

# Retrieve the leading EOF, expressed as the covariance between the leading PC
# time series and the input SLP anomalies at each grid point.
eof1 = solver.eofsAsCovariance(neofs=1)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_global()
iplt.contourf(eof1[0, 0], cmap=plt.cm.RdBu_r)
ax.set_title('EOF1 expressed as covariance', fontsize=16)
plt.show()
