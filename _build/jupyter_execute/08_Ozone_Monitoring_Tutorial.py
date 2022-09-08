#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# # Ozone Monitoring Tutorial
# 
# This tutorial provides guided examples of ozone monitoring using data from the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/). It is divided into three parts:
# 
# 1. View animation of Antarctic ozone hole
# 2. Calculate the size of the Antarctic ozone hole
# 3. View vertical profiles of global ozone
# 
# It uses CAMS global reanalysis (EAC4) data freely available from the [Atmosphere Data Store (ADS)](https://ads.atmosphere.copernicus.eu)

# <style>
# td, th {
#    border: 1px solid white;
#    border-collapse: collapse;
# }
# </style>
# <table align="left">
#   <tr>
#     <th>Run the tutorial via free cloud platforms: </th>
#     <th><a href="https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training-cams/master?labpath=08_Ozone_Monitoring_Tutorial.ipynb">
#         <img src = "https://mybinder.org/badge.svg" alt = "Binder"></th>
#     <th><a href="https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training-cams/blob/master/08_Ozone_Monitoring_Tutorial.ipynb">
#         <img src = "https://kaggle.com/static/images/open-in-kaggle.svg" alt = "Kaggle"></th>
#     <th><a href="https://colab.research.google.com/github/ecmwf-projects/copernicus-training-cams/blob/master/08_Ozone_Monitoring_Tutorial.ipynb">
#         <img src = "https://colab.research.google.com/assets/colab-badge.svg" alt = "Colab"></th>
#   </tr>
# </table>

# In[1]:


get_ipython().system('pip install cdsapi')


# In[1]:


# CDS API
import cdsapi

# Libraries for reading and working with multidimensional arrays
import numpy as np
import xarray as xr

# Libraries for plotting and animating data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import cartopy.crs as ccrs
from IPython.display import HTML

# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()

from collections import OrderedDict


# ## Download and read and pre-process ozone data
# 
# ### Download data
# 
# Copy your API key into the code cell below, replacing `#######` with your key. (Remember, to access data from the ADS, you will need first to register/login https://ads.atmosphere.copernicus.eu and obtain an API key from https://ads.atmosphere.copernicus.eu/api-how-to.)

# In[2]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'

# Replace the hashtags with your key:
KEY = '##########################################'


# Here we specify a data directory into which we will download our data and all output files that we will generate:

# In[3]:


DATADIR = './'


# For this tutorial, we will use CAMS Global Reanalysis (EAC4) data. The code below shows the subset characteristics that we will extract from this dataset as an API request.

# ```{note}
# Before running this code, ensure that you have **accepted the terms and conditions**. This is something you only need to do once for each CAMS dataset. You will find the option to do this by selecting the dataset in the ADS, then scrolling to the end of the *Download data* tab.
# ```

# In[4]:


c = cdsapi.Client(url=URL, key=KEY)
c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'variable': 'total_column_ozone',
        'date': '2020-07-01/2021-01-31',
        'time': '00:00',
        'format': 'netcdf',
        'area': [
            0, -180, -90,
            180,
        ],
    },
    f'{DATADIR}/TCO3_202007-202101_SHem.nc')


# ### Import and inspect data
# 
# Here we read the data we downloaded into an xarray Dataset, and view its contents:

# In[5]:


fn = f'{DATADIR}/TCO3_202007-202101_SHem.nc'
ds = xr.open_dataset(fn)
ds


# We can see that the data has three coordinate dimensions (longitude, latitude and time) and one variable, total column ozone. By inspecting the coordinates we can see the data is of the southern hemisphere from 1st July 2020 to 31 January 2021 at 00:00 UTC each day. This includes the entire period in which the Antarctic ozone hole appears.

# ```{note}
# This is only a subset of the available data from the [ADS](https://ads.atmosphere.copernicus.eu), which includes global data at 3 hourly resolution (in addition to monthly averages), from 2003 to the present, and at 60 model levels (vertical layers) in the atmosphere.
# ```

# To facilitate further processing, we convert the Xarray Dataset into an Xarray Data Array containing the single variable of total column ozone.

# In[6]:


tco3 = ds['gtco3']


# ### Unit conversion
# 
# We can see from the attributes of our data are represented as mass concentration of ozone, in units of kg m**-2. We would like to convert this into Dobson Units, which is a standard for ozone measurements. The Dobson unit is defined as the thickness (in units of 10 μm) of a layer of pure gas (in this case O3) which would be formed by the total column amount at standard conditions for temperature and pressure.

# In[7]:


convertToDU = 1 / 2.1415e-5


# In[8]:


tco3 = tco3 * convertToDU


# ## View southern hemisphere ozone hole 
# 
# Let us now visualise our data in form of maps and animations. 
# 
# We will first define the colour scale we would like to represent our data.

# ### Define colour scale

# Extract range of values

# In[9]:


tco3max = tco3.max()
tco3min = tco3.min()
tco3range = (tco3max - tco3min)/20.


# Define colourmap

# In[10]:


cmap = plt.cm.jet
norm = colors.BoundaryNorm(np.arange(tco3min, tco3max, tco3range), cmap.N)


# ### Plot map
# 
# We will now plot our data. Let us begin with a map for the first time step, 1 July 2020.

# In[11]:


fig = plt.figure(figsize=(5, 5)) 
ax = plt.subplot(1,1,1, projection=ccrs.Orthographic(central_latitude=-90)) 
ax.coastlines(color='black') # Add coastlines
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--') 
ax.set_title(f'Total Column Ozone, {str(tco3.time[0].values)[:-19]}', fontsize=12) 
im = plt.pcolormesh(tco3.longitude.values, tco3.latitude.values, tco3[0,:,:],
                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im,fraction=0.046, pad=0.04) 
cbar.set_label('Total Column Ozone (DU)') 


# ### Create animation
# 
# Let us now view the entire time series by means of an animation that shows all time steps. We see from the dataset description above, that the time dimension in our data includes 215 entries. These correspond to 215 time steps, i.e. 215 days, between 1 July 2020 and 31 January 2021. The number of frames of our animation is therefore 215.

# In[12]:


frames = 215


# Now we will create an animation. The last cell, which creates the animation in Javascript, may take a few minutes to run.

# In[13]:


def animate(i):
    array = tco3[i,:,:].values
    im.set_array(array.flatten())
    ax.set_title(f'Total Column Ozone, {str(tco3.time[i].values)[:-19]}', fontsize=12)


# In[15]:


ani = animation.FuncAnimation(fig, animate, frames, interval=100)
ani.save(f'{DATADIR}/TCO3_202007-202101_SHem.gif')
HTML(ani.to_jshtml())


# <div class="alert alert-block alert-success">
# <b>ACTIVITY</b>: <br>
#     In what date range does the ozone hole appear greatest? In the next section we will quantify the size of the ozone hole at each time step.
# </div>

# ## Calculate size of ozone hole
# 
# In this section we will calculate the actual size of the ozone hole, and plot its evolution over our time series. We will then apply the exact same method on monthly averaged data over the past decades to compare ozone hole extents at each yearly occurence since 2003.

# ### Define constants
# 
# In order to calculate the size of the ozone hole, we need to define some constants. These include the following:
# 1. Ozone hole limit, i.e. minimum ozone, in Dobson units, beneath which we consider there to be a hole in the ozone layer.

# In[16]:


OZONE_HOLE_LIMIT = 220.


# 2. Radius of the Earth, in order to calculate the size of each geographic grid cell of our data.

# In[17]:


Rearth = 6371009.


# ### Calculate area of each geographic grid cell of our data
# 
# Here we calculate the size of each grid cell of our data, which varies as a function of latitude. The formula we apply is the following:
# 
# area = |(sin(top latitude) - sin(bottom latitude))| * |Δlongitude| * Earth radius squared
# 
# First we define a function to apply this formula, taking as arguments the min and max of the latitudes and longitudes represented by each grid cell:

# In[18]:


def geo_area_array(lats1,lons1,lats2,lons2):
    area= np.abs(np.sin(np.deg2rad(lats1)) - np.sin(np.deg2rad(lats2))) * np.abs(np.deg2rad(lons1-lons2))
    area = area * Rearth * Rearth 
    return area


# Then we apply this function to our data, calculating first the lat/lon extents of each cell (0.75 is the spatial resolution, in degrees):

# In[19]:


tco3['areas'] = geo_area_array(tco3['latitude']-0.75/2, tco3['longitude']-0.75/2, 
                             tco3['latitude']+0.75/2, tco3['longitude']+0.75/2)


# ### Mask data belonging to Antarctic ozone hole
# 
# We now calculate the grid cells that meet the criteria of belonging to the Antarctic ozone hole: i.e. they have less than the minimum threshold of ozone, and they are below -60 degrees latitude.
# 
# Here we define a mask with the threshold conditions for ozone and latitude:

# In[20]:


mask = tco3.where((tco3 < OZONE_HOLE_LIMIT) & (tco3["latitude"] < -60.))


# All data points that do not meet the conditions are set to NaN (Not a Number) in the resulting array. Dividing this array by itself gives us a mask with each valid data point set to 1. Multiplying this by the corresponding areas leaves us with an array of grid cell areas that meet the conditions of belonging to the Antarctic ozone hole.

# In[21]:


area = (mask / mask) * mask['areas']


# ### Calculate ozone hole area

# If we now sum these grid cell areas, we have the total ozone hole extent. This is applied across each time step. We multiply by the number 1e-12 to convert from meters squared to million kilometers squared.

# In[23]:


ozone_hole = area.sum(dim=["latitude", "longitude"], skipna=True) * 1e-12


# Let us update the Data Array attributes:

# In[25]:


ozone_hole.attrs['long_name'] = 'Ozone hole area'
ozone_hole.attrs['units'] = 'million km^2'


# ### Plot ozone hole area at each time step
# 
# Finally we can plot the evolution of the ozone hole, in million km squared, throughout the time series.

# In[26]:


ozone_hole.plot()


# We can also calculate the maximum extent of the ozone hole:

# In[186]:


index = ozone_hole.argmax() # Here we find the index of the maximum value
print('Max extent of O3 hole:', ozone_hole[index].values, 'million km2')


# ... and find out when this was reached:

# In[188]:


print('This was reached on:', ozone_hole[index].time.values)


# <div class="alert alert-block alert-success">
# <b>ACTIVITY</b>: <br>
#     Now repeat the steps above with the data below (see API request in cell below). See what you find out about the evolution of the ozone hole since 2003!
# </div>

# In[ ]:


c.retrieve(
    'cams-global-reanalysis-eac4-monthly',
    {
        'format': 'netcdf',
        'variable': 'total_column_ozone',
        'year': [
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'product_type': 'monthly_mean',
    },
    f'{DATADIR}/TCO3_monthly_2003-2021.nc')


# ## Ozone vertical profile
# 
# So far we have concentrated on total column ozone. In this final part of the tutorial we will look at vertical profiles of ozone to see how ozone concentration differs across various altitudes.
# 
# This requires a new data request, including multi level data at 25 pressure levels represented in hectopascals (hPa).

# ```{note}
# The ADS provides freely available data at 60 atmospheric levels (model levels), but most of this is on slower access tapes. For the sake of simplicity, we will use in this tutorial only data from the 25 pressure levels.
# ```

# In[45]:


c.retrieve(
    'cams-global-reanalysis-eac4-monthly',
    {
        'variable': 'ozone',
        'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '150',
            '200', '250', '300',
            '400', '500', '600',
            '700', '800', '850',
            '900', '925', '950',
            '1000',
        ],
        'year': '2021',
        'month': '07',
        'product_type': 'monthly_mean',
        'format': 'netcdf',
    },
    f'{DATADIR}/O3_2021-07.nc')


# In[131]:


fn = f'{DATADIR}/O3_2021-07.nc'
ds = xr.open_dataset(fn)
ds


# ### Pressure levels in logarithmic scale
# 
# To facilitate visualisation of this data, we will create a new coordinate of the array with the pressure levels converted into logarithms.

# In[132]:


ds['level_log10'] = np.log10(ds['level'])


# In[133]:


ds = ds.set_coords('level_log10')


# ### Reorganise data and convert units
# 
# Here we create an Xarray Data Array object from the Dataset and remove the dimension of time, which has only one entry.

# In[134]:


da = ds['go3']


# In[135]:


da = da.squeeze('time')


# We will now convert the units of our data from mass mixing ratio (kg/kg) to Volume Mixing Ratio (VMR), in units of parts per billion (ppbv). We will also update the attributes of our array to reflect this.

# In[136]:


da = 28.9644 / 47.9982 * 1e9 * da


# In[137]:


da.attrs['long_name'] = 'Ozone Volume Mixing Ratio'
da.attrs['units'] = 'ppbv'


# ### Convert longitudes to +/- 180 degrees
# 
# Below, we would like to plot a 2 dimensional map of our data. Note however that the longitudinal coordinates are given in the range 0 to 360 degrees. To facilitate data visualisation we will convert the longitudes into a +/- 180 degree grid.

# In[138]:


da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby('longitude')


# ### Visualise geographical distribution of ozone at each layer
# 
# We will now visualise the global distribution of ozone at particular layers of the atmosphere. The description of our data above shows us that we have 25 layers of the atmosphere represented in our data, from 1000 hPa (lowest layer of atmosphere) to 1 (uppermost layer). These can be indexed in the first dimension of our array, where the 24th index corresponds to the lowest layer of the atmosphere, and the 0th represents the highest. 

# In[154]:


da[24,:,:].plot()


# <div class="alert alert-block alert-success">
# <b>ACTIVITY</b>: <br>
#     View the ozone at different levels of the atmosphere. Note the difference in the scale on the right!
# </div>

# ### Spatial aggregation
# 
# We will now visualise the global average of ozone at each layer in a profile plot. To do this, we must first average over the longitudinal and latitudinal dimensions to obtain a single figure at each layer. When averaging over the latitudinal dimension we need to account for the variation in area represented by each latitudinal grid cell. We do this by applying weights corresponding to the cosine of the latitudes. 

# In[140]:


weights = np.cos(np.deg2rad(da.latitude))
weights.name = "weights"
weighted = da.weighted(weights)


# In[141]:


o3 = weighted.mean(dim=["latitude", "longitude"])


# ### Plot ozone profile
# 
# We can now create our profile plot of the vertical distribution of ozone.

# In[142]:


fig, ax = plt.subplots(1, 1, figsize = (6, 9))

ax.set_title('Ozone global average July 2021', fontsize=12)
ax.set_ylabel('Log pressure (10^x hPa)')
ax.set_xlabel('O3 (ppbv)')
ax.invert_yaxis()
ax.plot(o3, o3.level_log10)

fig.savefig(f'{DATADIR}/O3_2021-07_profile.png')


# <div class="alert alert-block alert-success">
# <b>ACTIVITY</b>: <br>
#     At what level of the atmosphere is the concentration of ozone the greatest? Hint: To find the index of the maximum value of the array, use the same <code>.argmax()</code> function you used when calculating the size of the ozone hole above!
# </div>
