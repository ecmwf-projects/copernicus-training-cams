#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# <br>

# # Dust Monitoring

# ### About

# This notebook provides you a practical introduction to the [CAMS European air quality forecasts](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-europe-air-quality-forecasts?tab=overview) and [CAMS global atmospheric composition forecasts](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=overview) data sets. We will use both datasets to analyse a Saharan dust event that impacted Europe (primarily parts of Spain and France) between 21 to 23 February 2021. 

# The notebook has three main sections with the following outline:
# 
# * 1 - Define a discrete, not continuous color scale used in CAMS
# * 2 - CAMS European air quality forecasts
#   * 2.1 - Request data from the ADS programmatically with the CDS API
#   * 2.2 - Load and browse CAMS European air quality forecasts
#   * 2.3 - Select the variable *dust* and plot a map of dust over Europe in February 2021
#   * 2.4 - Select the variable *pm10_conc* and plot a map of PM10 over Europe in February 2021
#   * 2.5 - Animate dust concentration over Europe from 20-25 February 2021
# * 3 - CAMS global atmospheric composition forecasts
#   * 3.1 - Request data from the ADS programmatically with the CDS API
#   * 3.2 - Unzip the downloaded data file
#   * 3.3 - Load and browse CAMS global forecast of total Aerosol Optical Depth at 550nm
#   * 3.4 - Visualize a global map of total AOD at 550nm in February 2021
#   * 3.5 - Animate global total AOD at 550nm from 20 to 23 February 2021

# ### Data

# This notebook introduces you to the `CAMS European air quality forecasts` and the `CAMS global atmospheric composition forecasts`. The data has the following specifications:
# 
# > **Data**: `CAMS European air quality forecasts` <br>
# > **Variables**: `Dust`, `Particulate Matter 10` <br>
# > **Type**: `Analysis` <br>
# > **Temporal coverage**: `20-25 February 2021` <br>
# > **Temporal resolution**: `hourly` <br>
# > **Spatial coverage**: `Europe` <br>
# > **Format**: `NetCDF`
# 
# <br>
# 
# > **Data**: `CAMS global atmospheric composition forecasts` <br>
# > **Variables**: `Total Aerosol Optical Depth 550` <br>
# > **Type**: `Forecast` <br>
# > **Temporal coverage**: `20 February 2021` <br>
# > **Leadtime hour**: `0 to 90` <br>
# > **Spatial coverage**: `Global` <br>
# > **Format**: `NetCDF`
# 

# ### How to access the notebook
# 
# This tutorial is in the form of a [Jupyter notebook](https://jupyter.org/). You will not need to install any software for the training as there are a number of free cloud-based services to create, edit, run and export Jupyter notebooks such as this. Here are some suggestions (simply click on one of the links below to run the notebook):

# |Binder|Kaggle|Colab|NBViewer|
# |:-:|:-:|:-:|:-:|
# |[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training/HEAD?urlpath=lab/tree/CAMS_dust-monitoring.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training/blob/master/CAMS_dust-monitoring.ipynb)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf-projects/copernicus-training/blob/master/CAMS_dust-monitoring.ipynb)|[![NBViewer](https://raw.githubusercontent.com/ecmwf-projects/copernicus-training/master/img/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/ecmwf-projects/copernicus-training/blob/master/CAMS_dust-monitoring.ipynb)|
# |(Binder may take some time to load, so please be patient!)|(will need to login/register, and switch on the internet via *settings*)|(will need to run the command `!pip install cartopy` before importing the libraries)|(this will not run the notebook, only render it)|

# If you would like to run this notebook in your own environment, we suggest you install [Anaconda](https://docs.anaconda.com/anaconda/install/), which contains most of the libraries you will need. You will also need to install [Xarray](http://xarray.pydata.org/en/stable/) for working with multidimensional data in netcdf files, and the ADS API (`pip install cdsapi`) for downloading data programatically from the ADS.

# ### Further resources
# * [Saharan dust events in the spring of 2021 | ECMWF Newsletter Number 168](https://www.ecmwf.int/en/newsletter/168/news/saharan-dust-events-spring-2021)

# <hr>

# ### Install CDSAPI via pip

# In[ ]:


get_ipython().system('pip install cdsapi')


# ### Load libraries

# In[5]:


# CDS API
import cdsapi
import os

# Libraries for working with multi-dimensional arrays
import numpy as np
import xarray as xr
import pandas as pd

from IPython.display import HTML

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation

from datetime import datetime

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature


# ### Load helper functions

# In[6]:


def visualize_pcolormesh(data_array, longitude, latitude, projection, color_scale, unit, long_name, vmin, vmax, 
                         set_global=True, lonmin=-180, lonmax=180, latmin=-90, latmax=90):
    """ 
    Visualizes a xarray.DataArray with matplotlib's pcolormesh function.
    
    Parameters:
        data_array(xarray.DataArray): xarray.DataArray holding the data values
        longitude(xarray.DataArray): xarray.DataArray holding the longitude values
        latitude(xarray.DataArray): xarray.DataArray holding the latitude values
        projection(str): a projection provided by the cartopy library, e.g. ccrs.PlateCarree()
        color_scale(str): string taken from matplotlib's color ramp reference
        unit(str): the unit of the parameter, taken from the NetCDF file if possible
        long_name(str): long name of the parameter, taken from the NetCDF file if possible
        vmin(int): minimum number on visualisation legend
        vmax(int): maximum number on visualisation legend
        set_global(boolean): optional kwarg, default is True
        lonmin,lonmax,latmin,latmax(float): optional kwarg, set geographic extent is set_global kwarg is set to 
                                            False

    """
    fig=plt.figure(figsize=(20, 10))

    ax = plt.axes(projection=projection)
   
    img = plt.pcolormesh(longitude, latitude, data_array, 
                        cmap=plt.get_cmap(color_scale), transform=ccrs.PlateCarree(),
                        vmin=vmin,
                        vmax=vmax,
                        shading='auto')

    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    if (projection==ccrs.PlateCarree()):
        ax.set_extent([lonmin, lonmax, latmin, latmax], projection)
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels=False
        gl.right_labels=False
        gl.xformatter=LONGITUDE_FORMATTER
        gl.yformatter=LATITUDE_FORMATTER
        gl.xlabel_style={'size':14}
        gl.ylabel_style={'size':14}

    if(set_global):
        ax.set_global()
        ax.gridlines()

    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(long_name, fontsize=20, pad=20.0)

    return fig, ax


# <hr>

# ## 1. Define a discrete, not continuous color scale used in CAMS

# CAMS uses a `customized discrete, not continuous color scale` for many of their atmospheric composition charts. In this example, we want to represent the data in the CAMS style and for this, we want to define a customised color map.
# 
# The color scale consists of 14 colors, which are defined as red, green and blue color codes. As a first step, you can create a numpy array with 14 rows and 3 columns, where the color code for each color is stored. Each of the 14 colors in the color map will be multiplied by n (n=18) to create a final color map of `14*18+1=253` listed colors. The addition of 1 is to create a starting color of white.
# 
# The next step is then to create an empty numpy array (`cams`) with the shape of 253 rows and 4 columns. In a loop, which goes through each of the 14 colour codes defined in the matrix array and divides the RGB values with 256. In a final step, we create an a `ListedColormap()` object, which we can use with matplotlib plotting functions.

# In[7]:


matrix = np.array([[210, 214, 234],
                   [167, 174, 214],
                   [135, 145, 190],
                   [162, 167, 144],
                   [189, 188, 101],
                   [215, 209, 57],
                   [242, 230, 11],
                   [243, 197, 10],
                   [245, 164, 8],
                   [247, 131, 6],
                   [248, 98, 5],
                   [250, 65, 3],
                   [252, 32, 1],
                   [254, 0, 0]])


# Multiplication number
n = 18

# 'cams' is an initial empty colourmap, to be filled by the colours provided in 'matrix'.
cams = np.ones((253, 4))

# This loop fills in the empty 'cams' colourmap with each of the 14 colours in 'matrix'
# multiplied by 'n', with the first row left as 1 (white). Note that each colour value is 
# divided by 256 to normalise the colour range from 0 (black) to 1 (white). 
for i in range(matrix.shape[0]):
    cams[((i*n)+1):(((i+1)*n)+1),:] = np.array([matrix[i,0]/256, matrix[i,1]/256, matrix[i,2]/256, 1])

# The final color map is given by 'camscmp', which uses the Matplotlib class 'ListedColormap(Colormap)'
# to generate a colourmap object from the list of colours provided by 'cams'.
camscmp = ListedColormap(cams)


# <br>

# ## 2. CAMS European air quality forecasts

# First, let us discover the CAMS European air quality forecasts which produces daily air quality analyses and forecasts for the European domain at a spatial resolution of 0.1 degrees, which is approx. 10 km spatial resolution.

# ### 2.1 Request data from the ADS programmatically with the CDS API

# Let us request data from the Atmosphere Data Store programmatically with the help of the CDS API. We again set manually he CDS API credentials. First, you have to define two variables: `URL` and `KEY` which build together your CDS API key. Below, you have to replace the `#########` with your personal ADS key. Please find [here](https://ads.atmosphere.copernicus.eu/api-how-to) your personal ADS key.

# In[1]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'
KEY = '######################################'


# <br>

# The next step is then to request the data with the help of the CDS API. Below, we request 3-hourly European air quality analysis data for the period from  20 to 25 February 2021. The request below retrieves two variables, `dust` and `particulate_matter_10um`.
# 
# Let us store the dataset under the name `20210220-25_eur_aq_analysis_dust_pm10.nc`.

# In[ ]:


c = cdsapi.Client(url=URL, key=KEY)
c.retrieve(
    'cams-europe-air-quality-forecasts',
    {
        'model': 'ensemble',
        'date': '2021-02-20/2021-02-25',
        'format': 'netcdf',
        'variable': [
            'dust', 'particulate_matter_10um',
        ],
        'level': '0',
        'type': 'analysis',
        'time': [
            '00:00',
            '03:00',
            '06:00',
            '09:00',
            '12:00',
            '15:00',
            '18:00',
            '21:00',
        ],
        'leadtime_hour': '0',
    },
    './20210220-25_eur_aq_analysis_dust_pm10.nc')


# <br>

# ### 2.1 Load and browse CAMS European air quality forecasts
# 

# The European air quality forecasts data are available either in `GRIB` or in `NetCDF` format. We requested the analysis of dust and Particulate Matter of 10 microns for 20 to February 2021 in `NetCDF` format. You can use the Python library [xarray](http://xarray.pydata.org/en/stable/) and the function `open_dataset()` to load a NetCDF file as `xarray.Dataset`. A xarray Dataset is a collection of one or more variables that share the same dimensions. Below, you see that the Dataset has four dimensions, `latitude`, `level`, `longitude` and `time`, and two variables, `dust` and `pm10_conc`.

# In[8]:


ds_eur_aq = xr.open_dataset('./20210220-25_eur_aq_analysis_dust_pm10.nc')
ds_eur_aq


# <br>

# Let us inspect the coordinates of the file more in detail. You see above that the data set consists of 144 time steps, starting on 20 February 2021 00 UTC and ranging up to 5 days ahead and that the longitude values are on a [0, 360] grid. 
# 
# However, if you inspect time dimension more in detail, you see that the time is given in nanoseconds. As a next step, let us convert the time information into a human-readable time format and bring the longitude coordinates to a [-180, 180] grid.

# In[9]:


ds_eur_aq.time


# First, from the `long_name` information of the time dimension, we can retrieve the initial timestamp. With the function `strptime()` from Python's `datetime` library, we can convert it into a `datetime.datetime` object.

# In[10]:


timestamp = ds_eur_aq.time.long_name[19:27]

timestamp_init=datetime.strptime(timestamp,'%Y%m%d' )
timestamp_init


# In a next step, we then build a `DateTimeIndex` object with the help of Panda's `date_range()` function, making use of the length of the time dimension. The result is a `DateTimeIndex` object, which can be used to newly assign the time coordinate information.

# In[12]:


time_coords = pd.date_range(timestamp_init, periods=len(ds_eur_aq.time), freq='3h').strftime("%Y-%m-%d %H:%M:%S").astype('datetime64[ns]')
time_coords


# <br>

# And the last step is to assign the converted time information to the xarray.Dataset `ds_eur_aq`, with the function `assign_coords()`.

# In[13]:


ds_eur_aq = ds_eur_aq.assign_coords(time=time_coords)
ds_eur_aq


# <br>

# As a final step, we now also want to re-assign the longitude values and shift the grid from [0,360] to [-180,180]. At the end, you might want to sort the longitude values in an ascending order.

# In[14]:


ds_eur_aq = ds_eur_aq.assign_coords(longitude=(((ds_eur_aq.longitude + 180) % 360) - 180)).sortby('longitude')
ds_eur_aq


# <br>

# ### 2.2 Select the variable *dust* and plot a map of dust over Europe in February 2021

# As a next step, we want to select the variable `dust` from the Dataset above. You can select a variable from a Dataset by adding the name of the variable in square brackets. By selecting a variable, you load the variable as xarray.DataArray, which provides additional attribute information about the data, such as `units` or `standard_name`. For the variable `dust`, you see that the unit of the data is `µg/m3` and it reflects the `mass concentration of dust in air`.
# 
# Since the dimension `level` has only one entry reflecting the surface, we can apply the function `squeeze()`, which drops all coordinates with only one entry.

# In[15]:


dust = ds_eur_aq['dust']
dust = dust.squeeze(drop=True)
dust


# Let us store the attributes `units` and `species` as variables, which we can use later when visualising the data.

# In[16]:


dust_unit = dust.units
dust_name = dust.species


# <br>

# Now we can plot the data with the customised color map (`camscmp`) we have created at the beginning. The visualisation code below can be split in five main parts:
# * **Initiate a matplotlib figure:** with `plt.figure()` and an axes object
# * **Plotting function**: plot the data array with the matplotlib function `pcolormesh()`
# * **Define a geographic extent of the map**: use the minimum and maximum latitude and longitude bounds of the data
# * **Add additional mapping features**: such as coastlines, grid or a colorbar
# * **Set a title of the plot**: you can combine the `species name` and `time` information for the title

# In[17]:


# Index of forecast time step
time_index = 30

# Initiate a matplotlib figure
fig = plt.figure(figsize=(16,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

# Plotting function with pcolormesh
im = plt.pcolormesh(dust['longitude'].values, dust['latitude'].values,
                    dust[time_index,:,:], cmap=camscmp, transform=ccrs.PlateCarree())

# Define geographic extent of the map
ax.set_extent([dust.longitude.min(),dust.longitude.max(),dust.latitude.min(),dust.latitude.max()], crs=ccrs.PlateCarree())

# Add additional features such as coastlines, grid and colorbar
ax.coastlines(color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(im,fraction=0.046, pad=0.05)
cbar.set_label(dust_unit)

# Set the title of the plot
ax.set_title(dust_name + ' over Europe - ' + str(dust.time[time_index].values)+'\n', fontsize=16)


# <br>

# Above, you see that the forecast predicted dust in the atmosphere on 21 February at 6 UTC, heavily affecting parts of Spain and France.

# ### 2.3. Select the variable *pm10_conc* and plot a map of PM10 over Europe in February 2021
# 
# Let us now compare the dust forecast with the forecast of `Particulate Matter of particles with a size of 10 microns (PM10)`. Let us load the data array of the variable by adding the name of varible in square brackets. We also apply the function `squeeze()` to drop the dimension `level` which has only one entry.

# In[18]:


pm10 = ds_eur_aq['pm10_conc']
pm10 = pm10.squeeze(drop=True)
pm10


# Let us also store the attributes `units` and `species` as variables, which we can use during data visualisation.

# In[19]:


pm10_units = pm10.units
pm10_species = pm10.species


# <br>

# And now we can plot the `PM10` values with the customised color map (`camscmp`). We use the same visualisation code as before, which can be split in five main parts:
# * **Initiate a matplotlib figure:** with `plt.figure()` and an axes object
# * **Plotting function**: plot the data array with the matplotlib function `pcolormesh()`
# * **Define a geographic extent of the map**: use the minimum and maximum latitude and longitude bounds of the data
# * **Add additional mapping features**: such as coastlines, grid or a colorbar
# * **Set a title of the plot**: you can combine the `species name` and `time` information for the title

# In[20]:


# Index of forecast time step
time_index = 30

# Initiate a matplotlib figure
fig = plt.figure(figsize=(16,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

# Plotting function with pcolormesh
im = plt.pcolormesh(pm10['longitude'].values, pm10['latitude'].values,
                    pm10[time_index,:,:], cmap=camscmp, transform=ccrs.PlateCarree())

# Define geographic extent of the map
ax.set_extent([pm10.longitude.min(),pm10.longitude.max(),pm10.latitude.min(),pm10.latitude.max()], crs=ccrs.PlateCarree())

# Add additional features such as coastlines, grid and colorbar
ax.coastlines(color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(im,fraction=0.046, pad=0.05)
cbar.set_label(pm10_units)

# Set the title of the plot
ax.set_title(pm10_species + ' over Europe - ' + str(pm10.time[time_index].values)+'\n', fontsize=16)


# <br>

# Above, you see that on 21 February 2021, for many parts in central Spain, the PM10 values have by far exceeded the daily limit of 50 micrograms per cubic metre for particulate matter (PM10). This limit is defined in the EU air quality directive (2008/EC/50).

# <br>

# ### 2.4. Animate dust concentration over Europe from 20 to 25 February 2021

# In the last step, you can animate the `dust concentration` over Europe in order to see how the trace gas develops over a period of five days, from 20 to 25 February 2021.
# You can do animations with matplotlib's function `animation`. Jupyter's function `HTML` can then be used to display HTML and video content.

# The animation function consists of 4 parts:
# - **Setting the initial state:**<br>
#  Here, you define the general plot your animation shall use to initialise the animation. You can also define the number of frames (time steps) your animation shall have.
#  
#  
# - **Functions to animate:**<br>
#  An animation consists of three functions: `draw()`, `init()` and `animate()`. `draw()` is the function where individual frames are passed on and the figure is returned as image. In this example, the function redraws the plot for each time step. `init()` returns the figure you defined for the initial state. `animate()` returns the `draw()` function and animates the function over the given number of frames (time steps).
#  
#  
# - **Create a `animate.FuncAnimation` object:** <br>
#  The functions defined before are now combined to build an `animate.FuncAnimation` object.
#  
#  
# - **Play the animation as video:**<br>
#  As a final step, you can integrate the animation into the notebook with the `HTML` class. You take the generate animation object and convert it to a HTML5 video with the `to_html5_video` function

# In[22]:


# Setting the initial state:
# 1. Define figure for initial plot
fig, ax = visualize_pcolormesh(data_array=dust[0,:,:],
                               longitude=dust.longitude, 
                               latitude=dust.latitude,
                               projection=ccrs.PlateCarree(), 
                               color_scale=camscmp, 
                               unit=dust_unit,
                               long_name=dust_name + ' '+ str(dust.time[0].data),
                               vmin=0,
                               vmax=250, 
                               lonmin=dust.longitude.min(), 
                               lonmax=dust.longitude.max(), 
                               latmin=dust.latitude.min(), 
                               latmax=dust.latitude.max(),
                               set_global=False)

frames = 47

def draw(i):
    img = plt.pcolormesh(dust.longitude, 
                         dust.latitude, 
                         dust[i,:,:], 
                         cmap=camscmp, 
                         transform=ccrs.PlateCarree(),
                         vmin=0,
                         vmax=250,
                         shading='auto')
    
    ax.set_title(dust_name + ' '+ str(dust.time[i].data), fontsize=20, pad=20.0)
    return img


def init():
    return fig


def animate(i):
    return draw(i)

ani = animation.FuncAnimation(fig, animate, frames, interval=500, blit=False,
                              init_func=init, repeat=True)

HTML(ani.to_html5_video())
plt.close(fig)


# <br>

# **Play the animation video as HTML5 video**

# In[23]:


HTML(ani.to_html5_video())


# <br>

# ## 3. CAMS global atmospheric composition forecasts

# Now, let us discover the CAMS global atmospheric composition forecasts which produces global forecasts for more than 50 chemical species twice a day. The forecasts have a spatial resolution of 0.4 degrees and provide hourly forecasts for up to 5 days (leadtime hour 120h) ahead.

# ### 3.1 Request data from the ADS programmatically with the CDS API

# Let us request data from the Atmosphere Data Store programmatically with the help of the CDS API. We again set manually he CDS API credentials. First, you have to define two variables: `URL` and `KEY` which build together your CDS API key. Below, you have to replace the `#########` with your personal ADS key. Please find [here](https://ads.atmosphere.copernicus.eu/api-how-to) your personal ADS key.

# In[6]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'
KEY = '###########################'


# <br>

# The next step is then to request the data with the help of the CDS API. Below, we request global `Total Aerosol Optical Depth at 550nm` forecast data for 20 February 2021 with the model runtime at 00:00 UTC. The request below requests `forecast` data for every 3 hours up to hour 90 in advance.
# 
# Let us store the dataset under the name `20210220_cams_global_forecast_aod.zip`.

# In[ ]:


import cdsapi
c = cdsapi.Client(url=URL, key=KEY)
c.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': 'total_aerosol_optical_depth_550nm',
        'date': '2021-02-20/2021-02-20',
        'time': '00:00',
        'leadtime_hour': [
            '0', '12', '15',
            '18', '21', '24',
            '27', '3', '30',
            '33', '36', '39',
            '42', '45', '48',
            '51', '54', '57',
            '6', '60', '63',
            '66', '69', '72',
            '75', '78', '81',
            '84', '87', '9',
            '90',
        ],
        'type': 'forecast',
        'format': 'netcdf_zip',
    },
    './20210220_cams_global_forecast_aod.zip')


# <br>

# ### 3.2 Unzip the downloaded data file

# CAMS global atmospheric composition forecasts can be retrieved either in `GRIB` or in a `zipped NetCDF`. In this [data request](./200_atmosphere_data_store_intro.ipynb#cams_global_forecast), we requested the data in a zipped NetCDF and for this reason, we have to unzip the file before we can open it. You can unzip `zip archives` in Python with the Python package `zipfile` and the function `extractall()`. The file extracted is called per default `data.nc`.

# In[9]:


import zipfile
with zipfile.ZipFile('./20210220_cams_global_forecast_aod.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# <br>

# ### 3.3 Load and browse CAMS global forecast of total Aerosol Optical Depth at 550nm

# Once the data has been extracted from the zip archive, you can load the netCDF file with the Python library [xarray](http://xarray.pydata.org/en/stable/) and the function `open_dataset()`. The function loads a `xarray.Dataset`, which is a collection of one or more data variables that share the same dimensions. You see that the data files has three dimensions, `latitude`, `longitude` and `time` and one variable, `aod550`. 

# In[27]:


ds_global_aod = xr.open_dataset('./data.nc')
ds_global_aod


# <br>

# Let us now extract from the Dataset above the data variable `aod550` as `xarray.DataArray`. You can load a data array from a xarray.Dataset by specifying the name of the variable (`aod550`) in square brackets.

# In[28]:


aod550 = ds_global_aod['aod550']
aod550


# <br>

# If we inspect the `time` dimension a bit closer, you see that the data has 31 time steps representing the forecast of `Total Aerosol Optical Depth at 550 nm` from 20 February at 00:00 UTC every three hours up to 23 February 2021 18:00 UTC.

# In[29]:


aod550.time


# <br>

# The loaded data array above also has two attributes, `units` and `long_name`. Let us define two variables for those attributes. The variables can be used later during data visualisation.

# In[30]:


aod_unit = aod550.units
aod_long_name = aod550.long_name


# <br>

# ### 3.4 Visualize a global map of total AOD at 550nm in February 2021
# 

# And now we can plot the `Total Aerosol Optical Depth at 550 nm` values with the customised color map (`camscmp`). We use the same visualisation code as before (without setting a geographic extent), which can be split in four main parts:
# * **Initiate a matplotlib figure:** with `plt.figure()` and an axes object
# * **Plotting function**: plot the data array with the matplotlib function `pcolormesh()`
# * **Add additional mapping features**: such as coastlines, grid or a colorbar
# * **Set a title of the plot**: you can combine the `species name` and `time` information for the title

# In[31]:


time_index =  10

# Initiate the matplotlib figure
fig = plt.figure(figsize=(16,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

# Plotting function with pcolormesh
im = plt.pcolormesh(aod550.longitude, aod550.latitude, aod550[time_index,:,:],
                    cmap=camscmp, transform=ccrs.PlateCarree())

# Add additional mapping features
ax.coastlines(color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(im,fraction=0.046, pad=0.05)
cbar.set_label(aod_unit)

# Set a title of the plot
ax.set_title(aod_long_name + ' - ' + str(aod550.time[time_index].values)+'\n', fontsize=16)


# <br>

# <br>

# ### 3.5 Animate global total AOD at 550nm from 20 to 23 February 2021

# In the last step, you can animate the `total AOD at 550nm` in order to see how the trace gas develops over a period of four days, from 20 to 23 February 2021.
# You can do animations with matplotlib's function `animation`. Jupyter's function `HTML` can then be used to display HTML and video content.

# The animation function consists of 4 parts:
# - **Setting the initial state:**<br>
#  Here, you define the general plot your animation shall use to initialise the animation. You can also define the number of frames (time steps) your animation shall have.
#  
#  
# - **Functions to animate:**<br>
#  An animation consists of three functions: `draw()`, `init()` and `animate()`. `draw()` is the function where individual frames are passed on and the figure is returned as image. In this example, the function redraws the plot for each time step. `init()` returns the figure you defined for the initial state. `animate()` returns the `draw()` function and animates the function over the given number of frames (time steps).
#  
#  
# - **Create a `animate.FuncAnimation` object:** <br>
#  The functions defined before are now combined to build an `animate.FuncAnimation` object.
#  
#  
# - **Play the animation as video:**<br>
#  As a final step, you can integrate the animation into the notebook with the `HTML` class. You take the generate animation object and convert it to a HTML5 video with the `to_html5_video` function

# In[ ]:


# Setting the initial state:
# 1. Define figure for initial plot
fig, ax = visualize_pcolormesh(data_array=aod550[0,:,:],
                               longitude=aod550.longitude, 
                               latitude=aod550.latitude,
                               projection=ccrs.PlateCarree(), 
                               color_scale=camscmp, 
                               unit=aod_unit,
                               long_name=aod_long_name + ' '+ str(aod550.time[0].data),
                               vmin=0,
                               vmax=2, 
                               lonmin=aod550.longitude.min(), 
                               lonmax=aod550.longitude.max(), 
                               latmin=aod550.latitude.min(), 
                               latmax=aod550.latitude.max(),
                               set_global=False)

frames = 31

def draw(i):
    img = plt.pcolormesh(aod550.longitude, 
                         aod550.latitude, 
                         aod550[i,:,:], 
                         cmap=camscmp, 
                         transform=ccrs.PlateCarree(),
                         vmin=0,
                         vmax=2,
                         shading='auto')
    
    ax.set_title(aod_long_name + ' '+ str(aod550.time[i].data), fontsize=20, pad=20.0)
    return img


def init():
    return fig


def animate(i):
    return draw(i)

ani = animation.FuncAnimation(fig, animate, frames, interval=500, blit=False,
                              init_func=init, repeat=True)

HTML(ani.to_html5_video())
plt.close(fig)


# <br>

# **Play the animation video as HTML5 video**

# In[33]:


HTML(ani.to_html5_video())


# <hr>

# <p></p>
# <span style='float:right'><p style=\"text-align:right;\">This project is licensed under <a href="./LICENSE">APACHE License 2.0</a>. | <a href=\"https://github.com/ecmwf-projects/copernicus-training">View on GitHub</a></span>
