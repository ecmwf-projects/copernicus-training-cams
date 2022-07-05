#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# <br>

# # Vertical Cross Section Plot

# ### About

# This notebook provides you a practical introduction to the CAMS global atmospheric composition forecasts on single and pressure levels. We will use the data to analyse the transport of large plumes of sulphur dioxide over the Atlantic Ocean, after the eruptions of the Cumbre Vieja Volcano on La Palma in the Canary Islands on 19 September 2021. We first animate the plumes of the total column Sulphur Dioxide during October 2021. Afterwards, we will create a vertical cross-section plot of Sulphur Dioxide on different pressure levels to better assess the vertical distribution of SO<sub>2</sub> in the atmosphere over the Atlantic Ocean in October 2021.

# The notebook has the following outline:
# 
# * 1 - Request data from the ADS programmatically with the CDS API
# * 2 - Unzip the downloaded data file
# * 3 - Animation of total column sulphur dioxide over the Atlantic Ocean in October 2021
# * 4 - Compute meridional mean plot of Sulphur Dioxide over the Atlantic Ocean in October 2021

# ### Data

# This notebook introduces you to the [CAMS global atmospheric composition forecasts and analyses](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-atmospheric-composition-forecasts?tab=overview). The data has the following specifications:
# 
# > **Data**: `CAMS global atmospheric composition forecasts` <br>
# > **Variables**: `['sulphur_dioxide', 'total_column_sulphur_dioxide']` <br>
# > **Temporal coverage**: `1 to 31 October 2021` <br>
# > **Spatial coverage**: `Geographical subset: N:60, W:-90, S:10, E:0` <br>
# > **Level**: `surface and pressure levels` <br>
# > **Format**: `zipped NetCDF`
# 

# ### How to access the notebook
# 
# This tutorial is in the form of a [Jupyter notebook](https://jupyter.org/). You will not need to install any software for the training as there are a number of free cloud-based services to create, edit, run and export Jupyter notebooks such as this. Here are some suggestions (simply click on one of the links below to run the notebook):

# |Binder|Kaggle|Colab|NBViewer|
# |:-:|:-:|:-:|:-:|
# |[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training/HEAD?urlpath=lab/tree/CAMS_vertical-cross-section-volcanic-eruption.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training/blob/master/CAMS_vertical-cross-section-volcanic-eruption.ipynb)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf-projects/copernicus-training/blob/master/CAMS_vertical-cross-section-volcanic-eruption.ipynb)|[![NBViewer](https://raw.githubusercontent.com/ecmwf-projects/copernicus-training/master/img/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/ecmwf-projects/copernicus-training/blob/master/CAMS_vertical-cross-section-volcanic-eruption.ipynb)|
# |(Binder may take some time to load, so please be patient!)|(will need to login/register, and switch on the internet via *settings*)|(will need to run the command `!pip install cartopy` before importing the libraries)|(this will not run the notebook, only render it)|

# If you would like to run this notebook in your own environment, we suggest you install [Anaconda](https://docs.anaconda.com/anaconda/install/), which contains most of the libraries you will need. You will also need to install [Xarray](http://xarray.pydata.org/en/stable/) for working with multidimensional data in netcdf files, and the ADS API (`pip install cdsapi`) for downloading data programatically from the ADS.

# ### Further resources
# * [CAMS monitors transport of SO2 from La Palma volcano](https://atmosphere.copernicus.eu/cams-monitors-transport-so2-la-palma-volcano)

# <hr>

# ### Install CDSAPI via pip

# In[ ]:


get_ipython().system('pip install cdsapi')


# ### Load libraries

# In[3]:


# CDS API
import cdsapi
import os

# Libraries for working with multi-dimensional arrays
import numpy as np
import xarray as xr
import pandas as pd

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature


# ### Load helper functions

# In[12]:


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

# ## 1. Request data from the ADS programmatically with the CDS API

# Let us request data from the Atmosphere Data Store programmatically with the help of the CDS API. We again set manually he CDS API credentials. First, you have to define two variables: `URL` and `KEY` which build together your CDS API key. Below, you have to replace the `#########` with your personal ADS key. Please find [here](https://ads.atmosphere.copernicus.eu/api-how-to) your personal ADS key.

# In[5]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'
KEY = '######################################'


# <br>

# The next step is then to request the data with the help of the CDS API. Below, we request analysis data from the `CAMS global atmospheric composition forecast` dataset. We request Sulphur Dioxide on single levels (`total_column_sulphur_dioxide`) and on different pressure levels (`sulphur dioxide`) for the month of October 2021 for a geographical subset over the Atlantic Ocean.
# 
# Let us store the dataset under the name `202110_global_analysis_so2.netcdf_zip`.

# In[ ]:


c = cdsapi.Client(url=URL, key=KEY)

c.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': [
            'sulphur_dioxide', 'total_column_sulphur_dioxide',
        ],
        'date': '2021-10-01/2021-10-30',
        'time': [
            '00:00', '12:00',
        ],
        'leadtime_hour': '0',
        'type': 'forecast',
        'area': [
            60, -90, 10,
            0,
        ],
        'format': 'netcdf_zip',
        'pressure_level': [
            '200', '250', '300',
            '400', '500', '600',
            '700', '800', '850',
            '900', '925', '950',
            '1000',
        ],
    },
    './data/cams/202110_global_analysis_so2.netcdf_zip')


# <br>

# ## 2. Unzip the downloaded data file

# The CAMS global atmospheric composition forecasts can be retrieved from the Atmosphere Data Store in two data formats, `GRIB` and `Zipped netCDF`. We downloaded the data as `Zipped netCDF` (see CDSAPI request [here](./200_atmosphere_data_store_intro.ipynb#cams_forecast_pl_so2)) and for this reason, before we can load any data, we have to unzip the files. You can use the functions from the `zipfile` Python package to extract the content of a zip file. First, you construct a `ZipFile()` object and second, you apply the function `extractall()` to extract the content of the zip file.
# 
# From the zip archive, two `NetCDF` files are extracted: one with the name ending `_pl`, which contains the SO<sub>2</sub> values on the chosen pressure levels and one with the name ending `_sfc`, which contains the total column SO<sub>2</sub> values. 

# In[5]:


import zipfile
with zipfile.ZipFile('./202110_global_analysis_so2.netcdf_zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# <br>

# ## 2. Animation of total column sulphur dioxide over the Atlantic Ocean in October 2021

# As a first step, we want to load the `total column sulphur dioxide` and animate the variable over the Atlantic Ocean in October 2021. Once the data has been extracted from the zip archive, you can load the NetCDF file with the Python library [xarray](http://xarray.pydata.org/en/stable/) and the function `open_dataset()`. The function loads a `xarray.Dataset`, which is a collection of one or more data variables that share the same dimensions. You see that the data file has three dimensions, `latitude`, `longitude` and `time` and one variable, `tcso2. 

# In[4]:


ds_so2_tc = xr.open_dataset('./levtype_sfc.nc')
ds_so2_tc


# <br>

# Let us now extract from the Dataset above the data variable `tcso2` as `xarray.DataArray`. You can load a data array from a xarray.Dataset by specifying the name of the variable (`tcso2`) in square brackets. A `xarray.DataArray` object offers us also additional metadata information, such as the `units` and `long_name` attributes.

# In[5]:


da_so2_tc = ds_so2_tc['tcso2']
da_so2_tc


# <br>

# From the `units` attribute above, we see that the `Total column Sulphur dioxide` values are provided in `kg m**-2`, which is the SI unit. However, the concentration of a certain trace gas, such as SO<sub>2</sub>, in a column of air in the Earth's atmosphere is often given in `Dobson Units [DU]`. The `Dobson Unit` indicates how much of a given trace gas there is in the air above a certain point on earth. For SO<sub>2</sub>, the typical background level concentration (i.e. away from emissions related to pollution and volcanic eruptions) is much less than 1 DU. Emissions related to pollution and small volcanic eruptions are of the order of 1 DU or a few DU. Strong and explosive eruptions may lead to concentrations well above 10 DU, even as high as 100 DU.
# 
# In a next step, we would like to convert the SO<sub>2</sub>values from `kg m**-2` to `Dobson Units`. The conversion rate is as follows:
# * `1 Dobson Unit [DU] is equal to 2.1415 x 10-5 kg[SO2]/m2`
# 
# Thus, to convert the SO<sub>2</sub> to DU, we have to divide the SO<sub>2</sub> by 2.1415*10**-5.
# 
# **Note:** this conversion leads to a drop of the data array attributes, as we modified the data values.

# In[6]:


da_so2_tc_du = da_so2_tc / (2.1415*10**-5)
da_so2_tc_du


# <br>

# The next step is to re-assign the attributes from before and to change the `units` attribute to `DU` for Dobosn unit. You can re-assign attributes to a data array with the function `assign_attrs()`.

# In[7]:


da_so2_tc_du = da_so2_tc_du.assign_attrs(da_so2_tc.attrs)
da_so2_tc_du.attrs['units'] = 'DU'


# <br>

# And now we can plot the `Total column Sulphur Dioxide` values in Dobson Unit with matplotlib's function `pcolormesh()`.The visualisation code below can be split in four main parts:
# * **Initiate a matplotlib figure:** with `plt.figure()` and an axes object
# * **Plotting function**: plot the data array with the matplotlib function `pcolormesh()`
# * **Add additional mapping features**: such as coastlines, grid or a colorbar
# * **Set a title of the plot**: you can combine the `species name` and `time` information for the title

# In[8]:


time_index =  10

# Initiate the matplotlib figure
fig = plt.figure(figsize=(16,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

# Plotting function with pcolormesh
im = plt.pcolormesh(da_so2_tc_du.longitude, 
                    da_so2_tc_du.latitude, 
                    da_so2_tc_du[time_index,:,:],
                    cmap='afmhot_r', 
                    transform=ccrs.PlateCarree(), 
                    vmin=0, 
                    vmax=10)

# Add additional mapping features
ax.coastlines(color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(im,fraction=0.046, pad=0.05)
cbar.set_label(da_so2_tc_du.units)

# Set a title of the plot
ax.set_title(da_so2_tc_du.long_name + ' - ' + str(da_so2_tc_du.time[time_index].values)+'\n', fontsize=16)


# <br>

# In the last step, you can animate the `Total column Sulphur Dioxide in DU` in order to see how the trace gas develops over the Atlantic in October 2021.
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

# In[16]:


# Setting the initial state:
# 1. Define figure for initial plot
fig, ax = visualize_pcolormesh(data_array=da_so2_tc_du[0,:,:],
                               longitude=da_so2_tc_du.longitude, 
                               latitude=da_so2_tc_du.latitude,
                               projection=ccrs.PlateCarree(), 
                               color_scale='afmhot_r', 
                               unit=da_so2_tc_du.units,
                               long_name=da_so2_tc_du.long_name + ' '+ str(da_so2_tc_du.time[0].data),
                               vmin=0,
                               vmax=10, 
                               lonmin=da_so2_tc_du.longitude.min(), 
                               lonmax=da_so2_tc_du.longitude.max(), 
                               latmin=da_so2_tc_du.latitude.min(), 
                               latmax=da_so2_tc_du.latitude.max(),
                               set_global=False)

frames = 59

def draw(i):
    img = plt.pcolormesh(da_so2_tc_du.longitude, 
                         da_so2_tc_du.latitude, 
                         da_so2_tc_du[i,:,:], 
                         cmap='afmhot_r', 
                         transform=ccrs.PlateCarree(),
                         vmin=0,
                         vmax=10,
                         shading='auto')
    
    ax.set_title(da_so2_tc_du.long_name + ' '+ str(da_so2_tc_du.time[i].data), fontsize=20, pad=20.0)
    return img


def init():
    return fig


def animate(i):
    return draw(i)

ani = animation.FuncAnimation(fig, animate, frames, interval=500, blit=False,
                              init_func=init, repeat=True)


HTML(ani.to_html5_video())
plt.close(fig)


# In[19]:


HTML(ani.to_html5_video())


# <br>

# ## 3. Compute meridional mean plot of Sulphur Dioxide over the Atlantic Ocean in October 2021

# The next step is now to load the Sulphur Dioxide on different pressure levels to create a meridional mean plot to visualise the changing concentrations of SO<sub>2</sub> at different longitudes and at various pressure levels of the atmosphere. We will focus on the latitudes between 20 and 40 degrees and average over the latitudinal axis. This allows us to visualise a two dimensional plot of SO<sub>2</sub> concentrations by longitude and altitude.
# 
# 
# In a first step, we want to load the file with the ending `_pl`, which contains the Sulphur Dioxide values on different pressure levels. You can use the Python library [xarray](http://xarray.pydata.org/en/stable/) and its function `open_dataset` to read a NetCDF file. The result is a `xarray.Dataset` with four dimensions: `latitude`, `level`, `longitude` and `time`.

# In[14]:


ds_so2 = xr.open_dataset('./levtype_pl.nc')
ds_so2


# <br>

# The next step is to extract the data variable `so2`. You can extract a data variable as `xarray.DataArray` by specifying the name of the variable in square brackets. The loaded data array reveals additional attributes of the variable, such as `units` and `long_name`. 

# In[15]:


da_so2 = ds_so2['so2']
da_so2


# <br>

# Above, you see that the sulphur dioxide values on pressure levels are disseminated as mass mixing ratios in kg of gas / kg of air. Trace gas concentrations at a particular pressure level in the atmosphere are often given as a `volume mixing ratio`, or simply mixing ratio. This unit is defined as the ratio of the number density of the gas to the total number density of the atmosphere. 
# 
# In other words, the SO<sub>2</sub> volume mixing ratio is the density of SO<sub>2</sub> divided by the density of all constituents of the atmosphere in a unit volume (i.e. the number of molecules per unit volume).
# Therefore, an SO<sub>2</sub> mixing ratio of 10**-9 means that the number density of SO<sub>2</sub> is 10**-9 times the total number density of air in a unit volume. Following the standard convention for the earth's troposphere and stratosphere, this mixing ratio equals 1 ppbv (parts per billion by volume).
# 
# To convert data from MMR to VMR you only need molar masses of dry air and molar mass of the atmospheric species. For SO<sub>2</sub> the formula is: 
# * `VMR = 28.9644 / 64.0638 * 1e9 * MMR`
# 
# Find [here](https://confluence.ecmwf.int/pages/viewpage.action?pageId=153391710) more information about converting mass mixing ratio (MMR) to volume mixing ratio (VMR).
# 
# Below, we apply the formula above and convert the SO<sub>2</sub> values to volume mixing ratio. <br>
# **Note:** You will see that the operation drops the data array's attributes. We will re-assign the attributes after some more pre-processing steps.

# In[16]:


da_so2_vmr = 28.9644 / 64.0638 * 1e9 * da_so2
da_so2_vmr


# <br>

# In a next step, we create the daily averages of Sulphur Dioxide in the atmosphere over the Atlantic. The xarray function `resample()` allows us to resample the time dimension from a 12-hourly resolution to a daily average. With the function `resample()` you define the resolution and with a subsequent aggregation function (e.g. mean), you define the aggregation level.

# In[17]:


da_so2_daily = da_so2_vmr.resample(time='1D').mean()
da_so2_daily


# <br>

# ### 3.3. Compute meridional mean plot of Sulphur Dioxide on different pressure levels 

# The next step is now to prepare the data in order to create a meridional mean plot to visualise the changing concentrations of SO<sub>2</sub> at different longitudes and at various pressure levels of the atmosphere. We will focus on the latitudinal zone between 20° and 40° and average over the latitudinal axis. This allows us to visualise a two dimensional plot of SO<sub>2</sub> by longitude and altitude.

# The first step is to filter the data array for all latitude entries between 20° and 40°. The xarray function `where()` allows us to filter any dimension of a data array. You can specify the keyword argument `drop`, which drops the values that were filtered out by the operation.
# 
# The size of the latitude dimension of the resulting data array has decreased from 126 to 51 grid points.

# In[18]:


so2_lat_area = da_so2_daily.where((da_so2_daily.latitude >= 20) & (da_so2_daily.latitude <= 40), drop=True)
so2_lat_area


# In a last step before we are able to visualize the meridional mean plot of Sulphur Dioxide, we have to compute the average over the latitudinal axis. You can use the xarray function `mean()` to create the average over the `latitude` dimension. The result is then a data array with three dimensions: `time`, `level` and `longitude`.

# In[19]:


so2_lat_area_mean = so2_lat_area.mean(dim="latitude")
so2_lat_area_mean


# Let us now also add the attributes again to the data array, as we might want to use the `units` and `long_name` for title and label information in the resulting plot. Since we made various changes to the data array, we first can assign the attributes from the initial data array `da_so2`, but then modify and update specific attribute keys. The attributes we want to update are:
# * `units`: `ppbv`
# * `long_name`: `Sulphur dioxide volume mixing ratio`
# 

# In[20]:


so2_lat_area_mean = so2_lat_area_mean.assign_attrs(da_so2.attrs)
so2_lat_area_mean.attrs['long_name'] = 'Sulphur dioxide volume mixing ratio'
so2_lat_area_mean.attrs['units'] = 'ppbv'

so2_lat_area_mean.attrs


# <br>

# ### 3.4. Visualize the daily meridional mean of Sulphur Dioxide as cross-section plot

# Now we can visualize the vertical Sulphur dioxide distribution over the Atlantic in October 2021. The pressure levels are on a non-linear scale. To ensure equal spacing between the values, let us create a regular scale.

# In[21]:


y = np.arange(so2_lat_area_mean.level.shape[0])+1
y = y[::-1]


# <br>

# The visualization code can be divided in five main parts:
# * **Initiate a matplotlib figure**: Initiatie a matplotlib figure object with `plt.figure()` and `plt.subplot()`
# * **Plotting the data**: Plot the array with matplotlib's function `pcolormesh` with longitude on the x-axis and pressure levels on the y-axis
# * **Set y-axis tickmarks and labels**: Set regular-scaled y-axis ticks and labels
# * **Set axes labels and title**
# * **Specify a colorbar**: Specify a colorbar to be placed on the right and use the units attribute to add to the colorbar as unit

# In[22]:


time_index = 15

# Define the figure and specify size
fig = plt.figure(figsize=(16, 8))
ax = plt.subplot()

# Plot the figure with pcolormesh
im = plt.pcolormesh(so2_lat_area_mean.longitude, y, so2_lat_area_mean[time_index,:,:], cmap='jet', shading='auto')

# Set x and y axis tickmarks, labels
ax.yaxis.set_ticks(y)
ax.yaxis.set_ticklabels(so2_lat_area_mean.level.values)

# Set axes labels and title
ax.set_xlabel('\nLongitude', fontsize=14)
ax.set_ylabel('Atmospheric pressure level (millibars)\n', fontsize=14)
ax.set_title('\n'+ so2_lat_area_mean.long_name + ' - ' + str(so2_lat_area_mean.time[time_index].values)[0:10] + '\n', fontsize=16)

# Specify a colorbar
cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
cbar.set_label('\n'+ so2_lat_area_mean.units, fontsize=14)


# The plot above shows you how the concentration and longitudinal distribution of Sulphur Dioxide over the Atlantic varies at different pressure levels and at different longitudes. You see that most of the SO<sub>2</sub> concentrations from the Cumbre Vieja Volcano eruption was transported over the Atlantic in higher atmospheric levels at a level between 700 to 400 millibars.

# <hr>

# <p></p>
# <span style='float:right'><p style=\"text-align:right;\">This project is licensed under <a href="./LICENSE">APACHE License 2.0</a>. | <a href=\"https://github.com/ecmwf-projects/copernicus-training">View on GitHub</a></span>
