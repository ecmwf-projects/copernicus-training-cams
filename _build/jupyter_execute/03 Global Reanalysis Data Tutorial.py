#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# <br>

# # CAMS global reanalysis (EAC4) monthly averaged fields

# ### About

# This notebook provides you a practical introduction to the CAMS Global reanalysis (EAC4) data set. We will use the monthly averaged fields to analyse the distribution of Carbon Monoxide in the northern hemisphere and at different vertical layers in the atmosphere for August 2020. Carbon Monoxide is a colorless, odorless gas exisiting in the atmosphere, which is produced by burning (combustion) of fossil fuels (coal, oil and natural gas) and biomass (for example, woood in forest fires).

# The notebook has the following outline:
# 
# * 1 - Request data from the ADS programmatically with the CDS API
# * 2 - Unzip the downloaded data files
# * 3 - Load the data with xarray
# * 4 - Compute meridional mean plot of Carbon Monoxide
# * 5 - Visualize the meridional mean plot of Carbon Monoxide in August 2020

# ### Data

# This notebook introduces you to the CAMS global reanalysis (EAC4) monthly averaged fields. The data has the following specifications:
# 
# > **Data**: `CAMS global reanalysis (EAC4) monthly averaged fields` <br>
# > **Temporal coverage**: `Aug 2020` <br>
# > **Spatial coverage**: `Global` <br>
# > **Format**: `zipped NetCDF files`
# 

# ### How to access the notebook
# 
# This tutorial is in the form of a [Jupyter notebook](https://jupyter.org/). You will not need to install any software for the training as there are a number of free cloud-based services to create, edit, run and export Jupyter notebooks such as this. Here are some suggestions (simply click on one of the links below to run the notebook):

# |Binder|Kaggle|Colab|NBViewer|
# |:-:|:-:|:-:|:-:|
# |[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training/HEAD?urlpath=lab/tree/CAMS_global-reanalysis.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training/blob/master/CAMS_global-reanalysis.ipynb)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf-projects/copernicus-training/blob/master/CAMS_global-reanalysis.ipynb)|[![NBViewer](https://raw.githubusercontent.com/ecmwf-projects/copernicus-training/master/img/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/ecmwf-projects/copernicus-training/blob/master/CAMS_global-reanalysis.ipynb)|
# |(Binder may take some time to load, so please be patient!)|(will need to login/register, and switch on the internet via *settings*)|(will need to run the command `!pip install cartopy` before importing the libraries)|(this will not run the notebook, only render it)|

# If you would like to run this notebook in your own environment, we suggest you install [Anaconda](https://docs.anaconda.com/anaconda/install/), which contains most of the libraries you will need. You will also need to install [Xarray](http://xarray.pydata.org/en/stable/) for working with multidimensional data in netcdf files, and the ADS API (`pip install cdsapi`) for downloading data programatically from the ADS.

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

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature


# <hr>

# ## 1. Request data from the ADS programmatically with the CDS API

# Let us request data from the Atmosphere Data Store programmatically with the help of the CDS API. We again set manually he CDS API credentials. First, you have to define two variables: `URL` and `KEY` which build together your CDS API key. Below, you have to replace the `#########` with your personal ADS key. Please find [here](https://ads.atmosphere.copernicus.eu/api-how-to) your personal ADS key.

# In[6]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'
KEY = '###########################'


# <br>

# The next step is then to request the data with the help of the CDS API. Below, we request the monthly averages of CAMS global reanalysis data. We request the variable `Carbon Monoxide` for August 2020 on different pressure levels, from the surface up to pressure level 100 millibar.
# 
# Let us store the dataset under the name `cams_eac4_202008_pl.zip`.

# In[ ]:


c = cdsapi.Client(url=URL, key=KEY)

c.retrieve(
    'cams-global-reanalysis-eac4-monthly',
    {
        'format': 'netcdf',
        'variable': 'carbon_monoxide',
        'pressure_level': [
            '100', '150', '200',
            '250', '300', '400',
            '500', '600', '700',
            '800', '850', '900',
            '925', '950', '1000',
        ],
        'model_level': '60',
        'year': '2020',
        'month': '08',
        'product_type': 'monthly_mean',
    },
    './cams_eac4_202008_pl.zip')


# <br>

# ## 2. Unzip the downloaded data files

# The Atmosphere Data Store allows you to choose between two data formats, `GRIB` and `NetCDF`. However, the data is archived in `zip` file when downloaded. For this reason, before we can load any data, we have to unzip the files. Once you downloaded the data as a zip file (see CDSAPI request [here](./200_atmosphere_data_store_intro.ipynb#eac4_monthly)), you can use the functions from the `zipfile` Python package to extract the content of a zip file. First, you construct a `ZipFile()` object and second, you apply the function `extractall()` to extract the content of the zip file.

# In[3]:


import zipfile
with zipfile.ZipFile('./cams_eac4_202008_pl.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# <br>

# ## 3. Load the data with xarray

# From the zip archive, two `NetCDF` files are extracted: one with the name ending `_pl`, which contains the CO values on the chosen pressure levels and one with the name ending `_ml`, which contains the CO values on the surface. Since we are interested in the vertical distribution of CO in the northern hemisphere in August 2020, we want to load the file containing CO values on different pressure levels.
# 
# You can use the Python library [xarray](http://xarray.pydata.org/en/stable/) and its function `open_dataset` to read a NetCDF file. The result is a `xarray.Dataset` with four dimensions: `latitude`, `level`, `longitude`, `time`.

# In[6]:


ds_co = xr.open_dataset('./levtype_pl.nc')
ds_co


# The next step is to extract the data variable `co`. You can extract a data variable as `xarray.DataArray` by specifying the name of the variable in square brackets. The loaded data array reveals additional attributes of the variable, such as `units` and `long_name`. 

# In[7]:


da_co = ds_co['co']
da_co


# <br>

# If you have a closer look at the longitude values in the data array above, you see that the CAMS global reanalysis data are on a [0, 360] grid. Let us bring the longitude values to a [-180, 180] grid. You can use the xarray function `assign_coords()` to do so.
# 
# If you inspect the longitude values after the operation, you see that the values now range from -180 to 180.

# In[8]:


da_co_assigned = da_co.assign_coords(longitude=(((da_co.longitude + 180) % 360) - 180)).sortby('longitude')
da_co_assigned


# <br>

# ## 4. Compute meridional mean plot of Carbon Monoxide

# The next step is now to prepare the data in order to create a meridional mean plot to visualise the changing concentrations of CO at different longitudes and at various levels of the atmosphere. We will focus on the northern hemisphere and average over the latitudinal axis. This allows us to visualise a two dimensional plot of CO concentrations by longitude and altitude.

# The first step is to filter all data values over the northern hemisphere. The xarray function `where()` allows us to filter any dimension of a data array. Let us filter the array and keep only data where the latitude dimension is positive. You can specify the keyword argument `drop`, which drops the values that were filtered out by the operation.
# 
# The size of the latitude dimension of the resulting data array has decreased by half, from 241 to 121 grid points.

# In[9]:


north = da_co_assigned.where((da_co_assigned.latitude >= 0), drop=True)
north


# In a last step before we are able to visualize the meridional mean plot of Carbon Monoxide, we have to compute the average over the latitudinal axis. You can use the xarray function `mean()` to create the average over the `latitude` dimension. The result is then a data array with three dimensions: `time`, `level` and `longitude`.

# In[10]:


co = north.mean(dim="latitude")
co


# Above, you see that the operation drops the data array's attributes. Let us add the attributes again to the data array, as we might want to use the `units` and `long_name` for title and label information. Let us use the attributes from the data array `north`.

# In[11]:


co = co.assign_attrs(north.attrs)
co


# <br>

# ## 5. Visualize the meridional mean plot of Carbon Monoxide in August 2020

# Now we can visualize the vertical Carbon Monoxide distribution over the northern hemisphere in August 2020. The pressure levels are on a non-linear scale. To ensure equal spacing between the values, let us create a regular scale.

# In[12]:


y = np.arange(co.level.shape[0])+1
y = y[::-1]


# <br>

# The visualization code can be divided in five main parts:
# * **Initiate matplotlib figure**: Initiatie a matplotlib figure object
# * **Plotting the data**: Plot the array with matplotlib's function `pcolormesh` with longitude on the x-axis and pressure levels on the y-axis
# * **Set y-axis tickmarks and labels**: Set regular-scaled y-axis ticks and labels
# * **Set axes labels and title**
# * **Specify a colorbar**: Specify a colorbar to be placed on the right and use the units attribute to add as colorbar unit

# In[13]:


# Define the figure and specify size
fig = plt.figure(figsize=(16, 8))
ax = plt.subplot()

# Plot the figure with pcolormesh
im = plt.pcolormesh(co.longitude, y, co[0,:,:], cmap='jet', shading='auto')

# Set x and y axis tickmarks, labels
ax.yaxis.set_ticks(y)
ax.yaxis.set_ticklabels(co.level.values)

# Set axes labels and title
ax.set_xlabel('\nLongitude', fontsize=14)
ax.set_ylabel('Atmospheric pressure level (millibars)\n', fontsize=14)
ax.set_title('\n'+ co.long_name + ' in Aug 2020 in Northern Hemisphere\n', fontsize=16)

# Specify a colorbar
cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
cbar.set_label('\n'+ co.units, fontsize=14)


# The plot above shows you how the concentration and longitudinal distribution of Carbon Monoxide in the northern hemisphere varies at different levels of the atmosphere and at different longitudes. Striking are the elevated Carbon Monoxide values between 100 and 110 degrees East, which is an indicator of the strong industrial activity in China.

# <hr>

# <p></p>
# <span style='float:right'><p style=\"text-align:right;\">This project is licensed under <a href="./LICENSE">APACHE License 2.0</a>. | <a href=\"https://github.com/ecmwf-projects/copernicus-training">View on GitHub</a></span>
