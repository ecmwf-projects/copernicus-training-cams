#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# | | | |
# |:-:|:-:|:-:|
# |[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training/HEAD?urlpath=lab/tree/CAMS_atmospheric-composition.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training/blob/master/CAMS_atmospheric-composition.ipynb)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf-projects/copernicus-training/blob/master/CAMS_atmospheric-composition.ipynb)|

# # Plot Time Series of CAMS Data
# 
# This tutorial demonstrates how to plot time series of data from the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/). The example focusses on CO2, and we will visualise the "Keeling Curve" of global average increases in CO2 from the last decades.

# In[1]:


get_ipython().system('pip install cdsapi')


# In[4]:


# CDS API
import cdsapi

# Libraries for reading and working with multidimensional arrays
import numpy as np
import xarray as xr

# Library for plotting and visualising data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()


# ## Data access and preprocessing

# ### Download CAMS global reanalysis data
# 
# Copy your API key into the code cell below, replacing `#######` with your key. (Remember, to access data from the ADS, you will need first to register/login https://ads.atmosphere.copernicus.eu and obtain an API key from https://ads.atmosphere.copernicus.eu/api-how-to.)

# In[5]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'

# Replace the hashtags with your key:
KEY = '##########################################'


# Here we specify a data directory into which we will download our data and all output files that we will generate:

# In[6]:


DATADIR = './'


# For this tutorial, we will use *CAMS global greenhouse gas reanalysis (EGG4)* data, monthly averaged fields version. The code below shows the subset characteristics that we will extract from this dataset as an API request.

# ```{note}
# Before running this code, ensure that you have **accepted the terms and conditions**. This is something you only need to do once for each CAMS dataset. You will find the option to do this by selecting the dataset in the ADS, then scrolling to the end of the *Download data* tab.
# ```

# In[5]:


c = cdsapi.Client(url=URL, key=KEY)
c.retrieve(
    'cams-global-ghg-reanalysis-egg4-monthly',
    {
        'variable': 'co2_column_mean_molar_fraction',
        'year': [
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'product_type': 'monthly_mean',
        'format': 'netcdf',
    },
    f'{DATADIR}/CO2_2003-2020.nc')


# ### Read and inspect data

# Read the data into an Xarray dataset:

# In[9]:


fn = f'{DATADIR}/CO2_2003-2020.nc'
ds = xr.open_dataset(fn)
ds


# ### Spatial aggregation
# 
# We would like to visualise this data not in maps, but as a one dimensional time series of global average values. To do this, we will first need to aggregate the data spatially to create a single global average at each time step.

# In order to aggregate over the latitudinal dimension, we need to take into account the variation in area as a function of latitude. We will do this using the cosine of the latitude as a proxy:

# In[10]:


weights = np.cos(np.deg2rad(ds.latitude))
weights.name = "weights"
ds_weighted = ds.weighted(weights)


# Now we can average over both the longitude and latitude dimensions:

# In[11]:


# Average (mean) over the latitudinal axis
co_ds = ds_weighted.mean(dim=["latitude", "longitude"])


# ### Create xarray Data Array from Dataset object

# Here we create an Xarray Data Array object containing the CO2 variable.

# In[12]:


co = co_ds['tcco2']


# ## Plot time series of global CO
# 
# Now we can plot the time series of globally averaged CO2 data over time.

# ### Simple plot using xarray
# 
# The easiest way to plot data in an Xarray Data Array object is to use Xarray's own plotting functionality.

# In[13]:


co.plot()


# In this plot of monthly averaged CO2 we can see the seasonal variation and yearly increase in global CO2 over the last decades.

# ### Customised plot using matplotlib
# 
# In this example we use the Matplotlib library to create the same plot as above, with a few customisations

# In[14]:


fig, ax = plt.subplots(1, 1, figsize = (9, 6)) # Set size and dimensions of figure

ax.set_title('CO2 global time series', fontsize=12) # Set figure title
ax.set_ylabel('CO2 column-mean molar fraction (ppm)') # Set Y axis title
ax.set_xlabel('Year') # Set X axis title
ax.grid(linewidth=1, color='gray', alpha=0.5, linestyle='--') # Include gridlines
ax.plot(co.time, co) # Plot the data

fig.savefig(f'{DATADIR}/CAMS_CO2_reanalysis.png') # Save the figure


# In[ ]:




