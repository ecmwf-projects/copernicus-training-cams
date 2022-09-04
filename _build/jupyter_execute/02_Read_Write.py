#!/usr/bin/env python
# coding: utf-8

# ![logo](./img/LogoLine_horizon_CAMS.png)

# | | | |
# |:-:|:-:|:-:|
# |[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ecmwf-projects/copernicus-training/HEAD?urlpath=lab/tree/CAMS_atmospheric-composition.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/ecmwf-projects/copernicus-training/blob/master/CAMS_atmospheric-composition.ipynb)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf-projects/copernicus-training/blob/master/CAMS_atmospheric-composition.ipynb)|

# <br>

# # Tutorial on how to import, subset, aggregate and export CAMS Data
# 
# This tutorial provides practical examples that demonstrate how to download, read into Xarray, subset, aggregate and export data from the [Atmosphere Data Store (ADS)](https://ads.atmosphere.copernicus.eu/) of the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).

# ## Install ADS API
# 
# We will need to install the Application Programming Interface (API) of the [Atmosphere Data Store (ADS)](https://ads.atmosphere.copernicus.eu). This will allow us to programmatically download data.

# ```{note}
# Note the exclamation mark in the line of code below. This means the code will run as a shell (as opposed to a notebook) command.
# ```

# In[1]:


get_ipython().system('pip install cdsapi')


# ## Import libraries
# 
# Here we import a number of publicly available Python packages, needed for this tutorial.

# In[33]:


# CDS API
import cdsapi

# Library to extract data
from zipfile import ZipFile

# Libraries to read and process arrays
import numpy as np
import xarray as xr
import pandas as pd

# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()


# ## Access data
# 
# To access data from the ADS, you will need first to register (if you have not already done so), by visiting https://ads.atmosphere.copernicus.eu and selecting **"Login/Register"**
# 
# To obtain data programmatically from the ADS, you will need an API Key. This can be found in the page https://ads.atmosphere.copernicus.eu/api-how-to. Here your key will appear automatically in the black window, assuming you have already registered and logged into the ADS. Your API key is the entire string of characters that appears after `key:`
# 
# Now copy your API key into the code cell below, replacing `#######` with your key.

# In[2]:


URL = 'https://ads.atmosphere.copernicus.eu/api/v2'

# Replace the hashtags with your key:
KEY = '###################################'


# Here we specify a data directory into which we will download our data and all output files that we will generate:

# In[3]:


DATADIR = './'


# The data we will download and inspect in this tutorial comes from the CAMS Global Atmospheric Composition Forecast dataset. This can be found in the [Atmosphere Data Store (ADS)](https://ads.atmosphere.copernicus.eu) by scrolling through the datasets, or applying search filters as illustrated here:
# 
# ![logo](./img/ADS_search_and_result.png)

# Having selected the correct dataset, we now need to specify what product type, variables, temporal and geographic coverage we are interested in. These can all be selected in the **"Download data"** tab. In this tab a form appears in which we will select the following parameters to download:
# 
# - Variables (Single level): *Dust aerosol optical depth at 550nm*, *Organic matter aerosol optical depth at 550nm*, *Total aerosol optical depth at 550nm*
# - Date: Start: *2021-08-01*, End: *2021-08-08*
# - Time: *00:00*, *12:00* (default)
# - Leadtime hour: *0* (only analysis)
# - Type: *Forecast* (default)
# - Area: Restricted area: *North: 90*, *East: 180*, *South: 0*, *West: -180* 
# - Format: *Zipped netCDF (experimental)*
# 
# At the end of the download form, select **"Show API request"**. This will reveal a block of code, which you can simply copy and paste into a cell of your Jupyter Notebook (see cell below)...

# ```{note}
# Before running this code, ensure that you have **accepted the terms and conditions**. This is something you only need to do once for each CAMS dataset. You will find the option to do this by selecting the dataset in the ADS, then scrolling to the end of the *Download data* tab.
# ```

# In[5]:


c = cdsapi.Client(url=URL, key=KEY)
c.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': [
            'dust_aerosol_optical_depth_550nm', 'organic_matter_aerosol_optical_depth_550nm', 'total_aerosol_optical_depth_550nm',
        ],
        'date': '2021-08-01/2021-08-08',
        'time': [
            '00:00', '12:00',
        ],
        'leadtime_hour': '0',
        'type': 'forecast',
        'area': [
            90, -180, 0,
            180,
        ],
        'format': 'netcdf_zip',
    },
    f'{DATADIR}/2021-08_AOD.zip')


# ## Read data
# 
# Now that we have downloaded the data, we can read, plot and analyse it...
# 
# We have requested the data in NetCDF format. This is a commonly used format for gridded (array-based) scientific data. 
# 
# To read and process this data we will make use of the Xarray library. Xarray is an open source project and Python package that makes working with labelled multi-dimensional arrays simple and efficient. We will read the data from our NetCDF file into an Xarray **"dataset"**.

# First we extract the downloaded zip file:

# In[4]:


# Create a ZipFile Object and load zip file in it
with ZipFile(f'{DATADIR}/2021-08_AOD.zip', 'r') as zipObj:
   # Extract all the contents of zip file into a directory
   zipObj.extractall(path=f'{DATADIR}/2021-08_AOD/')


# For convenience, we create a variable with the name of our downloaded file:

# In[19]:


fn = f'{DATADIR}/2021-08_AOD/data.nc'


# Now we can read the data into an Xarray dataset:

# In[20]:


# Create Xarray Dataset
ds = xr.open_dataset(fn)


# Let's see how this looks by querying our newly created Xarray dataset ...

# In[21]:


ds


# We see that the dataset has three variables. Selecting the "show/hide attributes" icons reveals their names: **"omaod550"** is "Organic Matter Aerosol Optical Depth at 550nm", **"aod550"** is "Total Aerosol Optical Depth at 550nm" and **"duaod550"** is "Dust Aerosol Optical Depth at 550nm".
# The dataset also has three coordinates of **longitude**, **latitude** and **time**.
# 
# We will now look more carefully at the "Total Aerosol Optical Depth at 550nm" dataset.
# 
# While an Xarray **dataset** may contain multiple variables, an Xarray **data array** holds a single multi-dimensional variable and its coordinates. To make the processing of the **aod550** data easier, we convert in into an Xarray data array.

# In[22]:


# Create Xarray Data Array
da = ds['aod550']
da


# ## Subset data
# 
# This section provides some selected examples of ways in which parts of a dataset can be extracted. More comprehensive documentation on how to index and select data is available here: https://docs.xarray.dev/en/stable/user-guide/indexing.html.

# ### Temporal subset

# By inspecting the array, we notice that the first of the three dimensions is time. If we wish to select only one time step, the easiest way to do this is to use positional indexing. The code below creates a Data Array of only the first time step.

# In[23]:


time0 = da[0,:,:]
time0


# And this creates a Data Array of the first 5 time steps:

# In[24]:


time_5steps = da[0:5,:,:]
time_5steps


# Another way to select data is to use the `.sel()` method of xarray. The example below selects all data from the first of August.

# In[25]:


firstAug = da.sel(time='2021-08-01')


# We can also select a time range using label based indexing, with the `loc` attribute:

# In[26]:


period = da.loc["2021-08-01":"2021-08-03"]
period


# ### Geographic subset
# 
# Geographical subsetting works in much the same way as temporal subsetting, with the difference that instead of one dimension we now have two (or even three if we inlcude altitude).

# #### Select nearest grid cell
# 
# In some cases, we may want to find the geographic grid cell that is situated nearest to a particular location of interest, such as a city. In this case we can use `.sel()`, and make use of the `method` keyword argument, which enables nearest neighbor (inexact) lookups. In the example below, we look for the geographic grid cell nearest to Paris.

# In[27]:


paris_lat = 48.9
paris_lon = 2.4

paris = da.sel(latitude=paris_lat, longitude=paris_lon, method='nearest')


# In[28]:


paris


# #### Regional subset
# 
# Often we may wish to select a regional subset. Note that you can specify a region of interest in the [ADS](https://ads.atmosphere.copernicus.eu/) prior to downloading data. This is more efficient as it reduces the data volume. However, there may be cases when you wish to select a regional subset after download. One way to do this is with the `.where()` function. 
# 
# In the previous examples, we have used methods that return a subset of the original data. By default `.where()` maintains the original size of the data, with selected elements masked (which become "not a number", or `nan`). Use of the option `drop=True` clips coordinate elements that are fully masked.
# 
# The example below uses `.where()` to select a geographic subset from 30 to 60 degrees latitude. We could also specify longitudinal boundaries, by simply adding further conditions.

# In[29]:


mid_lat = da.where((da.latitude > 30.) & (da.latitude < 60.), drop=True)


# ## Aggregate data
# 
# Another common task is to aggregate data. This may include reducing hourly data to daily means, minimum, maximum, or other statistical properties. We may wish to apply over one or more dimensions, such as averaging over all latitudes and longitudes to obtain one global value.

# ### Temporal aggregation
# 
# To aggregate over one or more dimensions, we can apply one of a number of methods to the original dataset, such as `.mean()`, `.min()`, `.max()`, `.median()` and others (see https://docs.xarray.dev/en/stable/api.html#id6 for the full list). 
# 
# The example below takes the mean of all time steps. The `keep_attrs` parameter is optional. If set to `True` it will keep the original attributes of the Data Array (i.e. description of variable, units, etc). If set to false, the attributes will be stripped.

# In[30]:


time_mean = da.mean(dim="time", keep_attrs=True)
time_mean


# Instead of reducing an entire dimension to one value, we may wish to reduce the frequency within a dimension. For example, we can reduce hourly data to daily max values. One way to do this is using `groupby()` combined with the `.max()` aggregate function, as shown below:

# In[31]:


daily_max = da.groupby('time.day').mean(keep_attrs=True)
daily_max


# ### Spatial aggregation
# 
# We can apply the same principles to spatial aggregation. An important consideration when aggregating over latitude is the variation in area that the gridded data represents. To account for this, we would need to calculate the area of each grid cell. A simpler solution however, is to use the cosine of the latitude as a proxy. 
# 
# The example below demonstrates how to calculate a spatial average of total AOD, applied to the temporal mean we previously calculated, to obtain a single mean value of total AOD averaged in space and time.
# 
# We first calculate the cosine of the latitudes, having converted these from degrees to radians. We then apply these to the Data Array as weights.

# In[34]:


weights = np.cos(np.deg2rad(time_mean.latitude))
weights.name = "weights"
time_mean_weighted = time_mean.weighted(weights)


# Now we apply the aggregate function `.mean()` to obtain a weighted average.

# In[35]:


Total_AOD = time_mean_weighted.mean(["longitude", "latitude"])
Total_AOD


# ## Export data
# 
# This section includes a few examples of how to export data.

# ### Export data as NetCDF
# 
# The code below provides a simple example of how to export data to NetCDF.

# In[ ]:


paris.to_netcdf(f'{DATADIR}/2021-08_AOD_Paris.nc')


# ### Export data as CSV

# You may wish to export this data into a format which enables processing with other tools. A commonly used file format is CSV, or "Comma Separated Values", which can be used in software such as Microsoft Excel. This section explains how to export data from an xarray object into CSV. Xarray does not have a function to export directly into CSV, so instead we use the Pandas library. We will read the data into a Pandas Data Frame, then write to a CSV file using a dedicated Pandas function.

# In[19]:


df = paris.to_dataframe()


# In[20]:


df


# In[21]:


df.to_csv(f'{DATADIR}/2021-08_AOD_Paris.csv')


# ### Please see the following tutorials on how to visualise this data in maps, plots and animations!
