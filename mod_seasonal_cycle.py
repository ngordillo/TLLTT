
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd

load_dir = "~/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"
#load_dir = "/ourdisk/hpc/ai2es/nicojg/"

filename = 'small_2mtemp.nc' #'mjo_200_precip.nc'#small_2mtemp.nc'#tropic_200_z500.nc'

var     = np.float64(xr.open_dataset(load_dir+filename)['tas'].values)#*86400)#[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]
time = xr.open_dataset(load_dir+filename)['time'].values

#Get's the average sea level pressure on a daily basis. 

#number of days
num_days = 365

#Get the number of years to calculate an average psl per day
years = 200

#Create fake date and time data in order to use np.where()
fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 200 + 98).astype('datetime64[ns]')

fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 200 + 98)), "time": fake_dates})

#Take out the leap days!!
raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))

#extract year, months, and days.
all_years = raw_time["time.year"].values

all_months = raw_time["time.month"].values

all_days = raw_time["time.day"].values

months = np.unique(all_months)

days = np.unique(all_days)

#Loop by month and then day. Then, find all the indexs where it is that specific day and take the mean.
szn_cycle = []

#Get each month
for month in months:
    spec_month = np.where(all_months == month)

    #Get each day
    for day in days:
        spec_day = np.where(all_days == day)
        #Get the indexes that are common between the month and days
        all_vars = np.intersect1d(spec_month, spec_day)

        #If for some reason there are indexes for the date, don't proceed. This is redundant now.
        if(all_vars.size != 0):

            #Remove all indexes past the 200 years of the main dataset.
            all_vars = all_vars[all_vars < years*num_days]
            #Take the mean of the maps corresponding to the indexes.
            avg_day = np.mean(var[all_vars,: ,:], axis=0)

            szn_cycle.append(avg_day)
szn_cycle = np.asarray(szn_cycle)

#Make a dataarray and then make it a netcdf to save.
df = xr.DataArray(szn_cycle, coords=[('day', np.arange(0, szn_cycle.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200tempcycle')

df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/mjo_200year_temp_cycle.nc')

#Debugging code to test. Can be ignored.
feb = np.where(all_months == 2)

twenty = np.where(all_days == 20)

all_feb = np.intersect1d(feb, twenty)

all_feb = var[all_feb, 40, 0]

print(np.mean(all_feb - szn_cycle[50, 40, 0]))
