#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
# %%
def cut_longitudes(data,lon):
    print("lon: ")
    print(lon)
    ilon = np.where(np.logical_and(lon<=270,lon>=10))[0] #240
    print("ilon: ")
    print(ilon)
#     ilon = np.where((lon.values<=360) & (lon.values>=0))[0]
    lon = lon[ilon]
    data = data[:,:,ilon]    
    return data, lon
# %%
# make labels
load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"
filename = 'WH04_RMM.nc'
time     = xr.open_dataset(load_dir+filename)['RMM_amp']['time']    
rmm_amp  = xr.open_dataset(load_dir+filename)['RMM_amp']
rmm_ph   = xr.open_dataset(load_dir+filename)['RMM_ph']
print(rmm_ph.shape)

i = np.where(rmm_amp<.5) # RMM amplitudes less than 0.5 are labeled as "phase 0"
rmm_ph[i] = 0
print("time start")
print(time)
print("time end")
# print(np.unique(rmm_ph))    

# get the fields
filename = 'olr.20NS.noac_120mean_norm.nc'
lat      = xr.open_dataset(load_dir+filename)['olr']['latitude']
lon      = xr.open_dataset(load_dir+filename)['olr']['longitude']
olr      = xr.open_dataset(load_dir+filename)['olr'].values[:,:,:,np.newaxis]

filename = 'u200.20NS.noac_120mean_norm.nc'
u200     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis] 

filename = 'u850.20NS.noac_120mean_norm.nc'
u850     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis]
print("before olr shape:")
print(olr.shape)

olr, __   = cut_longitudes(olr,lon)
print("after olr shape:")
print(olr.shape)
u200, __  = cut_longitudes(u200,lon)
u850, lon = cut_longitudes(u850,lon)
data = np.concatenate((olr,u200,u850),axis=3)

# rmm_ph, data, lat, lon, time
# %%
# make labels
load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"
filename = 'small_slp.nc'
pres     = xr.open_dataset(load_dir+filename)['psl'].values[:,96:,80:241]
filename = 'small_2mtemp.nc'
temp     = xr.open_dataset(load_dir+filename)['tas'].values[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values[80:241]
print("lats: ")
print(lat)
# %%
test_data = 70

#Get's the average sea level pressure on a daily basis. 

#number of days
days = 365
avgtemp_day = []

#Get the number of years to calculate an average psl per day
years = (temp.shape[0]/days)-test_data

#Loop by day first.
for i in np.arange(0,365,1):
    temp_day = 0

    #Then loop through the number of specified years. 
    for j in np.arange(0,years,1):

        #Add up the sea level pressure for that particular day from every year.
        temp_day += temp[int(i+(days*j)), :,:]

    #Divide it by the number of years.
    temp_day /= years
    avgtemp_day.append(temp_day)

avgtemp_day = np.asarray(avgtemp_day)any
# %%
train_years = 70

full_loc_temp = []
#Loop by day first.
for i in np.arange(0,365,1):
    temp_loc = []
    
    #Then loop through the number of specified years. 
    for j in np.arange(0,train_years,1):

        #Add up the sea level pressure for that particular day from every year.
        temp_loc.append(temp[int(i+(days*j)),46,136])

    temp_loc -= avgtemp_day[i,46,136]
    full_loc_temp.append(temp_loc)


full_loc_temp = np.asarray(full_loc_temp).flatten()
# %%
true_class = []
for i in np.arange(0, train_years*365, 1):
    correct_day = (i+14)%365
    true_slp = temp[i,46,136] - avgtemp_day[correct_day,46,136]

    # print("true_slp: " + str(true_slp))
    if((true_slp <= np.percentile(full_loc_temp, 33.33))):
        # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
        true_class.append(0)
    elif((true_slp >= np.percentile(full_loc_temp, 66.66))):
        # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
        true_class.append(2)
    elif(((true_slp < np.percentile(full_loc_temp, 66.66)) and (true_slp > np.percentile(full_loc_temp, 33.33)))):
        true_class.append(1)
# %%
print(true_class)
# %%
np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_70year.txt', true_class, fmt='%d')

# %%
test_every = np.loadtxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_70year.txt')
print(test_every.shape)
# %%
print(true_class.shape)
# %%
load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"
filename = 'small_slp.nc'
time     = xr.open_dataset(load_dir+filename)['time'].values
# %%
print(type(time))
# %%
temp_label = np.loadtxt("Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_70year.txt")
# %%
