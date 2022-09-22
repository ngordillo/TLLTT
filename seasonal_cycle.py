
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio

load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"

filename = 'tropic_200_z500.nc'
# filename = 'small_slp.nc'
var     = xr.open_dataset(load_dir+filename)['zg500'].values# * 86400#[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

#Get's the average sea level pressure on a daily basis. 

#number of days
days = 365
avgvar_day = []

#Get the number of years to calculate an average psl per day
years = 200

test_add = 0
#Loop by day first.
for i in np.arange(0,365,1):
    var_day = 0

    #Then loop through the number of specified years. 
    for j in np.arange(0,years,1):

        #Add up the sea level pressure for that particular day from every year.
        if j == 0:
            var_day = var[int(i+(days*j)), :,:]
        else:
            var_day = np.add(var_day, var[int(i+(days*j)), :,:]) 
        # print(int(i+(days*j)))
        if i == 200:
            print("test_add: " + str(int(i+(days*j))))
            test_add += var[int(i+(days*j)), 46,0]

    #Divide it by the number of years.
    var_day /= years
    avgvar_day.append(var_day)

    # if i == 200:
    #     print("bonus test: " + str(var_day[46,0]))

test_add = test_add / 200


avgvar_day = np.asarray(avgvar_day)
print(avgvar_day.shape)
df = xr.DataArray(avgvar_day, coords=[('day', np.arange(0, avgvar_day.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200z500cycle')

df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tropic_200year_z500_cycle.nc')

var_c = []
for i in np.arange(0,years,1):
    # print("var_c: " + str((i*365)+200))
    var_c.append(var[(i*365)+200,46,0])

# var_c = np.asarray(var_c)
# print(np.mean(var_c))
print(test_add - avgvar_day[200,46,0])
var_c = var_c - avgvar_day[200,46,0]
# print(var_c.shape)
print("Mean: " + str(np.mean(var_c)))
# print(avgvar_day[199,46,0])
# print(avgvar_day[201,46,0])

# pres_c = []

# for i in np.arange(0,years*365,1):
#     pres_c.append(var[i,:,:] - avgvar_day[i%365,:,:])

# pres_c = np.asarray(pres_c)

# print("Mean: " + str(np.mean(pres_c[:70*365])))


# test_mean = [] 

# for i in np.arange(0,years,1):
#     test_mean.append(var_c[(i*365)+200,len(lat)-1, 136])
# test_mean = np.asarray(test_mean)
# print(test_mean.shape)
# print("Mean: " + str(np.sum(test_mean)/test_mean.shape[0]))
# # train_years = 200

# full_loc_temp = []
# #Loop by day first.
# for i in np.arange(0,365,1):
#     temp_loc = []
    
#     #Then loop through the number of specified years. 
#     for j in np.arange(0,train_years,1):

#         #Add up the sea level pressure for that particular day from every year.
#         temp_loc.append(temp[int(i+(days*j)),46,136])

#     temp_loc -= avgtemp_day[i,46,136]
#     full_loc_temp.append(temp_loc)


# full_loc_temp = np.asarray(full_loc_temp).flatten()

# train_years = 70
# test_years = 200

# train_class = [] 

# lead = 0

# true_temps = []
# for i in np.arange(0, train_years*365, 1):

#     #CHANGE

#     correct_day = (i+lead)%365
#     true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

#     true_temps.append(true_slp)

#     # print("true_slp: " + str(true_slp))
#     if((true_slp <= np.percentile(full_loc_temp, 33.33))):
#         # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
#         train_class.append(0)
#     elif((true_slp >= np.percentile(full_loc_temp, 66.66))):
#         # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
#         train_class.append(2)
#     elif(((true_slp < np.percentile(full_loc_temp, 66.66)) and (true_slp > np.percentile(full_loc_temp, 33.33)))):
#         train_class.append(1)

# count_arr = np.bincount(train_class)

# print("number of 0: " + str(count_arr[0]))
# print("number of 1: " + str(count_arr[1]))
# print("number of 2: " + str(count_arr[2]))

# for i in np.arange(train_years*365, test_years*365, 1):

#     #CHANGE

#     correct_day = (i+lead)%365
#     true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

#     true_temps.append(true_slp)


#     # print("true_slp: " + str(true_slp))
#     if((true_slp <= np.percentile(full_loc_temp, 33.33))):
#         # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
#         train_class.append(0)
#     elif((true_slp >= np.percentile(full_loc_temp, 66.66))):
#         # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
#         train_class.append(2)
#     elif(((true_slp < np.percentile(full_loc_temp, 66.66)) and (true_slp > np.percentile(full_loc_temp, 33.33)))):
#         train_class.append(1)

# count_arr = np.bincount(train_class)

# print("number of 0: " + str(count_arr[0]))
# print("number of 1: " + str(count_arr[1]))
# print("number of 2: " + str(count_arr[2]))

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_200years_zerodays.txt', train_class, fmt='%d')

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/temps_200years_zerodays.txt', true_temps, fmt='%d')

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_70years.txt', train_class, fmt='%d')

# val_class = [] 
# for i in np.arange(0, 365, 1):
#     correct_day = (i+14)%365
#     true_slp = temp_500[i+14,46,136] - avgtemp_day[correct_day,46,136]

#     # print("true_slp: " + str(true_slp))
#     if((true_slp <= np.percentile(full_loc_temp, 33.33))):
#         # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
#         val_class.append(0)
#     elif((true_slp >= np.percentile(full_loc_temp, 66.66))):
#         # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
#         val_class.append(2)
#     elif(((true_slp < np.percentile(full_loc_temp, 66.66)) and (true_slp > np.percentile(full_loc_temp, 33.33)))):
#         val_class.append(1)

# count_arr = np.bincount(val_class)

# print("number of 0: " + str(count_arr[0]))
# print("number of 1: " + str(count_arr[1]))
# print("number of 2: " + str(count_arr[2]))

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_500year.txt', val_class, fmt='%d')