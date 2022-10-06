
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"

# filename = '500_2mtemp.nc'
# temp_500     = xr.open_dataset(load_dir+filename)['tas'].values[:,96:,80:241]
#filename = 'NA_2mtemp.nc'
filename = 'small_2mtemp.nc'
temp     = xr.open_dataset(load_dir+filename)['tas'].values#[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

test_data = 200

#Get's the average sea level pressure on a daily basis. 

#number of days
days = 365
avgtemp_day = []

#Get the number of years to calculate an average psl per day
years = (temp.shape[0]/days)-test_data

years = 200


# #Loop by day first.
# for i in np.arange(0,365,1):
#     temp_day = 0

#     #Then loop through the number of specified years. 
#     for j in np.arange(0,years,1):

#         #Add up the sea level pressure for that particular day from every year.
#         temp_day += temp[int(i+(days*j)), :,:]

#     #Divide it by the number of years.
#     temp_day /= years
    # avgtemp_day.append(temp_day)

avgtemp_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/mjo_200year_temp_cycle.nc")['200tempcycle'].values

train_years = 200

full_loc_temp = []
#Loop by day first.
for i in np.arange(0,(train_years*365)+50,1):
    # temp_loc = []
    
    # #Then loop through the number of specified years. 
    # for j in np.arange(0,train_years,1):

    #     #Add up the sea level pressure for that particular day from every year.
    #     temp_loc.append(temp[int(i+(days*j)),46,136])

    #full_loc_temp.append(temp[i, 46,136] - avgtemp_day[i%365, 46,136]) #N
    #full_loc_temp.append(temp[i, 52,110] - avgtemp_day[i%365, 52,110])
    #full_loc_temp.append(temp[i, 39,111] - avgtemp_day[i%365, 39,111])
    full_loc_temp.append(temp[i, 64,88] - avgtemp_day[i%365, 64,88]) 

print("Mean: " + str(np.mean(full_loc_temp)))
full_loc_temp = running_mean(full_loc_temp, 7)

    
# full_loc_temp = np.asarray(full_loc_temp).flatten()

train_years = 70
test_years = 200

train_class = [] 

lead = 14

# true_temps = []

lower = np.percentile(full_loc_temp[:years*365], 33.33)

upper = np.percentile(full_loc_temp[:years*365], 66.66)

for i in np.arange(0, train_years*365, 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)

    true_slp = full_loc_temp[i+lead]

    # print("true_slp: " + str(true_slp))
    if((true_slp <= lower)):
        # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
        train_class.append(0)
    elif((true_slp >= upper)):
        # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
        train_class.append(2)
    elif(((true_slp < upper) and (true_slp > lower))):
        train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

for i in np.arange(train_years*365, test_years*365, 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)

    true_slp = full_loc_temp[i+lead]

    # print("true_slp: " + str(true_slp))
    if((true_slp <= lower)):
        # print("33.33: " + str(np.percentile(full_loc_temp, 33.33)))
        train_class.append(0)
    elif((true_slp >= upper)):
        # print("66.66: " + str(np.percentile(full_loc_temp, 66.66)))
        train_class.append(2)
    elif(((true_slp < upper) and (true_slp > lower))):
        train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/alas_tempclass_200years_fourteendays.txt', train_class, fmt='%d')

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/temp_200years_threedays_test.txt', true_temps)

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