import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# load_dir = "/ourdisk/hpc/ai2es/nicojg/TLLTT/data/"
load_dir = "/barnes-scratch/nicojg/"

# filename = '500_2mtemp.nc'
# temp_500     = xr.open_dataset(load_dir+filename)['tas'].values[:,96:,80:241]
#filename = 'NA_2mtemp.nc'
filename = 'MAIN_era5_2mtemp_mjo_notrend_anoms.nc'#era5_daily_2mtemp.nc'
temp     = xr.open_dataset(load_dir+filename)['t2m']#[:,96:,80:241]
# time     = xr.open_dataset(load_dir+filename)['time'][0:(60*365)+14]
lat  = xr.open_dataset(load_dir+filename)['lat'].values
lon   = xr.open_dataset(load_dir+filename)['lon'].values
# print(time)
# print(lat)
# print(lon)



#Find Alaska point

alaska_lat = find_nearest_index(lat, 61.2176)
alaska_lon = find_nearest_index(lon, 360-149.8997)

print(alaska_lat)
print(alaska_lon)

lead = 14
# print(temp.shift(time=lead))
# rolled_temp = temp.rolling(time = 5).mean().dropna("time", how='all')
rolled_temp = temp.rolling(time = 5).mean()
rolled_temp = rolled_temp.shift(time=-4).dropna("time", how='all')


shift_rolled_temp = rolled_temp.sel(time=slice("1960-01-15", "2021-01-14"))

print(shift_rolled_temp)

loc_shift_rolled_temp = shift_rolled_temp[:,alaska_lat, alaska_lon]
print(loc_shift_rolled_temp)
# for i in np.arange(0,(years*365)+14,1): #18 or 14
#     # temp_loc = []
    
#     # #Then loop through the number of specified years. 
#     # for j in np.arange(0,train_years,1):

#     #     #Add up the sea level pressure for that particular day from every year.
#     #     temp_loc.append(temp[int(i+(days*j)),46,136])

#     #full_loc_temp.append(temp[i, 46,136] - avgtemp_day[i%365, 46,136]) #Madison
#     # full_loc_temp.append(temp[i, 52,110] - avgtemp_day[i%365, 52,110]) #Vancouver
#     # full_loc_temp.append(temp[i, 36,113] - avgtemp_day[i%365, 36,113]) #Los Angeles
#     # full_loc_temp.append(temp[i, loc_lat,loc_lon] - avgtemp_day[i%365, loc_lat,loc_lon])
#     full_loc_temp.append(temp[i, alaska_lat,alaska_lon])  #Anchorage
#     # full_loc_temp.append(temp[i, 44,109] - avgtemp_day[i%365, 44,109])  #Crescent City


#     # full_loc_temp[-1] = full_loc_temp[-1] - temp_200_years[i, 64, 88] #Remove past two year mean

# print("Mean: " + str(np.mean(full_loc_temp)))
# print(len(full_loc_temp))

# #Only if wasnt done before!!!!
# full_loc_temp = running_mean(full_loc_temp, 5)


# df = xr.DataArray(full_loc_temp, coords=[('day', np.arange(0, full_loc_temp.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200years_2mtemp_nocycle_5back')

# df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/small_2mtemp_nocycle_5back.nc')
# exit()

year_range = np.arange(1960, 2021, 1)

date_range = []
for year in year_range:
    if year == 1960:
        date_range.append(pd.date_range(start= f'{year}-01-15',end = f'{year}-03-14',freq='d') + pd.offsets.Hour(11))
        date_range.append(pd.date_range(start= f'{year}-11-15',end = f'{year+1}-03-14',freq='d') + pd.offsets.Hour(11))
    elif year == 2020 :
        date_range.append(pd.date_range(start= f'{year}-11-15',end = f'{year+1}-01-14',freq='d')+ pd.offsets.Hour(11))
    else: 
        date_range.append(pd.date_range(start= f'{year}-11-15',end = f'{year+1}-03-14',freq='d') + pd.offsets.Hour(11))
        
date_range = [item for sublist in date_range for item in sublist]
# print(date_range)
# print(loc_shift_rolled_temp)
# print(loc_shift_rolled_temp.sel(time=date_range))

loc_shift_rolled_temp = loc_shift_rolled_temp.sel(time=date_range)

print(loc_shift_rolled_temp)

lower = np.percentile(loc_shift_rolled_temp, 33.33)
print("lower: " + str(lower))
# upper = np.percentile(full_loc_temp_winter[14:full_loc_temp_winter.shape[0] - 32], 66.66)
upper = np.percentile(loc_shift_rolled_temp, 66.66)
print("upper: " + str(upper))


train_dates = loc_shift_rolled_temp.sel(time=slice("1960-01-15", "2012-01-14"))
print("train_dates: " + str(train_dates.shape[0]))

dum_test = loc_shift_rolled_temp.sel(time=slice("1960-01-15", "1962-01-14"))
print("dum_test: " + str(dum_test.shape[0]))

train_class = []

for i in np.arange(0, train_dates.shape[0], 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)
    true_slp = loc_shift_rolled_temp[i]

    # print("true_slp: " + str(true_slp))
    if((true_slp < lower)):
        train_class.append(0)
    elif((true_slp > upper)):
        train_class.append(2)
    else:
        train_class.append(1)

    # if((true_slp < mid)):
    #     train_class.append(0)
    # else:
        #     train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

for i in np.arange(train_dates.shape[0], loc_shift_rolled_temp.shape[0], 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)
    true_slp = loc_shift_rolled_temp[i]

    # print(true_slp)
    # print(full_loc_temp_useful_days[i])

    # print("true_slp: " + str(true_slp))
    if((true_slp < lower)):
        train_class.append(0)
    elif((true_slp > upper)):
        train_class.append(2)
    else:
        train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

print(len(train_class))

np.savetxt('/barnes-scratch/nicojg/ERA5_winter_ternary_alaska_points_5days.txt', train_class, fmt='%d')

y_predict_class_plot = np.asarray(train_class)

y_predict_class_plot_low = y_predict_class_plot.astype('float64')
y_predict_class_plot_avg = y_predict_class_plot.astype('float64')
y_predict_class_plot_high = y_predict_class_plot.astype('float64')

y_predict_class_plot_low[np.where(y_predict_class_plot_low==2)[0]] = np.nan
y_predict_class_plot_low[np.where(y_predict_class_plot_low==1)[0]] = np.nan
y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==2)[0]] = np.nan
y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==0)[0]] = np.nan

y_predict_class_plot_high[np.where(y_predict_class_plot_high==1)[0]] = np.nan
y_predict_class_plot_high[np.where(y_predict_class_plot_high==0)[0]] = np.nan


plt.figure(figsize=(20,6))
plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
plt.yticks([0,1,2])
plt.title("Label Class", fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Class", fontsize=15)
plt.savefig(("../Figures" + "/timeseries/" + 'ERA5_WINTER_FULL_timeseries_mjo.png'), bbox_inches='tight')

for half_decade in np.arange(5,65,5):
    plt.figure(figsize=(20,6))
    plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
    plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
    plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
    plt.yticks([0,1,2])
    plt.title("Label Class by half decades", fontsize=20)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Class", fontsize=15)
    plt.xlim(half_decade-5,half_decade)
    plt.savefig(("../Figures" + "/timeseries/" + 'ERA5_WINTER_halfdecade_timeseries_' + str(half_decade)+ '_mjo.png'), bbox_inches='tight')
    # plt.show()

for decade in np.arange(10,70,10):
    plt.figure(figsize=(20,6))
    plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
    plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
    plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
    plt.yticks([0,1,2])
    plt.title("Label Class by decades", fontsize=20)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Class", fontsize=15)
    plt.xlim(decade-10,decade)
    plt.savefig(("../Figures" + "/timeseries/" + 'ERA5_WINTER_decade_timeseries_' + str(decade)+ '_mjo.png'), bbox_inches='tight')
    # plt.show()

quit()

all_months = time["time.month"].values

all_days = time["time.day"].values

full_loc_temp_winter = []

full_loc_temp_useful_days = []

for i in np.arange(0, len(full_loc_temp), 1):
    month = all_months[i]
    if (month == 1 or month == 2 or month == 12):
        full_loc_temp_winter.append(full_loc_temp[i])
        full_loc_temp_useful_days.append(full_loc_temp[i])
    elif(month == 11):
        if(all_days[i] in np.arange(1,15,1)):
            full_loc_temp_winter.append(full_loc_temp[i])
        else:
            full_loc_temp_useful_days.append(full_loc_temp[i])
            full_loc_temp_winter.append(full_loc_temp[i])
    elif(month == 3):
        if(all_days[i] in np.arange(1,15,1)):
            full_loc_temp_useful_days.append(full_loc_temp[i])
            full_loc_temp_winter.append(full_loc_temp[i])


full_loc_temp_winter = np.asarray(full_loc_temp_winter)

print(full_loc_temp_winter.shape)

full_loc_temp_useful_days = np.asarray(full_loc_temp_useful_days[14:len(full_loc_temp_useful_days)])

print("useful days: " + str(full_loc_temp_useful_days.shape))

train_years = 52
test_years = 60

train_class = [] 

lead = 14

# true_temps = []

# lower = np.percentile(full_loc_temp_winter[14:full_loc_temp_winter.shape[0] - 32], 33.33)
lower = np.percentile(full_loc_temp_useful_days, 33.33)
print("lower: " + str(lower))
# upper = np.percentile(full_loc_temp_winter[14:full_loc_temp_winter.shape[0] - 32], 66.66)
upper = np.percentile(full_loc_temp_useful_days, 66.66)
print("upper: " + str(upper))

# mid = np.percentile(full_loc_temp_winter[14:full_loc_temp_winter.shape[0] - 32], 50.)
mid = np.percentile(full_loc_temp_useful_days, 50.)

print(full_loc_temp_winter.shape[0] - 46)

for i in np.arange(0, 5400, 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)
    true_slp = full_loc_temp_useful_days[i]

    # print("true_slp: " + str(true_slp))
    if((true_slp < lower)):
        train_class.append(0)
    elif((true_slp > upper)):
        train_class.append(2)
    else:
        train_class.append(1)

    # if((true_slp < mid)):
    #     train_class.append(0)
    # else:
        #     train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

for i in np.arange(5400, full_loc_temp_useful_days.shape[0], 1):

    #CHANGE

    # correct_day = (i+lead)%365
    # true_slp = temp[i+lead,46,136] - avgtemp_day[correct_day,46,136]

    # true_temps.append(true_slp)
    true_slp = full_loc_temp_useful_days[i]

    # print(true_slp)
    # print(full_loc_temp_useful_days[i])

    # print("true_slp: " + str(true_slp))
    if((true_slp < lower)):
        train_class.append(0)
    elif((true_slp > upper)):
        train_class.append(2)
    else:
        train_class.append(1)

    # if((true_slp < mid)):
    #     train_class.append(0)
    # else:
    #     train_class.append(1)

count_arr = np.bincount(train_class)

print("number of 0: " + str(count_arr[0]))
print("number of 1: " + str(count_arr[1]))
print("number of 2: " + str(count_arr[2]))

    # np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/loc_'+str(loc_lon)+'_'+str(loc_lat)+'_5mean_14days.txt', train_class, fmt='%d')

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/temp_200years_threedays_test.txt', true_temps)


# np.savetxt('/ourdisk/hpc/ai2es/nicojg/TLLTT/data/ERA5_winter_ternary_alaska_points.txt', train_class, fmt='%d')
print(len(train_class))

np.savetxt('/barnes-scratch/nicojg/ERA5_winter_ternary_alaska_points_7days.txt', train_class, fmt='%d')

print(temp[0,alaska_lat, alaska_lon])
# y_predict_class_plot = np.asarray(train_class)

# y_predict_class_plot_low = y_predict_class_plot.astype('float64')
# y_predict_class_plot_avg = y_predict_class_plot.astype('float64')
# y_predict_class_plot_high = y_predict_class_plot.astype('float64')

# y_predict_class_plot_low[np.where(y_predict_class_plot_low==2)[0]] = np.nan
# y_predict_class_plot_low[np.where(y_predict_class_plot_low==1)[0]] = np.nan

# y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==2)[0]] = np.nan
# y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==0)[0]] = np.nan

# y_predict_class_plot_high[np.where(y_predict_class_plot_high==1)[0]] = np.nan
# y_predict_class_plot_high[np.where(y_predict_class_plot_high==0)[0]] = np.nan

# for half_decade in np.arange(5,30,5):
#     plt.figure(figsize=(20,6))
#     plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
#     plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
#     plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
#     plt.yticks([0,1,2])
#     plt.title("Model Class by Year"  + str(5), fontsize=20)
#     plt.xlabel("Year", fontsize=15)
#     plt.ylabel("Class", fontsize=15)
#     plt.xlim(half_decade-5,half_decade)
#     plt.savefig(("./figures" + "/timeseries/" + 'ERA5_WINTER_decade_timeseries_' + str(half_decade)+ '_mjo.png'), bbox_inches='tight')
#     # plt.show()




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