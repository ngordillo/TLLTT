
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd

load_dir = "/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"

filename = 'tropic_200_precip.nc'#small_2mtemp.nc'#tropic_200_z500.nc'
# filename = 'small_slp.nc'
var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values)* 86400#[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]
time = xr.open_dataset(load_dir+filename)['time'].values

#Get's the average sea level pressure on a daily basis. 

#number of days
days = 365
avgvar_day = []

#Get the number of years to calculate an average psl per day
years = 200

# print(time[0])
# print(time.indexes)

fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 200 + 98).astype('datetime64[ns]')

fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 200 + 98)), "time": fake_dates})
raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))



all_years = raw_time["time.year"].values

all_months = raw_time["time.month"].values

months = np.unique(all_months)

# print(all_years)

all_days = raw_time["time.day"].values

days = np.unique(all_days)


# feb = np.where(all_months == 2)

# twenty = np.where(all_days == 20)

# all_feb = np.intersect1d(feb, twenty)

# all_feb = np.float64(var[all_feb, 40, 0])
# print(all_feb)
# avg_feb = np.sum(all_feb)/all_feb.shape[0]

# print(avg_feb)

# print(np.mean(all_feb - avg_feb))

# print(type(all_feb[0]))

szn_cycle = []

for month in months:
    spec_month = np.where(all_months == month)
    for day in days:
        spec_day = np.where(all_days == day)
        all_vars = np.intersect1d(spec_month, spec_day)
        if(all_vars.size != 0):
            avg_day = np.mean(var[all_vars,: ,:], axis=0)
            szn_cycle.append(avg_day)
szn_cycle = np.asarray(szn_cycle)

df = xr.DataArray(szn_cycle, coords=[('day', np.arange(0, szn_cycle.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200precipcycle')

df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tropic_200year_precip_cycle.nc')


feb = np.where(all_months == 2)

twenty = np.where(all_days == 20)

all_feb = np.intersect1d(feb, twenty)

all_feb = var[all_feb, 40, 0]

print(np.mean(all_feb - szn_cycle[50, 40, 0]))

# direct_test = np.asarray([5849.47,   5892.7026, 5837.9277, 5837.989,  5857.9175, 5846.645,  5838.3687,
# 5869.84,   5840.329,  5845.9917, 5831.138, 5827.326,  5826.0815, 5850.15,
#  5845.161,  5834.413,  5853.635,  5847.1084, 5850.4043, 5873.24,   5870.1387,
#  5835.322,  5874.607,  5849.6113, 5843.1055, 5844.021,  5876.813,  5837.2896,
#  5842.0073, 5874.151,  5857.2754, 5834.229,  5866.341,  5848.0103, 5817.925,
#  5839.0557, 5832.8457, 5831.19,   5860.7373, 5853.0957, 5839.574,  5821.21,
#  5829.719,  5845.0503, 5862.2754, 5838.434,  5857.168,  5844.7866, 5824.6797,
#  5835.445,  5875.4395, 5857.264,  5879.661,  5841.0117, 5846.582,  5857.9004,
#  5837.885, 5844.8857, 5849.711,  5878.065,  5837.8325, 5847.6646, 5871.889,
#  5834.3174, 5863.5054, 5840.46,   5847.3843, 5856.4653, 5845.03,   5856.5156,
#  5865.832,  5862.9453, 5851.1025, 5876.953,  5826.9814, 5859.475,  5848.251,
#  5849.72,   5878.489, 5849.789,  5826.683,  5831.3335, 5843.3047, 5847.618,
#  5843.1943, 5861.5107, 5846.9795, 5862.712,  5852.1714, 5861.248,  5819.6465,
#  5856.886,  5854.225,  5825.787,  5854.446,  5871.087,  5846.8965, 5831.574,
#  5859.1875, 5852.4146, 5817.0767, 5845.8413, 5824.384,  5850.778,  5859.3794,
#  5849.7427, 5868.246,  5866.537,  5835.053,  5826.845,  5848.7246, 5834.52,
#  5844.104,  5858.9526, 5854.7803, 5860.8394, 5826.1284, 5849.292,  5867.1724,
#  5834.7554, 5820.5425, 5832.26,   5834.214,  5818.196,  5866.4175, 5870.8843,
#  5836.082,  5823.558,  5859.4688, 5838.006,  5850.913,  5855.085,  5846.991,
#  5867.7383, 5858.864,  5838.032,  5852.997, 5839.549,  5863.223,  5847.1606,
#  5892.999,  5818.859,  5829.4287, 5848.342,  5848.6895, 5865.482,  5864.9233,
#  5831.222,  5839.082,  5854.358,  5856.6543, 5841.698,  5840.447,  5827.1587,
#  5864.8174, 5861.1743, 5857.0396, 5874.8315, 5856.4175, 5853.8022, 5877.614,
#  5831.5605, 5845.654,  5827.6055, 5834.558,  5849.6445, 5863.6426, 5861.5796,
#  5848.204,  5877.313,  5851.5503, 5830.7476, 5845.1406, 5836.7266, 5845.961,
#  5835.186,  5870.6025, 5881.0405, 5845.6084, 5835.3687, 5839.9795, 5839.0767,
#  5866.26,   5873.951,  5855.5166, 5848.335,  5860.6777, 5868.3525, 5845.686,
#  5840.1675, 5867.1904, 5863.5425, 5855.471,  5881.821,  5860.6797, 5877.6377,
#  5844.3984, 5843.964,  5849.4014, 5839.329])

# print(type(direct_test[0]))


# avg_direct = np.mean(direct_test)

# print(avg_direct)

# print(np.mean(direct_test - avg_direct))

# avg_direct = np.sum(direct_test)/direct_test.shape[0]

# print(avg_direct)

# print(np.mean(direct_test - avg_direct))


# print(raw_time)

# datetimeindex = xr.open_dataset(load_dir+filename).indexes['time'].to_datetimeindex()
# print(datetimeindex)
# test_add = 04
# #Loop by day first.


# for i in np.arange(0,365,1):
#     var_day = 0

#     #Then loop through the number of specified years. 
#     for j in np.arange(0,years,1):

#         #Add up the sea level pressure for that particular day from every year.
#         if j == 0:
#             var_day = var[int(i+(days*j)), :,:]
#         else:
#             var_day = np.add(var_day, var[int(i+(days*j)), :,:]) 
#         # print(int(i+(days*j)))
#         if i == 200:
#             print("test_add: " + str(int(i+(days*j))))
#             test_add += var[int(i+(days*j)), 46,0]

#     #Divide it by the number of years.
#     var_day /= years
#     avgvar_day.append(var_day)

#     # if i == 200:
#     #     print("bonus test: " + str(var_day[46,0]))

# test_add = test_add / 200


# avgvar_day = np.asarray(avgvar_day)
# print(avgvar_day.shape)
# df = xr.DataArray(avgvar_day, coords=[('day', np.arange(0, avgvar_day.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200z500cycle')

# df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tropic_200year_z500_cycle.nc')

# var_c = []
# for i in np.arange(0,years,1):
#     # print("var_c: " + str((i*365)+200))
#     var_c.append(var[(i*365)+200,46,0])

# # var_c = np.asarray(var_c)
# # print(np.mean(var_c))
# print(test_add - avgvar_day[200,46,0])
# var_c = var_c - avgvar_day[200,46,0]
# # print(var_c.shape)
# print("Mean: " + str(np.mean(var_c)))
# print(avgvar_day[199,46,0])
# print(avgvar_day[201,46,0])

# pres_c = []

# for i in np.arange(0,years*365,1):
#     pres_c.append(var[i,:,:] - avgvar_day[i%365,:,:])

# pres_c = np.asarray(pres_c)

# print("Mean: " + str(np.mean(pres_c[:70*365])))

