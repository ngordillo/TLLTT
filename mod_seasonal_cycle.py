# import py3nvml
# py3nvml.grab_gpus(num_gpus=1, gpu_select=[2])


from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd
import cartopy.crs as ccrs
import cmasher as cmr

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.zeros(x.shape[2]), axis = 0), axis =0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


load_dir = "~/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/"
#load_dir = "/ourdisk/hpc/ai2es/nicojg/TLLTT/data/"

filename =  'mjo_precip_local.nc'#'mjo_200_precip.nc'#small_2mtemp.nc'#tropic_200_z500.nc'

var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values)*86400#)[:,96:,80:241]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]
time = xr.open_dataset(load_dir+filename)['time'].values

#Get's the average sea level pressure on a daily basis. 

#number of days
num_days = 365

#Get the number of years to calculate an average psl per day
years = 15

#Create fake date and time data in order to use np.where()
# fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 200 + 98).astype('datetime64[ns]')

# fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 200 + 98)), "time": fake_dates})

fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * years + 3).astype('datetime64[ns]')

fake_time = xr.Dataset({"foo": ("time", np.arange(365 * years + 3)), "time": fake_dates})

#Take out the leap days!!
raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))

print(raw_time)

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
df = xr.DataArray(szn_cycle, coords=[('day', np.arange(0, szn_cycle.shape[0], 1)), ('lat', lat), ('lon', lon)], name='15precipcycle')

df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/mjo_15year_precip_cycle.nc')

#df.to_netcdf('/ourdisk/hpc/ai2es/nicojg/TLLTT/data/mjo_15year_precip_cycle.nc')

print("Starting Removal of Seasonal cycle code #######################################################################################################################################################################################################")

filename = 'mjo_4back_precip_local.nc'
var_rem     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values * 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
# time     = xr.open_dataset(load_dir+filename)['time'].values#[:train_years*365]
lat  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
lon   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

var_c = []

full_years = 15


#Make a smarter way of doing this eventually
for i in np.arange(0,(full_years*365)+4,1):
    #   REMOVE -4 AS NEEDED
    #var_c.append(var[i,:,:,:] - avgvar_day[(i-4)%365,:,:,:])
    var_c.append(var_rem[i,:,:] - szn_cycle[(i-4)%365,:,:])

var_c = np.asarray(var_c)
print("WOOOSAVVEEEEE")
print(var_c.shape)
var_c_fw = running_mean(var_c, 5)
print("WOOOSAVOKKKKKVEEEEE")
print(var_c_fw.shape)
var_c_fw = np.asarray(var_c_fw)

print("RUNNING MEAN ARRAY SIZE: "  + str(var_c_fw.shape))

var_200years = var_c[4:, :, :]

var_checkcycle = []

var_checkloc = []

for i in np.arange(0,full_years,1):
    var_checkcycle.append(var_200years[150+(365*i),:,:])
    var_checkloc.append(var_200years[150+(365*i),40,40])

var_checkcycle = np.asarray(var_checkcycle)

var_checkcycle = np.mean(var_checkcycle, axis=0)

var_checkloc = np.asarray(var_checkloc)

var_checkloc = np.mean(var_checkloc)

fig = plt.figure(figsize=(20, 16))
fig.tight_layout()

spec = fig.add_gridspec(4, 5)

plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

plt.set_cmap('cmr.copper')
img = sub1.contourf(np.asarray(lon), np.asarray(lat), np.asarray(var_checkcycle), np.linspace(-4, 4, 41), transform=ccrs.PlateCarree())
# plt.xticks(np.arange(-180,181,30), np.concatenate((np.arange(0,181,30),np.arange(-160,1,30)), axis = None))
# sub1.set_xticks(np.arange(-180,181,30), np.arange(-180,181,30))
sub1.set_xticks(np.arange(-180,181,30))
sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
sub1.set_yticks(np.arange(-90,91,15))
sub1.set_xlim(-140,120)
sub1.set_ylim(-30,30)
sub1.set_xlabel("Longitude (degrees)",fontsize=25)
sub1.set_ylabel("Latitude (degrees)",fontsize=25)
cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
cbar.set_label("mm/day", fontsize=25)

sub1.coastlines()
#plt.savefig(('/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/new_senscycle_test.png'), bbox_inches='tight', dpi=400)
plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/local_szncycle_examine.png'), bbox_inches='tight', dpi=400)
# plt.show()

loc_pres = []
for i in range(var_c_fw.shape[0]):
    loc_pres.append(var_c_fw[i,50,40])

loc_pres = np.asarray(loc_pres)

plt.figure(figsize=(20,10))
plt.title("Tropics Precip - Seasonal Cycle removed", fontsize=40)
plt.xlabel("Years", fontsize=30)
plt.xlim(0,10)
plt.ylabel("Precip Anamoly", fontsize=30)
plt.plot(np.arange(0, len(loc_pres),1)/365, loc_pres)
# plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'data_examine.png'), bbox_inches='tight', dpi=400)
loc_pres = []

print(int(var_c_fw.shape[0]/365))
for i in range(int(var_c_fw.shape[0]/365)):
    loc_pres.append(var_c_fw[(i*365)+100,50,40])

loc_pres = np.asarray(loc_pres)
print("Mean: " + str(np.mean(var_checkloc)))
# plt.show()


print("another test")
print(var_c_fw.shape)
df = xr.DataArray(var_c_fw, coords=[('day', np.arange(0, var_c_fw.shape[0], 1)), ('lat', lat), ('lon', lon)], name='200precip5back')

df.to_netcdf('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/mjo_15year_precip_nocycle_5back.nc')

#df.to_netcdf('/ourdisk/hpc/ai2es/nicojg/TLLTT/data/mjo_200year_precip_nocycle_5back.nc')


#Debugging code to test. Can be ignored.
# feb = np.where(all_months == 2)

# twenty = np.where(all_days == 20)

# all_feb = np.intersect1d(feb, twenty)

# all_feb = var[all_feb, 40, 0]

# print(np.mean(all_feb - szn_cycle[50, 40, 0]))
