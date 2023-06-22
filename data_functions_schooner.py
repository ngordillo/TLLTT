import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd
import cartopy.crs as ccrs
import cmasher as cmr

from imblearn.under_sampling import RandomUnderSampler

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "23 November 2021"

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.zeros(x.shape[2]), axis = 0), axis =0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)



# def get_and_process_mjo_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    
#     if(colored == True):
#         raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
#     else:
#         raw_data = np.asarray(raw_data, dtype='float')
#     raw_labels   = np.asarray(raw_labels, dtype='float')

#     if(shuffle==True):
#         # shuffle the data
#         print('shuffling the data before train/validation/test split.')
#         index      = np.arange(0,raw_data.shape[0])
#         rng.shuffle(index)  
#         raw_data    = raw_data[index,:,:,:]
#         raw_labels  = raw_labels[index,]

#     # separate the data into training, validation and testing
#     all_years = raw_time["time.year"].values
#     years     = np.unique(all_years)

#     nyr_val   = int(len(years)*.2)    
#     nyr_train = len(years) - nyr_val
    
#     years_train = np.random.choice(years,size=(nyr_train,),replace=False)
# #     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng
#     years_val   = np.setxor1d(years,years_train)
#     years_test  = 2010
    
#     ic(years_train)
#     ic(years_val)
#     ic(years_test)    
    
#     years_train = np.isin(all_years, years_train)
#     years_val   = np.isin(all_years, years_val)
#     years_test  = np.isin(all_years, years_test)
    
#     iyears_train = np.where(years_train==True)[0]
#     iyears_val = np.where(years_val==True)[0]
#     iyears_test = np.where(years_test==True)[0]   
        
#     # Standardize the input based on training data only
#     X_train_raw = raw_data[iyears_train,:,:,:]
#     if( (standardize==True) or (standardize=='all')):
#         X_mean  = np.mean(X_train_raw.flatten())
#         X_std   = np.std(X_train_raw.flatten())
#     elif(standardize=='pixel'):
#         X_mean  = np.mean(X_train_raw,axis=(0,))
#         X_std   = np.std(X_train_raw,axis=(0,))
#         X_std[X_std==0] = 1.
#     else:
#         X_mean  = 0. 
#         X_std   = 1. 
    
#     # Create the target vectors, which includes a second dummy column.
#     X_train = (raw_data[iyears_train,:,:,:] - X_mean) / X_std
#     X_val   = (raw_data[iyears_val,:,:,:] - X_mean) / X_std
#     X_test  = (raw_data[iyears_test,:,:,:] - X_mean) / X_std

#     X_train[X_train==0.] = 0.
#     X_val[X_val==0.]     = 0.
#     X_test[X_test==0.]   = 0.    
    
#     y_train = raw_labels[iyears_train]
#     y_val   = raw_labels[iyears_val]
#     y_test  = raw_labels[iyears_test]

#     time_train = raw_time[iyears_train]
#     time_val   = raw_time[iyears_val]
#     time_test  = raw_time[iyears_test]
    
    
#     print(f"raw_data        = {np.shape(raw_data)}")
#     print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
#     print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
#     print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
#     if(standardize != 'pixel'):
#         print(f"X_mean          = {X_mean}")
#         print(f"X_std           = {X_std}")    
#     else:
#         print(f"X_mean.shape    = {X_mean.shape}")
#         print(f"X_std.shape     = {X_std.shape}")    
        
    
#     return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test)

def load_tropic_data(load_dir, loc_lon = 0, loc_lat = 0, coast=False):

    # train_years = 200

    # make labels
    filename = 'mjo_4back_200_precip.nc'#'mjo_4back_precip_local.nc'
    var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values * 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time'][4:]#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    if(coast):
        temp_label = np.loadtxt(load_dir+"loc_"+str(loc_lon)+"_"+str(loc_lat)+"_5mean_14days.txt")
    else:
        # temp_label = np.loadtxt(load_dir+"local_alaska_points.txt")
        temp_label = np.loadtxt(load_dir+"alaska_points.txt")

    # avgvar_day = xr.open_dataset(load_dir+"mjo_200year_precip_cycle.nc")['200precipcycle'].values#[:,:,:,np.newaxis]


    # var_c = []

    # full_years = 15


    # #Make a smarter way of doing this eventually
    # for i in np.arange(0,(full_years*365)+4,1):
    #     #   REMOVE -4 AS NEEDED
    #     #var_c.append(var[i,:,:,:] - avgvar_day[(i-4)%365,:,:,:])
    #     var_c.append(var[i,:,:] - avgvar_day[(i-4)%365,:,:])

    # var_c = np.asarray(var_c)

    # var_c_fw = running_mean(var_c, 5)[:,:,:,np.newaxis]

    # print("RUNNING MEAN ARRAY SIZE: "  + str(var_c_fw.shape))

    # var_200years = var_c[4:, :, :]

    # var_checkcycle = []

    # var_checkloc = []

    # for i in np.arange(0,full_years,1):
    #     var_checkcycle.append(var_200years[150+(365*i),:,:])
    #     var_checkloc.append(var_200years[150+(365*i),40,40])

    # var_checkcycle = np.asarray(var_checkcycle)

    # var_checkcycle = np.mean(var_checkcycle, axis=0)

    # var_checkloc = np.asarray(var_checkloc)

    # var_checkloc = np.mean(var_checkloc)


    # # area_lats = []
    # # for lat in lats:
    # #     area_lats.append(np.sqrt(np.cos(np.deg2rad(lat))))

    # # area_lats = np.asarray(area_lats)
    
    # # for i in range(len(area_lats)):
    # #     var_c[:,i,:,:] = var_c[:,i,:,:] * area_lats[i]


    # fig = plt.figure(figsize=(20, 16))
    # fig.tight_layout()

    # spec = fig.add_gridspec(4, 5)

    # plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

    # sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

    # plt.set_cmap('cmr.copper')
    # img = sub1.contourf(np.asarray(lons), np.asarray(lats), np.asarray(var_checkcycle), np.linspace(-100, 100, 41), transform=ccrs.PlateCarree())
    # # plt.xticks(np.arange(-180,181,30), np.concatenate((np.arange(0,181,30),np.arange(-160,1,30)), axis = None))
    # # sub1.set_xticks(np.arange(-180,181,30), np.arange(-180,181,30))
    # sub1.set_xticks(np.arange(-180,181,30))
    # sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
    # sub1.set_yticks(np.arange(-90,91,15))
    # sub1.set_xlim(-140,120)
    # sub1.set_ylim(-30,30)
    # sub1.set_xlabel("Longitude (degrees)",fontsize=25)
    # sub1.set_ylabel("Latitude (degrees)",fontsize=25)
    # cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
    # cbar.set_label("mm/day", fontsize=25)

    # sub1.coastlines()
    # # plt.savefig(('/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/senscycle_test.png'), bbox_inches='tight', dpi=400)

    # # plt.show()

    # loc_pres = []
    # for i in range(var_c_fw.shape[0]):
    #     loc_pres.append(var_c_fw[i,50,40,0])

    # loc_pres = np.asarray(loc_pres)

    # plt.figure(figsize=(20,10))
    # plt.title("Tropics Precip - Seasonal Cycle removed", fontsize=40)
    # plt.xlabel("Years", fontsize=30)
    # plt.xlim(0,10)
    # plt.ylabel("Precip Anamoly", fontsize=30)
    # plt.plot(np.arange(0, len(loc_pres),1)/365, loc_pres)
    # # plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'data_examine.png'), bbox_inches='tight', dpi=400)
    # loc_pres = []

    # print(int(var_c_fw.shape[0]/365))
    # for i in range(int(var_c_fw.shape[0]/365)):
    #     loc_pres.append(var_c_fw[(i*365)+100,50,40,0])

    # loc_pres = np.asarray(loc_pres)
    # print("Mean: " + str(np.mean(var_checkloc)))
    # # plt.show()

    # var_c_fw = xr.open_dataset(load_dir+"mjo_15year_precip_nocycle_5back.nc")['200precip5back'].values
    var_c_fw = xr.open_dataset(load_dir+"mjo_200year_precip_nocycle_5back.nc")['200precip5back'].values[:,:,:,np.newaxis]

    # all_years = time["time.year"].values

    all_months = time["time.month"].values[:73000]

    # all_days = time["time.day"].values

    # months = np.unique(all_months)

    # days = np.unique(all_days)

    # winter_months = []

    # print(var_c_fw.shape)
    # print(len(all_months))
    # for i in np.arange(len(all_months)):
    #     month = all_months[i]
    #     if (month == 1 or month == 2 or month == 11 or month == 12):
    #         winter_months.append(var_c_fw[i,:,:])
    
    # winter_months = np.asarray(winter_months)[:,:,:,np.newaxis]


    # print("winter_months shape: " + str(winter_months.shape))

    # print(time)
    # print(type(time))

    # filename = 'small_2mtemp_200_nomarch_winter.nc'
    # time     = xr.open_dataset(load_dir+filename)['time']
    # time = time[:time.shape[0]- 50]
    
    return np.asarray(temp_label), var_c_fw, lats, lons, time

def get_and_process_tropic_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    print(raw_data.shape)
    ####USER

    #CHECK YEARS
    # fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 15 + 50 + 3).astype('datetime64[ns]')

    # fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 15 + 50 + 3)), "time": fake_dates})
    # raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))
    # print(durs['time'])
    #####

    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:]
        raw_labels  = raw_labels[index,]


    # separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    # nyr_val   = int(len(years)*.2)

    #useless
    # nyr_val = 140
    # nyr_train = len(years) - nyr_val
    
    # years_train = np.random.choice(years,size=(nyr_train,),replace=False)
    years_train = years[:70]
    #years_train = years[:7]

#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng

    # years_val   = np.setxor1d(years,years_train)

    years_val = years[70:90]
    #years_val = years[7:10]

    # years_test  = 2010
    years_test = years[90:200]
    #years_test = years[10:15]
    
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)
    
    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    ic(iyears_val)
    iyears_test = np.where(years_test==True)[0]   
        
    # Standardize the input based on training data only
    print("wdadadadasdada")
    print(iyears_train)
    print(iyears_train.shape)
    print(raw_data.shape)
    # print(years)
    print(years.shape)
    print(years_test.shape)
    X_train_raw = raw_data[iyears_train,:,:]
    if( (standardize==True) or (standardize=='all')):
        X_mean  = np.mean(X_train_raw.flatten())
        X_std   = np.std(X_train_raw.flatten())
    elif(standardize=='pixel'):
        X_mean  = np.mean(X_train_raw,axis=(0,))
        X_std   = np.std(X_train_raw,axis=(0,))
        X_std[X_std==0] = 1.
    else:
        X_mean  = 0. 
        X_std   = 1. 

    # Create the target vectors, which includes a second dummy column.
    X_train = (raw_data[iyears_train,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[iyears_train];
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    time_train = raw_time['time'][iyears_train]
    time_val   = raw_time['time'][iyears_val]
    time_test  = raw_time['time'][iyears_test]
    
    
    print(f"raw_data        = {np.shape(raw_data)}")
    print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
    print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
    print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
    if(standardize != 'pixel'):
        print(f"X_mean          = {X_mean}")
        print(f"X_std           = {X_std}")    
    else:
        print(f"X_mean.shape    = {X_mean.shape}")
        print(f"X_std.shape     = {X_std.shape}")    
        
    
    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test)

def load_tropic_data_winter(load_dir, loc_lon = 0, loc_lat = 0, coast=False):

    # train_years = 200

    # make labels
    filename = 'mjo_4back_200_precip.nc'#'mjo_4back_precip_local.nc'
    var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values * 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time'][4:]#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    if(coast):
        temp_label = np.loadtxt(load_dir+"winter_ternary_loc_"+str(loc_lon)+"_"+str(loc_lat)+".txt")
    else:
        # temp_label = np.loadtxt(load_dir+"local_alaska_points.txt")
        temp_label = np.loadtxt(load_dir+"winter_ternary_alaska_points.txt")



    # var_c_fw = xr.open_dataset(load_dir+"mjo_15year_precip_nocycle_5back.nc")['200precip5back'].values
    var_c_fw = xr.open_dataset(load_dir+"mjo_200year_precip_nocycle_5back.nc")['200precip5back'].values

    # all_years = time["time.year"].values

    all_months = time["time.month"].values[:73000]

    # all_days = time["time.day"].values

    # months = np.unique(all_months)

    # days = np.unique(all_days)

    winter_months = []

    # print(var_c_fw.shape)
    # print(len(all_months))
    for i in np.arange(len(all_months)):
        month = all_months[i]
        if (month == 1 or month == 2 or month == 11 or month == 12):
            winter_months.append(var_c_fw[i,:,:])
    
    winter_months = np.asarray(winter_months)[:,:,:,np.newaxis]


    print("winter_months shape: " + str(winter_months.shape))

    print(time)
    print(type(time))

    filename = 'small_2mtemp_200_nomarch_winter.nc'
    time     = xr.open_dataset(load_dir+filename)['time']
    time = time[:time.shape[0]- 50]
    
    return np.asarray(temp_label), winter_months, lats, lons, time

def load_tropic_data_winter_ERA5(load_dir, loc_lon = 0, loc_lat = 0, coast=False):

    # train_years = 200s

    # make labels
    filename = 'MAIN_era5_precip_mjo_notrend_anoms.nc'#'mjo_4back_precip_local.nc'
    var_raw     = xr.open_dataset(load_dir+filename)['tp'] #* 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time']#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    var_rolled = var_raw.rolling(time = 5).mean().dropna("time", how='all')

    var_rolled_sliced = var_rolled.sel(time=slice("1960-01-01", "2020-12-31"))

    var = var_rolled_sliced.sel(time=var_rolled_sliced.time.dt.month.isin([1, 2, 11, 12]))
    # tropics_lats = np.squeeze(np.where((lats>= -30) & (lats <=30)))
    # tropics_lons = np.squeeze(np.where((lons>= -70) & (lons <=70)))

    # print(tropics_lats)
    # print(lats[tropics_lats])
    # print(tropics_lons)
    # print(lons[tropics_lons])
    # print(tropics_lats.shape)
    # print(tropics_lons.shape)
    # print(var.shape)

    if(coast):
        temp_label = np.loadtxt(load_dir+"loc_"+str(loc_lon)+"_"+str(loc_lat)+"_5mean_14days.txt")
    else:
        # temp_label = np.loadtxt(load_dir+"local_alaska_points.txt")
        temp_label = np.loadtxt(load_dir+"ERA5_winter_ternary_alaska_points_5days.txt")

    # var = var[:,tropics_lats,:]
    # var = var[:,:,tropics_lons]


    # all_years = time["time.year"].values

    all_months = time["time.month"].values#[:60*365]

    # print(all_months)
    # print(all_months.shape)
    # print(var.shape)
    # print(var_rolled_sliced.shape)
    # print("hello??")

    # all_days = time["time.day"].values

    # months = np.unique(all_months)

    # days = np.unique(all_days)

    # winter_months = []

    # # print(var.shape)
    # # print(len(all_months))
    # for i in np.arange(len(all_months)):
    #     month = all_months[i]
    #     if (month == 1 or month == 2 or month == 11 or month == 12):
    #         winter_months.append(var[i,:,:])
    
    # winter_months = np.asarray(winter_months)[:,:,:,np.newaxis]


    # print("winter_months shape: " + str(winter_months.shape))

    # print(time)
    # print(type(time))

    # filename = 'small_2mtemp_200_nomarch_winter.nc'
    # time     = xr.open_dataset(load_dir+filename)['time']
    # time = time[:time.shape[0]- 50]
    # print(time)
    # print(time.shape)
    # rus = RandomUnderSampler(random_state=0)
    # print(np.arange(0,var.shape[0],1)[0:120*50])
    # data_amt = np.arange(0,var.shape[0],1)[0:120*50].reshape(-1,1)
    # print(data_amt.shape)
    # # print(data_amt)
    # crap, shit = rus.fit_resample(data_amt, temp_label[0:120*50])

    # print("CRAPPPPPPPPPPPPPPP")
    # print(crap.shape)
    # print(crap)

    # print("SHITTTTTTTTTTTTT")
    # print(shit.shape)
    # print(shit)
    # print(temp_label[0:120*50])

    # crap, shit = (np.asarray(t) for t in zip(*sorted(zip(crap, shit))))

    # print("SORTTTTTTTTTTTTTT")
    # print(crap)
    # print(shit)
    # print(temp_label[0:120*50])
    
    return np.asarray(temp_label), var.values[:,:,:,np.newaxis], lats, lons, var.time

def get_and_process_tropic_data_winter(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    print(raw_data.shape)
    ####USER

    #CHECK YEARS
    # fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 15 + 50 + 3).astype('datetime64[ns]')

    # fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 15 + 50 + 3)), "time": fake_dates})
    # raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))
    # print(durs['time'])
    #####

    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    # nyr_val   = int(len(years)*.2)

    #useless
    # nyr_val = 140
    # nyr_train = len(years) - nyr_val
    
    # years_train = np.random.choice(years,size=(nyr_train,),replace=False)
    years_train = years[:70]
    #years_train = years[:7]

#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng

    # years_val   = np.setxor1d(years,years_train)

    years_val = years[70:90]
    #years_val = years[7:10]

    # years_test  = 2010
    years_test = years[90:200]
    #years_test = years[10:15]
    
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)
    
    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    ic(iyears_val)
    iyears_test = np.where(years_test==True)[0]   
        
    # Standardize the input based on training data only
    # print("wdadadadasdada")
    # print(iyears_train)
    # print(iyears_train.shape)
    # print(raw_data.shape)
    # # print(years)
    # print(years.shape)
    # print(years_test.shape)
    X_train_raw = raw_data[iyears_train,:,:]
    if( (standardize==True) or (standardize=='all')):
        X_mean  = np.mean(X_train_raw.flatten())
        X_std   = np.std(X_train_raw.flatten())
    elif(standardize=='pixel'):
        X_mean  = np.mean(X_train_raw,axis=(0,))
        X_std   = np.std(X_train_raw,axis=(0,))
        X_std[X_std==0] = 1.
    else:
        X_mean  = 0. 
        X_std   = 1. 

    # Create the target vectors, which includes a second dummy column.
    X_train = (raw_data[iyears_train,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[iyears_train];
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    time_train = raw_time['time'][iyears_train]
    time_val   = raw_time['time'][iyears_val]
    time_test  = raw_time['time'][iyears_test]
    
    
    print(f"raw_data        = {np.shape(raw_data)}")
    print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
    print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
    print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
    if(standardize != 'pixel'):
        print(f"X_mean          = {X_mean}")
        print(f"X_std           = {X_std}")    
    else:
        print(f"X_mean.shape    = {X_mean.shape}")
        print(f"X_std.shape     = {X_std.shape}")    
        
    
    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test)

def get_and_process_tropic_data_winter_ERA5(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False, r_seed = 0):
    print(raw_data.shape)
    ####USER

    #CHECK YEARS
    # fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 15 + 50 + 3).astype('datetime64[ns]')

    # fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 15 + 50 + 3)), "time": fake_dates})
    # raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))
    # print(durs['time'])
    #####

    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)
    

    ##########################
    rng.shuffle(years)
    ##########################
    # nyr_val   = int(len(years)*.2)

    #useless
    # nyr_val = 140
    # nyr_train = len(years) - nyr_val
    
    # years_train = np.random.choice(years,size=(nyr_train,),replace=False)
    years_train = years[:52] #can go up to 61
    #years_train = years[:7]

#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng

    # years_val   = np.setxor1d(years,years_train)

    years_val = years[52:61]
    #years_val = years[7:10]

    # years_test  = 2010
    years_test = years[55:61]
    #years_test = years[10:15]
    
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)
    
    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    ic(iyears_val)
    iyears_test = np.where(years_test==True)[0]   


    print("wdadadadasdada")
    print(r_seed)
    count_arr = np.bincount(raw_labels.astype(int)[iyears_val])
    
    print("number of 0: " + str(count_arr[0]))
    print("number of 1: " + str(count_arr[1]))
    print("number of 2: " + str(count_arr[2]))
        # print(iyears_train)
    # print(iyears_train.shape)
    # print(raw_data.shape)
    # # print(years)
    # print(years.shape)
    print(iyears_train)
    ################
    rus = RandomUnderSampler(random_state=r_seed)
    data_amt = iyears_train.reshape(-1,1)
    bal_data_train, bal_labels_train = rus.fit_resample(data_amt, raw_labels[iyears_train])
    bal_data_train, bal_labels_train = (np.asarray(t) for t in zip(*sorted(zip(bal_data_train, bal_labels_train))))
    bal_data_train = bal_data_train.reshape(-1,1)[:,0]

    rus = RandomUnderSampler(random_state=r_seed)
    data_amt = iyears_val.reshape(-1,1)
    bal_data_val, bal_labels_val = rus.fit_resample(data_amt, raw_labels[iyears_val])
    bal_data_val, bal_labels_val = (np.asarray(t) for t in zip(*sorted(zip(bal_data_val, bal_labels_val))))
    bal_data_val = bal_data_val.reshape(-1,1)[:,0]

    rus = RandomUnderSampler(random_state=r_seed)
    data_amt = iyears_test.reshape(-1,1)
    bal_data_test, bal_labels_test = rus.fit_resample(data_amt, raw_labels[iyears_test])
    bal_data_test, bal_labels_test = (np.asarray(t) for t in zip(*sorted(zip(bal_data_test, bal_labels_test))))
    bal_data_test = bal_data_test.reshape(-1,1)[:,0]

    ################
    # Standardize the input based on training data only
    # X_train_raw = raw_data[iyears_train,:,:]
    # if( (standardize==True) or (standardize=='all')):
    #     X_mean  = np.mean(X_train_raw.flatten())
    #     X_std   = np.std(X_train_raw.flatten())
    # elif(standardize=='pixel'):
    #     X_mean  = np.mean(X_train_raw,axis=(0,))
    #     X_std   = np.std(X_train_raw,axis=(0,))
    #     X_std[X_std==0] = 1.
    # else:
    #     X_mean  = 0. 
    #     X_std   = 1. 

    # # Create the target vectors, which includes a second dummy column.

    # ###########################################################
    # X_train = (raw_data[iyears_train,:,:] - X_mean) / X_std
    # X_val   = (raw_data[iyears_val,:,:] - X_mean) / X_std
    # X_test  = (raw_data[iyears_test,:,:] - X_mean) / X_std

    # X_train[X_train==0.] = 0.
    # X_val[X_val==0.]     = 0.
    # X_test[X_test==0.]   = 0.    
    
    # y_train = raw_labels[iyears_train]
    # y_val   = raw_labels[iyears_val]
    # y_test  = raw_labels[iyears_test]


    # print(raw_time['time'])
    # print(iyears_train)
    # print(iyears_train.shape)
    # time_train = raw_time['time'][iyears_train]
    # time_val   = raw_time['time'][iyears_val]
    # time_test  = raw_time['time'][iyears_test]

    ###########################################################

    X_train_raw = raw_data[bal_data_train,:,:]
    if( (standardize==True) or (standardize=='all')):
        X_mean  = np.mean(X_train_raw.flatten())
        X_std   = np.std(X_train_raw.flatten())
    elif(standardize=='pixel'):
        X_mean  = np.mean(X_train_raw,axis=(0,))
        X_std   = np.std(X_train_raw,axis=(0,))
        X_std[X_std==0] = 1.
    else:
        X_mean  = 0. 
        X_std   = 1. 


    X_train = (raw_data[bal_data_train,:,:] - X_mean) / X_std
    X_val   = (raw_data[bal_data_val,:,:] - X_mean) / X_std
    X_test  = (raw_data[bal_data_test,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[bal_data_train]
    y_val   = raw_labels[bal_data_val]
    y_test  = raw_labels[bal_data_test]

    count_arr = np.bincount(raw_labels.astype(int)[bal_data_val])

    print("number of 0: " + str(count_arr[0]))
    print("number of 1: " + str(count_arr[1]))
    print("number of 2: " + str(count_arr[2]))

    print(raw_time['time'])
    print(bal_data_train)
    print(bal_data_train.shape)
    time_train = raw_time['time'][bal_data_train]
    time_val   = raw_time['time'][bal_data_val]
    time_test  = raw_time['time'][bal_data_test]
    
    
    print(f"raw_data        = {np.shape(raw_data)}")
    print(f"training data   = {np.shape(X_train)}, {np.shape(y_train)}")
    print(f"validation data = {np.shape(X_val)}, {np.shape(y_val)}")
    print(f"test data       = {np.shape(X_test)}, {np.shape(y_test)}")
    if(standardize != 'pixel'):
        print(f"X_mean          = {X_mean}")
        print(f"X_std           = {X_std}")    
    else:
        print(f"X_mean.shape    = {X_mean.shape}")
        print(f"X_std.shape     = {X_std.shape}")    
        
    # For bigger val
    ##################
    X_test = X_val
    y_test = y_val
    ##################
    #dum
    # X_val = X_train
    # y_val = y_train
    # X_test = X_train
    # y_test = y_train

    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test)


def get_raw_temp_data(load_dir):
    #number of days

    train_years = 200

    # make labels
    filename = 'small_2mtemp.nc'
    temp     = xr.open_dataset(load_dir+filename)['tas'].values#[:,96:,80:241,np.newaxis]

    #number of days
    days = 365

    avgtemp_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tropic_200year_temp_cycle.nc")['200tempcycle'].values

    full_loc_temp = []

    for i in np.arange(0,train_years*days,1):

        full_loc_temp.append(temp[i, 52,110] - avgtemp_day[i%days, 52,110])

    return (full_loc_temp, avgtemp_day, temp)

