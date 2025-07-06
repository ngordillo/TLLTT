# This Looks Like That There - S2S Release Version


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd
import cartopy.crs as ccrs
import cmasher as cmr

from imblearn.under_sampling import RandomUnderSampler

# Function to calculate running mean over a specified window
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.zeros(x.shape[2]), axis = 0), axis =0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Function to load tropical data from specified directory
def load_tropic_data(load_dir, loc_lon = 0, loc_lat = 0, coast=False):
    # train_years = 200

    # make labels
    filename = 'mjo_4back_200_precip.nc'#'mjo_4back_precip_local.nc'
    var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values * 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time'][4:]#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    # Load temperature labels based on coast parameter
    if(coast):
        temp_label = np.loadtxt(load_dir+"loc_"+str(loc_lon)+"_"+str(loc_lat)+"_5mean_14days.txt")
    else:
        # temp_label = np.loadtxt(load_dir+"local_alaska_points.txt")
        temp_label = np.loadtxt(load_dir+"alaska_points.txt")

    # Load additional precipitation data
    var_c_fw = xr.open_dataset(load_dir+"mjo_200year_precip_nocycle_5back.nc")['200precip5back'].values[:,:,:,np.newaxis]

    # all_years = time["time.year"].values

    all_months = time["time.month"].values[:73000]
    
    return np.asarray(temp_label), var_c_fw, lats, lons, time

# Function to process tropical data for training, validation, and testing
def get_and_process_tropic_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    print(raw_data.shape)

    # Process raw data based on color parameter
    if(colored == True):
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels   = np.asarray(raw_labels, dtype='float')

    # Shuffle data if specified
    if(shuffle==True):
        # shuffle the data
        print('shuffling the data before train/validation/test split.')
        index      = np.arange(0,raw_data.shape[0])
        rng.shuffle(index)  
        raw_data    = raw_data[index,:,:]
        raw_labels  = raw_labels[index,]

    # Separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    # years_train = np.random.choice(years,size=(nyr_train,),replace=False)
    years_train = years[:70]
    #years_train = years[:7]

    # Define validation and test years
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

    X_train_raw = raw_data[iyears_train,:,:]
    
    # Standardization logic
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

    # Set zero values to zero
    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[iyears_train];
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    time_train = raw_time['time'][iyears_train]
    time_val   = raw_time['time'][iyears_val]
    time_test  = raw_time['time'][iyears_test]
    
    # Print shapes of the datasets
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
    # Function to load winter tropical data

    # train_years = 200

    # make labels
    filename =  'rolled_mjo_4back_550_precip.nc' #'anom_mjo_4back_1000_precip.nc'#'mjo_4back_precip_local.nc'
    print(load_dir+filename)
    var_raw     = xr.open_dataset(load_dir+filename)['pr']#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]

    time     = xr.open_dataset(load_dir+filename)['time'][4:]#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    print(var_raw)

    # quit()

    var_rolled = var_raw
    #var_rolled = var_raw.rolling(time = 5).mean().dropna("time", how='all')
    # print(var_rolled)
    # dates = xr.cftime_range(start="1700", periods=73050, freq="D", calendar="noleap").to_datetimeindex()
    # var_rolled['time'] = dates

    # old_var_rolled_sliced = var_rolled.sel(time=slice("1700-01-01", "1898-12-31"))

    # old_time = old_var_rolled_sliced.sel(time=old_var_rolled_sliced.time.dt.month.isin([1, 2, 11, 12]))['time']

    # var_rolled_sliced = var_rolled.sel(time=slice("1700-11-01", "1899-02-28"))

    # var_rolled_sliced = var_rolled.sel(time=slice("1679-11-01", "2229-02-28"))

    # var_rolled_sliced = var_rolled.sel(time=slice("1679-11-01", "2257-02-28"))

    var_rolled_sliced = var_rolled.sel(time=xr.cftime_range(start= '0201-11-01',end = '0751-02-28',freq='D', calendar='noleap'))

    var = var_rolled_sliced.sel(time=var_rolled_sliced.time.dt.month.isin([1, 2, 11, 12]))

    # var['time'] = old_time
    # print(var)

    print(var.time)
    if(coast):
        # Load temperature labels based on coast parameter
        temp_label = np.loadtxt(load_dir+'temp_class/GCM_' + str(loc_lon) + '_' + str(loc_lat) + '_wint_550yrs_ternary_14day.txt')
    else:
        # Load temperature labels for non-coastal locations
        temp_label = np.loadtxt(load_dir+"temp_class/GCM_89_64_wint_550yrs_ternary_14day.txt")

    # print(var)
    # print(var.time)
    # print(len(temp_label))
    
    filename =    'anom_2mtemp_550_years.nc' #'anom_2mtemp_550_years.nc' # 'anom_large_550_2mtemp.nc' #'rolled_small_2mtemp.nc'  #'era_2mtemp_mjo_notrend_anoms.nc'  #'NEWera5_post1980_2mtemp_anoms.nc'  #'anom_2mtemp_full_era5_remap_order3.nc' #'anom_2mtemp_full_era5_remap_order1.nc' #'anom_2mtemp_full_era5_remap.nc' #'rolled_small_2mtemp.nc' #'MAIN_era5_2mtemp_mjo_notrend_anoms.nc'#era5_daily_2mtemp.nc'
    
    # Load raw temperature data
    raw_temps     = xr.open_dataset(load_dir+filename)['tas']#[:,96:,80:241] #era2mtempanom
    # time     = xr.open_dataset(load_dir+filename)['time'][0:(60*365)+14]
    t_lat  = xr.open_dataset(load_dir+filename)['lat'].values
    t_lon   = xr.open_dataset(load_dir+filename)['lon'].values
   
    return np.asarray(temp_label), var.values[:,:,:,np.newaxis], lats, lons, var.time, raw_temps, t_lat, t_lon

def load_tropic_data_winter_ERA5(load_dir, loc_lon = 0, loc_lat = 0, coast=False):
    # Function to load winter tropical data from ERA5

    # train_years = 200s

    # make labels
    filename =  'anom_precip_full_era5_remap_order3.nc' #'V2_4back_OLDera5_daily_precip_remap.nc' # 'anom_precip_full_era5_remap_order3.nc'#'mjo_4back_precip_local.nc' MAIN_era5_precip_mjo_notrend_anoms.nc
    var_raw     = xr.open_dataset(load_dir+filename)['tp'] #* 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time']#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    var_rolled = var_raw.rolling(time = 5).mean().dropna("time", how='all')

    # var_rolled_sliced = var_rolled.sel(time=slice("1951-01-01", "2020-12-31"))

    # var_rolled_sliced = var_rolled.sel(time=slice("1960-01-01", "2020-12-31"))
    
    # var_rolled_sliced = var_rolled.sel(time=slice("1960-01-01", "2020-12-31"))

    var_rolled_sliced = var_rolled.sel(time=slice("1951-11-01", "2021-02-28"))

    var = var_rolled_sliced.sel(time=var_rolled_sliced.time.dt.month.isin([1, 2, 11, 12]))

    if(coast):
        # Load temperature labels based on coast parameter
        temp_label = np.loadtxt(load_dir+"temp_class/ERA5_" + str(loc_lon) + '_' + str(loc_lat) + "_wint_550yrs_ternary_14day.txt")
    else:
        # Load temperature labels for alaska locations
        temp_label = np.loadtxt(load_dir+"temp_class/ERA5_89_64_wint_550yrs_ternary_14day.txt")

    all_months = time["time.month"].values#[:60*365]

    filename =    'anom_2mtemp_full_era5_remap_order3.nc' #'anom_2mtemp_550_years.nc' # 'anom_large_550_2mtemp.nc' #'rolled_small_2mtemp.nc'  #'era_2mtemp_mjo_notrend_anoms.nc'  #'NEWera5_post1980_2mtemp_anoms.nc'  #'anom_2mtemp_full_era5_remap_order3.nc' #'anom_2mtemp_full_era5_remap_order1.nc' #'anom_2mtemp_full_era5_remap.nc' #'rolled_small_2mtemp.nc' #'MAIN_era5_2mtemp_mjo_notrend_anoms.nc'#era5_daily_2mtemp.nc'
    
    # Load raw temperature data
    raw_temps     = xr.open_dataset(load_dir+filename)['t2m']#[:,96:,80:241] #era2mtempanom
    # time     = xr.open_dataset(load_dir+filename)['time'][0:(60*365)+14]
    t_lat  = xr.open_dataset(load_dir+filename)['lat'].values
    t_lon   = xr.open_dataset(load_dir+filename)['lon'].values

    return np.asarray(temp_label), var.values[:,:,:,np.newaxis], lats, lons, var.time, raw_temps, t_lat, t_lon
def get_and_process_tropic_data_winter(raw_labels, raw_data, raw_time, rng, train_yrs, val_yrs, test_yrs, temp_anoms, colored=False, standardize=False, shuffle=False, bal_data=False, r_seed = 0):
    # Print the shape of the raw input data
    print(raw_data.shape)

    # ============================
    # Optional: Shuffle the data (commented out)
    # ============================
    # if(shuffle==True):
    #     # shuffle the data
    #     print('shuffling the data before train/validation/test split.')
    #     index      = np.arange(0,raw_data.shape[0])
    #     rng.shuffle(index)  
    #     raw_data    = raw_data[index,:,:]
    #     raw_labels  = raw_labels[index,]

    # ============================
    # Define and flatten the year range into daily time steps
    # ============================
    # year_range = np.arange(1700, 1899, 1)
    # year_range = np.arange(1678, 2256, 1)
    # year_range = np.arange(1679, 2229, 1)
    year_range = np.arange(201, 751, 1)

    date_range = []
    for year in year_range:
        # date_range.append(pd.date_range(start= f'{year}-03-01',end = f'{year}-06-28',freq='d') + pd.offsets.Hour(00))
        date_range.append(xr.cftime_range(start= str(year).zfill(4) + '-03-01',end = str(year).zfill(4) + '-06-28',freq='D', calendar='noleap'))

    date_range = [item for sublist in date_range for item in sublist]

    print(len(date_range))

    # ============================
    # Extract years from raw_time and update it with new date_range
    # ============================
    raw_time_for_years = raw_time['time.year']
    raw_time_for_years['time'] = date_range
    all_years = raw_time_for_years["time.year"].values
    years     = np.unique(all_years)
    print(years)

    # ============================
    # Optional: Shuffle the years
    # ============================
    if(shuffle==True):
        rng.shuffle(years)

    # ============================
    # Split the years into training, validation, and testing sets
    # ============================
    years_train = years[:train_yrs] # 70 and 120
    #years_train = years[:7]
    # years_val   = np.setxor1d(years,years_train)
    years_val = years[train_yrs:val_yrs] # 70:90 and 120:160
    #years_val = years[7:10]
    # years_test  = 2010
    years_test = years[val_yrs:test_yrs] # 90:200 and 160:200
    #years_test = years[10:15]

    ###########################################################################################################
    # Index boolean masks and extract indices for training/validation/test
    ###########################################################################################################
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)

    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    iyears_test = np.where(years_test==True)[0]   

    # ============================
    # Optionally rebalance the data using undersampling
    # ============================
    if (bal_data):
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

        iyears_train = bal_data_train
        iyears_val = bal_data_val
        iyears_test = bal_data_test

    ##########################################################################################################
    # Print training indices for inspection
    ##########################################################################################################
    ic(iyears_train)

    # ============================
    # Standardize the input data using training set statistics
    # ============================
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

    # ============================
    # Normalize data and create input/output sets
    # ============================
    X_train = (raw_data[iyears_train,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    

    y_train = raw_labels[iyears_train]
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    temp_train = temp_anoms[iyears_train,:,:]
    temp_val   = temp_anoms[iyears_val,:,:]
    temp_test  = temp_anoms[iyears_test,:,:]

    time_train = raw_time['time'][iyears_train]
    time_val   = raw_time['time'][iyears_val]
    time_test  = raw_time['time'][iyears_test]
    ic(time_train)

    # ============================
    # Final data overview and return
    # ============================
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

    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test)

def get_and_process_tropic_data_winter_ERA5(raw_labels, raw_data, raw_time, rng, train_yrs, val_yrs, test_yrs, temp_anoms, translation = False, colored=False, standardize=False, shuffle=False, bal_data=False, r_seed = 0):
    # ============================
    # Initial preprocessing of input data
    # ============================
    print(raw_data.shape)
    
    if(colored == True):
        # If images are colored, convert RGB values to inverse grayscale float range
        raw_data = np.abs(1. - np.asarray(raw_data, dtype='float')/255.)
    else:
        raw_data = np.asarray(raw_data, dtype='float')
    raw_labels = np.asarray(raw_labels, dtype='float')

    # if(shuffle==True):
    #     # shuffle the data
    #     print('shuffling the data before train/validation/test split.')
    #     index      = np.arange(0,raw_data.shape[0])
    #     rng.shuffle(index)  
    #     raw_data    = raw_data[index,:,:]
    #     raw_labels  = raw_labels[index,]

    # ============================
    # Generate date range for winter season (Feb 28 – Jun 27) across all years
    # ============================
    year_range = np.arange(1951, 2021, 1)

    date_range = []
    for year in year_range:
        # Generate daily timestamps for each winter season
        date_range.append(pd.date_range(start= f'{year}-02-28',end = f'{year}-06-27',freq='d') + pd.offsets.Hour(00))
            
    # Flatten the nested list of date ranges
    date_range = [item for sublist in date_range for item in sublist]

    print(len(date_range))

    raw_time_for_years = raw_time['time.year']

    raw_time_for_years['time'] = date_range  # Replace with constructed winter timestamps

    all_years = raw_time_for_years["time.year"].values

    # Overwriting with actual year values from raw_time
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    # ============================
    # Select training, validation, and test years
    # ============================
    shuffle = False
    ############################################################################################################################################################
    if(shuffle==True):
        rng.shuffle(years)
    ##########################
    # nyr_val   = int(len(years)*.2)

    # Select year partitions based on indices or specified counts
    print(train_yrs)
    years_train = years[:train_yrs] #62 #can go up to 61
    #years_train = years[:7]

#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax nexƒt time to keep everything using rng

    # years_val   = np.setxor1d(years,years_train)

    years_val = years[train_yrs:val_yrs]

    if(translation):
        # If translation flag is on, use a bigger validation set
        years_val = years[0:test_yrs]

    # years_val = [list of specific years]
    #years_val = years[7:10]

    years_test = years[val_yrs:test_yrs]
    #years_test = years[10:15]

    # Print year selections for debugging
    ic(years_train)
    ic(years_val)
    ic(years_test)

    # Convert year masks to boolean arrays
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)

    # Index arrays for each split
    iyears_train = np.where(years_train==True)[0]
    iyears_val   = np.where(years_val==True)[0]
    ic(iyears_val)
    iyears_test  = np.where(years_test==True)[0]

    # Print class distribution for validation split
    count_arr = np.bincount(raw_labels.astype(int)[iyears_val])
    print("number of 0: " + str(count_arr[0]))
    print("number of 1: " + str(count_arr[1]))
    print("number of 2: " + str(count_arr[2]))

    # ============================
    # Optionally balance the dataset using RandomUnderSampler
    # ============================
    if (bal_data):
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

        # iyears_test not used here for balancing
        iyears_train = bal_data_train
        iyears_val = bal_data_val
        iyears_test = bal_data_val

    # ============================
    # Standardize the input data using training statistics
    # ============================
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

    # Apply normalization
    X_train = (raw_data[iyears_train,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:] - X_mean) / X_std

    # Maintain zeros as zeros
    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    # Create label sets
    y_train = raw_labels[iyears_train]
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    # Extract temperature anomaly data
    temp_train = temp_anoms[iyears_train,:,:]
    temp_val   = temp_anoms[iyears_val,:,:]
    temp_test  = temp_anoms[iyears_test,:,:]

    # ============================
    # Re-check label balance after balancing (if applied)
    # ============================
    print(iyears_val)
    count_arr = np.bincount(raw_labels.astype(int)[iyears_val])
    print("number of 0: " + str(count_arr[0]))
    print("number of 1: " + str(count_arr[1]))
    print("number of 2: " + str(count_arr[2]))
    print(bal_data)

    # ============================
    # Extract time values for each dataset
    # ============================
    print(raw_time['time'])
    print(iyears_train)
    print(iyears_train.shape)
    time_train = raw_time['time'][iyears_train]
    time_val   = raw_time['time'][iyears_val]
    time_test  = raw_time['time'][iyears_test]

    # ============================
    # Print shapes and stats for sanity check
    # ============================
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

    # ============================
    # For bigger val
    # ============================
    ##################
    if(translation):
        X_test = X_val
        y_test = y_val
        time_test = time_val

    # ============================
    # Final return of processed datasets
    # ============================
    return (X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test)



def get_temp_anoms(load_dir):
    # ============================
    # Load 2-meter temperature anomaly dataset from NetCDF
    # ============================

    # filename = '500_2mtemp.nc'
    # temp_500     = xr.open_dataset(load_dir+filename)['tas'].values[:,96:,80:241]
    #filename = 'NA_2mtemp.nc'
    filename = 'anom_2mtemp_full_era5_remap_order3.nc' 
    # Other example filenames:
    # 'anom_2mtemp_550_years.nc'
    # 'anom_large_550_2mtemp.nc'
    # 'rolled_small_2mtemp.nc'
    # 'era_2mtemp_mjo_notrend_anoms.nc'
    # 'NEWera5_post1980_2mtemp_anoms.nc'
    # 'anom_2mtemp_full_era5_remap_order1.nc'
    # 'anom_2mtemp_full_era5_remap.nc'
    # 'MAIN_era5_2mtemp_mjo_notrend_anoms.nc'
    # 'era5_daily_2mtemp.nc'

    temp = xr.open_dataset(load_dir+filename)['t2m']  # Load temperature anomaly data
    # time = xr.open_dataset(load_dir+filename)['time'][0:(60*365)+14]

    t_lat = xr.open_dataset(load_dir+filename)['lat'].values  # Latitude values
    t_lon = xr.open_dataset(load_dir+filename)['lon'].values  # Longitude values

    return (temp, t_lon, t_lat)


def process_nino_data(load_dir):
    # ============================
    # Load Niño 3.4 SST anomaly data from plain text file
    # ============================
    filename = 'nino34.long.anom.data.txt'

    with open(load_dir+filename) as f:
        # Read file line by line, converting each row to a list of floats
        nino_table = [[float(x) for x in line.split()] for line in f]

    # print(nino_table)
    # nino_table.index = nino_table.index.values

    return nino_table
