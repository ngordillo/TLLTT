import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import pandas as pd
import cartopy.crs as ccrs
import cmasher as cmr

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "23 November 2021"

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, np.zeros(x.shape[2]), axis = 0), axis =0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def makeRasterArray(fig, ax, fig_shape=(4,3)):

    fig.add_axes(ax)
    fig.set_dpi(100)
    ax.axis('off')
    fig.set_size_inches(fig_shape)  # fig_shape in inches
    
    return makeRasterArray_main(fig)

def makeRasterArray_main(fig):
    """ Make a raster array from a pyplot figure.

    References:
       https://matplotlib.org/stable/gallery/misc/agg_buffer_to_array.html
       https://stackoverflow.com/questions/51059581/matplotlib-convert-plot-to-numpy-array-without-borders

    """
    
    fig.canvas.draw()                                      # Force a draw so we can grab the pixel buffer.
    return np.array(fig.canvas.renderer.buffer_rgba())     # Grab the pixel buffer and dump it into a numpy array.


def get_and_process_mjo_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    
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
        raw_data    = raw_data[index,:,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    all_years = raw_time["time.year"].values
    years     = np.unique(all_years)

    nyr_val   = int(len(years)*.2)    
    nyr_train = len(years) - nyr_val
    
    years_train = np.random.choice(years,size=(nyr_train,),replace=False)
#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng
    years_val   = np.setxor1d(years,years_train)
    years_test  = 2010
    
    ic(years_train)
    ic(years_val)
    ic(years_test)    
    
    years_train = np.isin(all_years, years_train)
    years_val   = np.isin(all_years, years_val)
    years_test  = np.isin(all_years, years_test)
    
    iyears_train = np.where(years_train==True)[0]
    iyears_val = np.where(years_val==True)[0]
    iyears_test = np.where(years_test==True)[0]   
        
    # Standardize the input based on training data only
    X_train_raw = raw_data[iyears_train,:,:,:]
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
    X_train = (raw_data[iyears_train,:,:,:] - X_mean) / X_std
    X_val   = (raw_data[iyears_val,:,:,:] - X_mean) / X_std
    X_test  = (raw_data[iyears_test,:,:,:] - X_mean) / X_std

    X_train[X_train==0.] = 0.
    X_val[X_val==0.]     = 0.
    X_test[X_test==0.]   = 0.    
    
    y_train = raw_labels[iyears_train]
    y_val   = raw_labels[iyears_val]
    y_test  = raw_labels[iyears_test]

    time_train = raw_time[iyears_train]
    time_val   = raw_time[iyears_val]
    time_test  = raw_time[iyears_test]
    
    
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

def cut_longitudes(data,lon):
    ilon = np.where(np.logical_and(lon<=270,lon>=10))[0] #240
    # print(ilon)
#     ilon = np.where((lon.values<=360) & (lon.values>=0))[0]
    lon = lon[ilon]
    data = data[:,:,ilon]    
    return data, lon

def load_mjo_data(load_dir):
    
    # make labels
    filename = 'WH04_RMM.nc'
    time     = xr.open_dataset(load_dir+filename)['RMM_amp']['time']    
    rmm_amp  = xr.open_dataset(load_dir+filename)['RMM_amp']
    rmm_ph   = xr.open_dataset(load_dir+filename)['RMM_ph']
    
    i = np.where(rmm_amp<.5) # RMM amplitudes less than 0.5 are labeled as "phase 0"
    rmm_ph[i] = 0

    print(np.unique(rmm_ph))    
    
    # get the fields
    filename = 'olr.20NS.noac_120mean_norm.nc'
    lat      = xr.open_dataset(load_dir+filename)['olr']['latitude']
    lon      = xr.open_dataset(load_dir+filename)['olr']['longitude']
    olr      = xr.open_dataset(load_dir+filename)['olr'].values[:,:,:,np.newaxis]
    
    filename = 'u200.20NS.noac_120mean_norm.nc'
    u200     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis] 
    
    filename = 'u850.20NS.noac_120mean_norm.nc'
    u850     = xr.open_dataset(load_dir+filename)['u'].values[:,:,:,np.newaxis]
    # print("olr shape:")
    # print(olr.shape)

    olr, __   = cut_longitudes(olr,lon)
    u200, __  = cut_longitudes(u200,lon)
    u850, lon = cut_longitudes(u850,lon)
    data = np.concatenate((olr,u200,u850),axis=3)
    
    return rmm_ph, data, lat, lon, time


def get_and_process_data(filename, rng, colored=False, standardize=False, shuffle=True):
    print('loading ' + filename)
    
    mat_contents = sio.loadmat(filename)
    raw_data     = mat_contents['data']
    raw_labels   = mat_contents['labels'].T[:,0]
    try:
        lat      = mat_contents['lat'][0]
        lon      = mat_contents['lon'][0]
    except:
        lat      = np.nan
        lon      = np.nan
        
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
        raw_data    = raw_data[index,:,:,:]
        raw_labels  = raw_labels[index,]

    # separate the data into training, validation and testing
    ndata       = len(raw_data)
    ntest       = np.int32(0.000001 * ndata)
    nval        = np.int32(0.2 * ndata)
#     ntest       = np.int32(0.001 * ndata)
#     nval        = np.int32(0.25 * ndata)
    ntrain      = np.int32(ndata - nval - ntest)

    # Standardize the input based on training data only
    X_train_raw = raw_data[0:ntrain,:,:]
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

    X_train = (raw_data[0:ntrain,:,:] - X_mean) / X_std
    X_val   = (raw_data[ntrain:ntrain+nval,:,:] - X_mean) / X_std
    X_test  = (raw_data[ntrain+nval:,:,:] - X_mean) / X_std

    y_train = raw_labels[0:ntrain]
    y_val   = raw_labels[ntrain:ntrain+nval]
    y_test  = raw_labels[ntrain+nval:]

    if(len(np.shape(X_train))==3):
        X_train = np.asarray(X_train)[:,:,:,np.newaxis]
        X_val   = np.asarray(X_val)[:,:,:,np.newaxis]    
        X_test  = np.asarray(X_test)[:,:,:,np.newaxis]        
    
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
   
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, lat, lon)

def subsample_extremes(random_seed, X_train, y_train, X_val, y_val, X_test, y_test):
    # subsample training data
    np.random.seed(random_seed)
    labels0 = np.where(y_train==0)[0]
    labels1 = np.where(y_train==1)[0]
    labels2 = np.where(y_train==2)[0]    
    labels0_subsample = np.random.choice(labels0,size=len(labels1),replace=False)
    X_train = X_train[np.concatenate((labels0_subsample,labels1,labels2)),:,:,:]
    y_train = y_train[np.concatenate((labels0_subsample,labels1,labels2))]

    # subsample validation data
    np.random.seed(random_seed)
    labels0 = np.where(y_val==0)[0]
    labels1 = np.where(y_val==1)[0]
    labels2 = np.where(y_val==2)[0]    
    labels0_subsample = np.random.choice(labels0,size=len(labels1),replace=False)
    X_val = X_val[np.concatenate((labels0_subsample,labels1,labels2)),:,:,:]
    y_val = y_val[np.concatenate((labels0_subsample,labels1,labels2))]

    # show it worked
    labels0 = np.where(y_train==0)[0]
    labels1 = np.where(y_train==1)[0]
    labels2 = np.where(y_train==2)[0]    
    ic(len(labels0),len(labels1),len(labels2))

    labels0 = np.where(y_val==0)[0]
    labels1 = np.where(y_val==1)[0]
    labels2 = np.where(y_val==2)[0]    
    __ = ic(len(labels0),len(labels1),len(labels2))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_tropic_data(load_dir, loc_lon = 0, loc_lat = 0, coast=False):

    train_years = 200

    # make labels
    filename = 'mjo_4back_200_precip.nc'
    var     = np.float64(xr.open_dataset(load_dir+filename)['pr'].values * 86400)#[:,:,:,np.newaxis]#[:,96:,80:241,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time'].values#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    if(coast):
        temp_label = np.loadtxt(load_dir+"loc_"+str(loc_lon)+"_"+str(loc_lat)+"_5mean_14days.txt")
    else:
        temp_label = np.loadtxt(load_dir+"alas_tempclass_200years_5mean_14days.txt")

    avgvar_day = xr.open_dataset(load_dir+"mjo_200year_precip_cycle.nc")['200precipcycle'].values#[:,:,:,np.newaxis]


    var_c = []

    full_years = 200

    for i in np.arange(0,(full_years*365)+4,1):
        #   REMOVE -4 AS NEEDED
        #var_c.append(var[i,:,:,:] - avgvar_day[(i-4)%365,:,:,:])
        var_c.append(var[i,:,:] - avgvar_day[(i-4)%365,:,:])

    var_c = np.asarray(var_c)

    var_c_fw = running_mean(var_c, 5)[:,:,:,np.newaxis]

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


    # area_lats = []
    # for lat in lats:
    #     area_lats.append(np.sqrt(np.cos(np.deg2rad(lat))))

    # area_lats = np.asarray(area_lats)
    
    # for i in range(len(area_lats)):
    #     var_c[:,i,:,:] = var_c[:,i,:,:] * area_lats[i]


    fig = plt.figure(figsize=(20, 16))
    fig.tight_layout()

    spec = fig.add_gridspec(4, 5)

    plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

    sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

    plt.set_cmap('cmr.copper')
    img = sub1.contourf(np.asarray(lons), np.asarray(lats), np.asarray(var_checkcycle), np.linspace(-100, 100, 41), transform=ccrs.PlateCarree())
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
    plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/senscycle_test.png'), bbox_inches='tight', dpi=400)

    # plt.show()

    loc_pres = []
    for i in range(var_c_fw.shape[0]):
        loc_pres.append(var_c_fw[i,50,40,0])

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
        loc_pres.append(var_c_fw[(i*365)+100,50,40,0])

    loc_pres = np.asarray(loc_pres)
    print("Mean: " + str(np.mean(var_checkloc)))
    # plt.show()


    return np.asarray(temp_label), np.asarray(var_c_fw), lats, lons, time

def get_and_process_tropic_data(raw_labels, raw_data, raw_time, rng, colored=False, standardize=False, shuffle=False):
    
    ####USER
    fake_dates = pd.date_range("1800-01-01", freq="D", periods=365 * 200 + 50).astype('datetime64[ns]')

    fake_time = xr.Dataset({"foo": ("time", np.arange(365 * 200 + 50)), "time": fake_dates})
    raw_time = fake_time.sel(time=~((fake_time.time.dt.month == 2) & (fake_time.time.dt.day == 29)))
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
    nyr_val = 140
    nyr_train = len(years) - nyr_val
    
    # years_train = np.random.choice(years,size=(nyr_train,),replace=False)
    years_train = years[:70]
#     years_train = rng.choice(years,size=(nyr_train,),replace=False)     #use this syntax next time to keep everything using rng
    # years_val   = np.setxor1d(years,years_train)
    years_val = years[70:120]
    # years_test  = 2010
    years_test = years[120:200]
    
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
    
    y_train = raw_labels[iyears_train]
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

def load_z500_precip_data(load_dir):

    full_years = 200

    # make labels
    filename = 'NA_precip.nc'
    precip = xr.open_dataset(load_dir+filename)['pr'].values[:,:,:,np.newaxis] * 86400
    filename = 'NA_z500.nc'
    z500     = xr.open_dataset(load_dir+filename)['zg500'].values[:,:,:,np.newaxis]
    time     = xr.open_dataset(load_dir+filename)['time'].values#[:train_years*365]
    lats  = xr.open_dataset(load_dir+filename)['lat'].values#[96:]
    lons   = xr.open_dataset(load_dir+filename)['lon'].values#[80:241]

    temp_label = np.loadtxt("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_200years_threedays.txt")

    avgz500_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/200year_z500_cycle.nc")['200z500cycle'].values[:,:,:,np.newaxis]

    avgprecip_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/200year_precip_cycle.nc")['200precipcycle'].values[:,:,:,np.newaxis]

    z500_c = []

    precip_c = []
    
    for i in np.arange(0,full_years*365,1):
        z500_c.append(z500[i,:,:,:] - avgz500_day[i%365,:,:,:])
        precip_c.append(precip[i,:,:,:] - avgprecip_day[i%365,:,:,:])

    z500_c = np.asarray(z500_c)
    precip_c = np.asarray(precip_c)

    # area_lats = []
    # for lat in lats:
    #     area_lats.append(np.sqrt(np.cos(np.deg2rad(lat))))

    # area_lats = np.asarray(area_lats)
    
    # for i in range(len(area_lats)):
    #     pres_c[:,i,:,:] = pres_c[:,i,:,:] * area_lats[i]


    fig = plt.figure(figsize=(20, 16))
    fig.tight_layout()

    spec = fig.add_gridspec(4, 5)

    plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

    sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

    # main_corr = pres[0,:,:, 0]
    # c_avgpsl_day = avgpres_day[0,:,:, 0]

    # sub1.figure(figsize = (20, 16))
    # ax = sub1.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # ax.set_extent((-20, 60, -40, 45), crs=ccrs.PlateCarree())
    # print(c_avg_psl.shape)

    plt.set_cmap('cmr.copper')
    # sub1.plot(46,136,'go')
    print(z500_c.shape)
    img = sub1.contourf(np.asarray(lons), np.asarray(lats), np.asarray(precip_c[175,:,:,0]), np.linspace(-1, 4, 40), transform=ccrs.PlateCarree())
    # plt.xticks(np.arange(-180,181,30), np.concatenate((np.arange(0,181,30),np.arange(-160,1,30)), axis = None))
    # sub1.set_xticks(np.arange(-180,181,30), np.arange(-180,181,30))
    sub1.set_xticks(np.arange(-180,181,30))
    sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
    sub1.set_yticks(np.arange(-90,91,15))
    sub1.set_xlim(-80,120)
    sub1.set_ylim(0,90)
    sub1.set_xlabel("Longitude (degrees)",fontsize=25)
    sub1.set_ylabel("Latitude (degrees)",fontsize=25)
    cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
    cbar.set_label("mm/day", fontsize=25)

    sub1.coastlines()
    # plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'example_anom.png'), bbox_inches='tight', dpi=400)

    plt.show()

    loc_pres = []
    for i in range(z500_c.shape[0]):
        loc_pres.append(z500_c[i,46,136,0])

    loc_pres = np.asarray(loc_pres)

    plt.figure(figsize=(20,10))
    plt.title("Madison Temp - Seasonal Cycle removed", fontsize=40)
    plt.xlabel("Years", fontsize=30)
    plt.xlim(0,1)
    plt.ylabel("Temp Anamoly", fontsize=30)
    plt.plot(np.arange(0, len(loc_pres),1)/365, loc_pres)
    # plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'data_examine.png'), bbox_inches='tight', dpi=400)

    plt.show()

    data = np.concatenate((z500_c,precip_c),axis=3)

    return np.asarray(temp_label), np.asarray(data), lats, lons, time