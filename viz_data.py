
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from icecream import ic
import scipy.io as sio
import cartopy.crs as ccrs
import cmasher as cmr
from netCDF4 import Dataset
import sys

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

a = np.arange(0,20,1)

print(a)
print(running_mean(a,7)[:15])
# avgpres_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/200year_pres_cycle.nc")
# # print(avgpres_day.variables)
# print(avgpres_day.data_vars)
# print(avgpres_day.data_vars['200prescycle'])
# test = avgpres_day.data_vars['200prescycle']
# print(test.shape)
# print(test[0,:,:])
# print("survived")
# sys.exit()

### for white background...
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='white')
plt.rc('axes',facecolor='white')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('xtick',color='black', labelsize='20')
plt.rc('ytick',color='black', labelsize='20')
################################  
################################  
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([]) 
            


# temp_vals = np.loadtxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/temps_200years_zerodays.txt')
# temp_labels = np.loadtxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/tempclass_200years_zerodays.txt')

# print(temp_labels)

# print(temp_vals.shape)

# # plt.figure(figsize=(20,10))
# fig, ax1 = plt.subplots(figsize=(20,10))
# ax1.set_title("Madison pressure - Seasonal Cycle removed", fontsize=40)
# ax1.set_xlabel("Years", fontsize=30)
# ax1.set_ylabel("Pres Anamoly", fontsize=30)
# ax1.plot(np.arange(0, len(temp_vals),1)/365, temp_vals, marker='o')
# ax1.axhline(np.percentile(temp_vals, 33.33))
# ax1.axhline(np.percentile(temp_vals, 66.66))
# ax1.set_xlim(0,1)

# ax2 = ax1.twinx()
# ax2.scatter(np.arange(0,len(temp_labels))/365, temp_labels, color='orange')
# # plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'madison_data_examine.png'), bbox_inches='tight', dpi=400)
# plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'LABEL_madison_data_examine.png'), bbox_inches='tight', dpi=400)

# plt.show()

# sys.exit()



# temp_label = np.loadtxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/temps_200years_zerodays.txt')
# plt.figure(figsize=(20,10))
# plt.title("Temp - Seasonal Cycle removed", fontsize=40)
# plt.xlabel("Years", fontsize=30)
# plt.ylabel("Temp Anamoly", fontsize=30)
# plt.plot(np.arange(0, len(temp_label),1)/365, temp_label)
# plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'data_examine.png'), bbox_inches='tight', dpi=400)
# plt.show()

# plt.show()

plt.clf()


train_years = 200
load_dir = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/'
# make labels
filename = 'small_2mtemp.nc'
# pres     = xr.open_dataset(load_dir+filename)['pr'].values[:,96:,80:241,np.newaxis] * 86400
# time     = xr.open_dataset(load_dir+filename)['time'].values[:train_years*365]
lats  = xr.open_dataset(load_dir+filename)['lat'].values[96:]
lons   = xr.open_dataset(load_dir+filename)['lon'].values[80:241]
print(lats[64])
print(180-(lons[88]-180))




print(lats)
print(lons)
exit()

# #number of days
# days = 365
# avgpres_day = []

# #Get the number of years to calculate an average psl per day
# # years = (pres.shape[0]/days)-train_years

# #Loop by day first.
# for i in np.arange(0,365,1):
#     pres_day = 0

#     #Then loop through the number of specified years. 
#     for j in np.arange(0,train_years,1):

#         #Add up the sea level pressure for that particular day from every year.
#         pres_day += pres[int(i+(days*j)), :,:,:]

#     #Divide it by the number of years.
#     pres_day /= train_years
#     avgpres_day.append(pres_day)

avgpres_day = xr.open_dataset("/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/200year_precip_cycle.nc")['200precipcycle'].values[:,:,:,np.newaxis]

# avgpres_day = avgpres_day.data_vars['200prescycle']

# avgpres_day = avgpres_day[:,:,:,np.newaxis]

# np.savetxt('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/data/avgpres_Day_200years.txt', avgpres_day, fmt='%d')

pres_c = []

full_years = 200

for i in np.arange(0,full_years*365,1):
    pres_c.append(pres[i,:,:,:] - avgpres_day[i%365, : ,:,:])

pres_c = np.asarray(pres_c)

#Calculate latitude factors
area_lats = []
for lat in lats:
    area_lats.append(np.sqrt(np.cos(np.deg2rad(lat))))

area_lats = np.asarray(area_lats)

print(lats)
print(area_lats)

print(area_lats.shape)

# for i in range(len(area_lats)):
#     pres_c[:,i,:,:] = pres_c[:,i,:,:] * area_lats[i]

# print(pres_c[0,:,:,0])

test_mean = []

for i in np.arange(0,full_years,1):
    test_mean.append(pres_c[(i*365)+200,len(lats)-1, 136, 0])
test_mean = np.asarray(test_mean)
print(test_mean.shape)
print("Mean: " + str(np.sum(test_mean)/test_mean.shape[0]))

bounds = 20

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
print(pres_c.shape)
img = sub1.contourf(np.asarray(lons), np.asarray(lats), np.asarray(pres_c[175,:,:,0])/100, np.linspace(-1, 1, 20), transform=ccrs.PlateCarree())
#img = sub1.contourf(np.asarray(lons), np.asarray(lats), np.asarray(pres_c[175,:,:,0])/100, np.arange(-60, 65, 5), transform=ccrs.PlateCarree())

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
cbar.set_label("hectopascals", fontsize=25)

sub1.coastlines()
# plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + 'example_anom.png'), bbox_inches='tight', dpi=400)

plt.show()

loc_pres = []
for i in range(pres_c.shape[0]):
    loc_pres.append(pres_c[i,len(lats)-1,136,0])

loc_pres = np.asarray(loc_pres)

plt.figure(figsize=(20,10))
plt.title("Arctic precipiation - Seasonal Cycle removed", fontsize=40)
plt.xlabel("Years", fontsize=30)
plt.ylabel("Precip Anamoly (mm/day)", fontsize=30)
plt.plot(np.arange(0, len(loc_pres),1)/365, loc_pres/100)
plt.xlim(0,5)
plt.savefig(('/Users/nicojg/Documents/Work/2021_Fall_IAI/Code/TLLTT/figures/' + 'zeroday_test' + '/vizualization/' + 'zeroday_test' + '5year_arctic_data_examine.png'), bbox_inches='tight', dpi=400)

plt.show()

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