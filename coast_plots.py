import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[2])

# # This Looks Like That There
# 
# Visualize the prototypes


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from icecream import ic          # pip install icecream
import scipy.io as sio

import xarray as xr

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
# import seaborn as sns
import cmasher as cmr            # pip install cmasher

import cartopy as ct
import cartopy.crs as ccrs



filename = '/ourdisk/hpc/ai2es/nicojg/TLLTT/data/small_2mtemp.nc'
file_lats  = xr.open_dataset(filename)['lat'].values[96:]
file_lons   = xr.open_dataset(filename)['lon'].values[80:241]

fig = plt.figure(figsize=(20, 16))
fig.tight_layout()

spec = fig.add_gridspec(4, 5)

plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

plt.set_cmap('cmr.copper')

all_coast_points = np.loadtxt("/ourdisk/hpc/ai2es/nicojg/TLLTT/data/all_coast_points.txt")

accuracy_20_loc = []
accuracy_100_loc = []
coast_locs = []
lons = []
lats = []
for coast_point in all_coast_points:
    accuracies = np.loadtxt("/ourdisk/hpc/ai2es/nicojg/TLLTT/data/accuracies/accuracies_"+str(int(coast_point[1]))+"_"+str(int(coast_point[0]))+".txt")
    accuracy_100_loc.append(accuracies[-1])
    accuracy_20_loc.append(accuracies[2])

    coast_locs.append((file_lats[int(coast_point[0])], file_lons[int(coast_point[1])]))
    lats.append(file_lats[int(coast_point[0])])
    lons.append(file_lons[int(coast_point[1])])



coast_locs = np.asarray(coast_locs)


grid_accuracy_100 = np.zeros((np.unique(file_lons).shape[0], np.unique(file_lats).shape[0]))
grid_accuracy_20 = np.zeros((np.unique(file_lons).shape[0], np.unique(file_lats).shape[0]))

print(grid_accuracy_100.shape)


for coast_ind in np.arange(0, coast_locs.shape[0], 1):
    for i in np.arange(0, grid_accuracy_100.shape[0]):
        for j in np.arange(0, grid_accuracy_100.shape[1]):
            if (file_lats[j] == coast_locs[coast_ind][0]) and (file_lons[i] == coast_locs[coast_ind][1]):
                grid_accuracy_100[i][j] = accuracy_100_loc[coast_ind]
                grid_accuracy_20[i][j] = accuracy_20_loc[coast_ind]

for i in np.arange(0, grid_accuracy_100.shape[0]):
    for j in np.arange(0, grid_accuracy_100.shape[1]):
        if(grid_accuracy_100[i][j] == 0.):
            grid_accuracy_100[i][j] = np.nan
        if(grid_accuracy_20[i][j] == 0.):
            grid_accuracy_20[i][j] = np.nan
            
grid_accuracy_diff = grid_accuracy_20 - grid_accuracy_100

[mesh_lats, mesh_lons] = np.meshgrid(file_lats,file_lons)


fig = plt.figure(figsize=(20, 16))
fig.tight_layout()

spec = fig.add_gridspec(4, 5)

plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

plt.set_cmap('cmr.copper')

sub1.coastlines()
img = sub1.scatter(mesh_lons, mesh_lats, s=80, c=grid_accuracy_100, cmap = 'copper', transform=ccrs.PlateCarree(), vmin = 33, vmax = 44, marker = 'o')

sub1.set_xticks(np.arange(-180,181,30))
sub1.xaxis.set_tick_params(labelsize=15)
sub1.yaxis.set_tick_params(labelsize=15)
sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
sub1.set_yticks(np.arange(-90,91,15))
sub1.set_xlim(0,80)
sub1.set_ylim(15,90)
sub1.set_xlabel("Longitude (degrees)",fontsize=25)
sub1.set_ylabel("Latitude (degrees)",fontsize=25)
cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
cbar.set_label("Accuracy %", fontsize=25)
cbar.ax.tick_params(labelsize=15) 
plt.title("100 - Coast Accuracy", fontsize = 30)
plt.savefig("/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/100_coast.png")


fig = plt.figure(figsize=(20, 16))
fig.tight_layout()

spec = fig.add_gridspec(4, 5)

plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

plt.set_cmap('cmr.copper')

sub1.coastlines()
img = sub1.scatter(mesh_lons, mesh_lats, s=80, c=grid_accuracy_20, cmap = 'copper', transform=ccrs.PlateCarree(), vmin = 33, vmax = 44, marker = 'o')

sub1.set_xticks(np.arange(-180,181,30))
sub1.xaxis.set_tick_params(labelsize=15)
sub1.yaxis.set_tick_params(labelsize=15)
sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
sub1.set_yticks(np.arange(-90,91,15))
sub1.set_xlim(0,80)
sub1.set_ylim(15,90)
sub1.set_xlabel("Longitude (degrees)",fontsize=25)
sub1.set_ylabel("Latitude (degrees)",fontsize=25)
cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
cbar.set_label("Accuracy %", fontsize=25)
cbar.ax.tick_params(labelsize=15) 
plt.title("20 -Coast Accuracy", fontsize = 30)
plt.savefig("/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/20_coast.png")


fig = plt.figure(figsize=(20, 16))
fig.tight_layout()

spec = fig.add_gridspec(4, 5)

plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

plt.set_cmap('cmr.copper')

sub1.coastlines()
img = sub1.scatter(mesh_lons, mesh_lats, s=80, c=grid_accuracy_diff, cmap = 'copper', transform=ccrs.PlateCarree(), marker = 'o')

sub1.set_xticks(np.arange(-180,181,30))
sub1.xaxis.set_tick_params(labelsize=15)
sub1.yaxis.set_tick_params(labelsize=15)
sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
sub1.set_yticks(np.arange(-90,91,15))
sub1.set_xlim(0,80)
sub1.set_ylim(15,90)
sub1.set_xlabel("Longitude (degrees)",fontsize=25)
sub1.set_ylabel("Latitude (degrees)",fontsize=25)
cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
cbar.set_label("Accuracy %", fontsize=25)
cbar.ax.tick_params(labelsize=15) 
plt.title("Diff Coast Accuracy", fontsize = 30)
plt.savefig("/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/diff_coast.png")


print(coast_locs)
print(accuracy_100_loc)

