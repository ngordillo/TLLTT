# # This Looks Like That There
# 
# Visualize the prototypes
#TODO: fix everything
import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import sys
import imp 

import numpy, warnings
numpy.warnings = warnings
import numpy as np
from tqdm import tqdm
from tqdm import trange
from icecream import ic          # pip install icecream
import scipy.io as sio

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
# import seaborn as sns
import cmasher as cmr            # pip install cmasher

import cartopy as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import tensorflow as tf

import random

from sklearn.metrics import confusion_matrix

# import network_temp_withfix as network
import network
# import experiment_settings_coast_550
import data_functions_schooner
import push_prototypes
import plots
import common_functions

import heapq as hq

import warnings


import matplotlib.font_manager
print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

warnings.filterwarnings( "ignore", module = "cartopy\..*" )
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "13 December 2021"

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
dpiFig = 400.

# ## Print the detailed system info

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"tensorflow version = {tf.__version__}")

np.set_printoptions(suppress=True)

if len(sys.argv) < 2:
    EXP_NAME = 'GCM_alas_lr_wint_550yrs_seed132'  #'GCM_alas_wint_550yrs_shuf_bal_seed125' #'GCM_alas_wint_550yrs_shuf_bal_seed117'#'smaller_test'#'quadrants_testcase'
    file_lon = 89
    file_lat = 64
    # import experiment_settings_shuf_550bal_seeds as experiment_settings
    import experiment_settings_multiple_seeds_lr as experiment_settings
elif len(sys.argv) == 2:
    num = int(sys.argv[1])
    #EXP_NAME = 'GCM_alas_lr_wint_drop_550yrs_seed'+str(num) #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'

    # learning_rate = float(sys.argv[1])
    # print(learning_rate)
    # EXP_NAME = 'GCM_alas_lr_wint_550yrs_seed144_nopre_lrtest_epochs'
    EXP_NAME = 'GCM_alas_lr_wint_550redo_seed'+str(num) #+ '_nopre' #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'
    import experiment_settings_multiple_seeds_lr_redo as experiment_settings
    # import experiment_settings_shuf_550bal_seeds as experiment_settings
    file_lon = 89
    file_lat = 64
else:
    file_lon = int(sys.argv[2])
    file_lat = int(sys.argv[1])
    EXP_NAME = 'GCM_'+ str(file_lon) + '_' + str(file_lat) +'_wint_550yrs_shuf_bal_seed134_redo'
    import experiment_settings_coast_550_lr_adjust_134_redo as experiment_settings

# EXP_NAME = 'GCM_alas_wint_550yrs_shuf_bal_seed117'

#'smaller_test'#'quadrants_testcase'

imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

imp.reload(common_functions)

# model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_schooner(EXP_NAME)
# model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories(EXP_NAME)
model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_falco(EXP_NAME)

# ## Define the network parameters

RANDOM_SEED          = settings['random_seed']
BATCH_SIZE_PREDICT   = settings['batch_size_predict']
BATCH_SIZE           = settings['batch_size']
NLAYERS              = settings['nlayers']
NFILTERS             = settings['nfilters']   
DOUBLE_CONV          = settings['double_conv']   
assert(len(NFILTERS)==NLAYERS)

NCLASSES             = settings['nclasses']
PROTOTYPES_PER_CLASS = settings['prototypes_per_class']
NPROTOTYPES          = np.sum(PROTOTYPES_PER_CLASS)

NEPOCHS              = settings['nepochs']
LR_INIT              = settings['lr']
LR_EPOCH_BOUND       = 10000
PATIENCE             = 100

STAGE                = settings['analyze_stage']

# ## Initialize
gpus = tf.config.list_physical_devices('GPU')

# the next line will restrict tensorflow to the first GPU 
# you can select other gpus from the list instead
tf.config.set_visible_devices(gpus[0], 'GPU')

tf.config.list_logical_devices('GPU')

tf.keras.backend.clear_session()
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

tf.config.experimental.enable_op_determinism()
# ## Get and process the data

imp.reload(data_functions_schooner)
DATA_NAME = settings['data_name']
DATA_DIR = settings['data_dir']                                                                                

train_yrs = settings['train_yrs']
val_yrs = settings['val_yrs']
test_years = settings['test_yrs']

train_yrs_era5 = settings['train_yrs_era5']
val_yrs_era5 = settings['val_yrs_era5']
test_years_era5 = settings['test_yrs_era5']

if(EXP_NAME[:3]=='ERA' or settings['plot_ERA5_convert']):   
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR, file_lon, file_lat, False)
    X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test = data_functions_schooner.get_and_process_tropic_data_winter_ERA5(labels,
                                                                                            data,
                                                                                            time,
                                                                                            rng,
                                                                                            train_yrs_era5,
                                                                                            val_yrs_era5,
                                                                                            test_years_era5,
                                                                                            temp_anoms,
                                                                                            translation = settings['plot_ERA5_convert'],
                                                                                            colored=settings['colored'],
                                                                                            standardize=settings['standardize'],
                                                                                            shuffle=settings['shuffle'],
                                                                                            bal_data = settings['balance_data'],
                                                                                            r_seed = RANDOM_SEED,
                                                                                        )
elif(EXP_NAME[:3] == 'GCM'):
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon  = data_functions_schooner.load_tropic_data_winter(DATA_DIR, file_lon, file_lat, False)
    X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test = data_functions_schooner.get_and_process_tropic_data_winter(labels,
                                                                                            data,
                                                                                            time,
                                                                                            rng, 
                                                                                            train_yrs,
                                                                                            val_yrs,
                                                                                            test_years,
                                                                                            temp_anoms,
                                                                                            colored=settings['colored'],
                                                                                            standardize=settings['standardize'],
                                                                                            shuffle=settings['shuffle'],
                                                                                            bal_data = settings['balance_data'],
                                                                                            r_seed = RANDOM_SEED,
                                                                                )
else:
    print("Expermient name is bad")
    quit()

# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)
# quit()
    

proto_class_mask = network.createClassIdentity(PROTOTYPES_PER_CLASS)

prototypes_of_correct_class_train = np.zeros((len(y_train),NPROTOTYPES))
for i in range(0,prototypes_of_correct_class_train.shape[0]):
    prototypes_of_correct_class_train[i,:] = proto_class_mask[:,int(y_train[i])]
    
prototypes_of_correct_class_val   = np.zeros((len(y_val),NPROTOTYPES))    
for i in range(0,prototypes_of_correct_class_val.shape[0]):
    prototypes_of_correct_class_val[i,:] = proto_class_mask[:,int(y_val[i])]

prototypes_of_correct_class_test   = np.zeros((len(y_test),NPROTOTYPES))    
for i in range(0,prototypes_of_correct_class_test.shape[0]):
    prototypes_of_correct_class_test[i,:] = proto_class_mask[:,int(y_test[i])]
    
###########################################################################################################################################################################
base_model_filename = model_dir + 'pretrained_model_' + EXP_NAME

if(EXP_NAME[:9] == 'GCM_SGold'):
    base_model_filename = './saved_models/GCM_alas_wint_583yrs_gold_redo/' + 'pretrained_model_' + 'GCM_alas_wint_583yrs_gold_redo'

if(settings['pretrain'] == True):

    base_model = common_functions.load_model(base_model_filename)


# ## Get the model and make predictions

model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(STAGE)
# model_filename = model_dir + str(learning_rate) + "_" + 'model_' + EXP_NAME + '_stage' + str(STAGE)
model = common_functions.load_model(model_filename)

model_final_conv_layer = network.get_model_final_conv_layer(model)
model_prototype_layer  = network.get_model_prototype_layer(model)

local_mask = np.exp(model.layers[-3].get_weights()[1])

# # Compute the prototypes

imp.reload(network)

model_cnn_only = network.build_model(
    nlayers=NLAYERS,
    nfilters=NFILTERS,
    input_shape= X_train.shape[1:],
    output_shape=NCLASSES,
    prototypes_per_class=PROTOTYPES_PER_CLASS,
    network_seed=RANDOM_SEED,
    cnn_only=True,
    double_conv=DOUBLE_CONV,    
)
# model_cnn_only.summary()

model_cnn_only = network.get_model_cnn_only_conv_layer(model_cnn_only)
# model_cnn_only.summary()

print('running receptive field calculation...')
receptive_field = network.ReceptiveField(model_cnn_only)
print('receptive field calculation complete.')

input_train      = [[X_train,prototypes_of_correct_class_train]]

# get the prototypes
imp.reload(push_prototypes)
model, push_info = push_prototypes.push(model, 
                                        input_train[0], 
                                        prototypes_of_correct_class_train,
                                        # perform_push=False,
                                        perform_push=True,
                                        batch_size=BATCH_SIZE_PREDICT,
                                        verbose=0,
                                    )
prototype_sample  = push_info[0]
prototype_indices = push_info[-1]
similarity_scores = push_info[-2]

print("Push Info:")


# print(prototype_sample.shape)
# print(type(prototype_sample))
# print(similarity_scores.shape)
# print(type(similarity_scores))
# print(similarity_scores)

np.savetxt(exp_data_dir + "_1_"+ EXP_NAME + 'viz_push_protos.txt', prototype_sample, fmt='%d')

era5_plots = (settings['plot_ERA5_translated'])

if(era5_plots):
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR, file_lon, file_lat, False)
    X_train, y_train, time_train_era5, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test = data_functions_schooner.get_and_process_tropic_data_winter_ERA5(labels,
                                                                                            data,
                                                                                            time,
                                                                                            rng,
                                                                                            train_yrs_era5,
                                                                                            val_yrs_era5,
                                                                                            test_years_era5,
                                                                                            temp_anoms,
                                                                                            translation = era5_plots,
                                                                                            colored=settings['colored'],
                                                                                            standardize=settings['standardize'],
                                                                                            shuffle=settings['shuffle'],
                                                                                            bal_data = settings['balance_data'],
                                                                                            r_seed = RANDOM_SEED,
                                                                                        )

    proto_class_mask = network.createClassIdentity(PROTOTYPES_PER_CLASS)

    prototypes_of_correct_class_train = np.zeros((len(y_train),NPROTOTYPES))
    for i in range(0,prototypes_of_correct_class_train.shape[0]):
        prototypes_of_correct_class_train[i,:] = proto_class_mask[:,int(y_train[i])]
        
    prototypes_of_correct_class_val   = np.zeros((len(y_val),NPROTOTYPES))    
    for i in range(0,prototypes_of_correct_class_val.shape[0]):
        prototypes_of_correct_class_val[i,:] = proto_class_mask[:,int(y_val[i])]

    prototypes_of_correct_class_test   = np.zeros((len(y_test),NPROTOTYPES))    
    for i in range(0,prototypes_of_correct_class_test.shape[0]):
        prototypes_of_correct_class_test[i,:] = proto_class_mask[:,int(y_test[i])]
else:
    era5_plots = (settings['plot_ERA5_convert'])

###################################################################################################################
# prototype_sample = np.loadtxt(exp_data_dir + EXP_NAME + 'final_push_protos.txt').astype(int)
# prototype_indices = np.loadtxt(exp_data_dir + EXP_NAME + 'final_protos_loc.txt').astype(int)
# similarity_scores = np.load(exp_data_dir + EXP_NAME + 'similarity_scores.npy')
###################################################################################################################


# print("Load Info:")
# print(prototype_sample.shape)
# print(type(prototype_sample))
# print(similarity_scores.shape)
# print(type(similarity_scores))
# print(prototype_indices.shape)
# print(prototype_indices)


# prototype_date = time_train.dt.strftime("%b %d %Y").values[prototype_sample]    

model.summary()

# ## Validation samples

input_val  = [[X_val,prototypes_of_correct_class_val]]

# print('running model.predict()...')
y_predict_val = model.predict(input_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)
# print('model.predict() complete.')

model.evaluate(input_val,y_val,batch_size=BATCH_SIZE_PREDICT, verbose=1)

# print('Accuracies by class: ')

# for c in np.arange(0,NCLASSES):
#     i = np.where(y_val==c)[0]
#     j = np.where(y_val[i]==np.argmax(y_predict_val[i],axis=1))[0]
#     acc = np.round(len(j)/len(i),3)
#     print(np.argmax(y_predict_val[i],axis=1))
    
#     print('   phase ' + str(c) + ' = ' + str(acc))
    

# #-------------
# y_predict  = y_predict_val
# y_true     = y_val
# # time       = time_val
# input_data = input_val
# #-------------

## Testing samples

if(settings['plot_ERA5_convert'] == True):
    
# print(np.unique(prototype_sample))
# print(prototype_sample)
# print(X_train[np.unique(prototype_sample)].shape)
# print(X_train.shape)
    X_test=np.delete(X_test,np.unique(prototype_sample), axis = 0)
    
    y_test = np.delete(y_test,np.unique(prototype_sample), axis = 0)

    nino_time_test = np.delete(time_test,np.unique(prototype_sample), axis = 0)
    
    prototypes_of_correct_class_test   = np.zeros((len(y_test),NPROTOTYPES))    
    for i in range(0,prototypes_of_correct_class_test.shape[0]):
        prototypes_of_correct_class_test[i,:] = proto_class_mask[:,int(y_test[i])]
    
# print(X_train.shape)
# print(prototype_indices)

input_test  = [[X_test,prototypes_of_correct_class_test]]

print('running model.predict()...')
y_predict_test = model.predict(input_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)
# print(y_predict_test)
print('model.predict() complete.')

model.evaluate(input_test,y_test,batch_size=BATCH_SIZE_PREDICT, verbose=1)

print('Accuracies by class: ')

if(settings['plot_ERA5_convert'] == True):
    nino_y_test = []
    # print(time_test[0].values)
    # print(time_test[0].astype('datetime64[Y]').astype(int) + 1970)

    # print(time_test.values.tolist())
    era_years = [x.year for x in nino_time_test.values.astype("datetime64[D]").tolist()]
    era_months = [x.month for x in nino_time_test.values.astype("datetime64[D]").tolist()]
    # print(era_months)
    nino_table = data_functions_schooner.process_nino_data(DATA_DIR)


    for i in np.arange(0, len(era_years), 1):
        nino_index = nino_table[era_years[i]-1870][era_months[i]]
        if(nino_index < -.4):
            nino_y_test.append(0)
        elif(nino_index > .4):
            nino_y_test.append(2)
        else:
            nino_y_test.append(1)
    print(len(nino_y_test))
    print(len(y_test))

    nino_test_accur = []
    for i in np.arange(0,len(y_test), 1):
        if(nino_y_test[i] == y_test[i]):
            nino_test_accur.append(1)
        else:
            nino_test_accur.append(0)

    nino_final_accur = np.sum(nino_test_accur)/len(nino_test_accur)

did_not_train = False


for c in np.arange(0,NCLASSES):
    i = np.where(y_test==c)[0]
    j = np.where(y_test[i]==np.argmax(y_predict_test[i],axis=1))[0]
    acc = np.round(len(j)/len(i),3)

    if(acc >= .995):
        did_not_train = True
        # quit()
    print(np.argmax(y_predict_test[i],axis=1))
    
    print('   phase ' + str(c) + ' = ' + str(acc))

######################################################################################################################################################################
if(settings['pretrain'] == True):
    print('running base_model.predict()...')
    base_y_predict_test = base_model.predict(X_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)
    base_y_predict_val = base_model.predict(X_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

    # print(y_predict_test)
    print('base_model.predict() complete.')

    base_model.evaluate(X_test,y_test,batch_size=BATCH_SIZE_PREDICT, verbose=1)

    print('Base CNN Accuracies by class: ')

    for c in np.arange(0,NCLASSES):
        i = np.where(y_test==c)[0]
        j = np.where(y_test[i]==np.argmax(base_y_predict_test[i],axis=1))[0]
        acc = np.round(len(j)/len(i),3)
        print(np.argmax(base_y_predict_test[i],axis=1))
        
        print('   phase ' + str(c) + ' = ' + str(acc))

    base_y_predict = base_y_predict_test

######################################################################################################################################################################


#-------------
y_predict  = y_predict_test
y_true     = y_test
# time       = time_val
input_data = input_test
#-------------

imp.reload(push_prototypes)

# get similarity maps
inputs_to_prototype_layer = model_final_conv_layer.predict(input_data)
prototypes = model.layers[-3].get_weights()[0]
similarity_scores = push_prototypes.get_similarity_maps(inputs_to_prototype_layer, 
                                                        prototypes, 
                                                        local_mask,
                                                        batch_size=BATCH_SIZE_PREDICT,
                                                    )
# get winning similarity scores across maps for each prototype and sample
max_similarity_score = model_prototype_layer.predict(input_data)

# get final weights
w = np.round(model.layers[-2].get_weights()[0],3)

# # Plot Prototypes and Samples

### for white background...
plt.rc('text',usetex=False)
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']}) 

# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) 
plt.rc('savefig',facecolor='white')
plt.rc('axes',facecolor='white')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('xtick',color='dimgrey')
plt.rc('ytick',color='dimgrey')
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
            
# adjust_spines(ax, ['left', 'bottom'])
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('dimgrey')
# ax.spines['bottom'].set_color('dimgrey')
# ax.spines['left'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)            



# imp.reload(data_functions)

# full_loc_temp, avgtemp_day, temp = data_functions.get_raw_temp_data(DATA_DIR)

##################################################################################################################################################################################################################

prototype_date = time_train.dt.strftime("%b %d %Y").values[prototype_sample]
if(settings['plot_ERA5_convert'] == True):
    prototype_date = time_val.dt.strftime("%b %d %Y").values[prototype_sample]
sample_date = time_test.dt.strftime("%b %d %Y").values



##################################################################################################################################################################################################################

def make_confuse_matrix(y_predict, y_test, data_amount, base):
#Generate the confusion matrix

    y_predict_class = np.argmax(y_predict,axis=1)

    cf_matrix = confusion_matrix(y_test, y_predict_class)
    cf_matrix_pred = confusion_matrix(y_test, y_predict_class, normalize='pred')
    cf_matrix_true = confusion_matrix(y_test, y_predict_class, normalize='true')
    cf_matrix = np.around(cf_matrix,3)
    cf_matrix_pred = np.around(cf_matrix_pred,3)
    cf_matrix_true = np.around(cf_matrix_true,3)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf_matrix, cmap=plt.cm.Blues, alpha=0.3)

    correct_preds = 0
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            ax.text(x=j, y=i,s=cf_matrix[i, j], va='center', ha='center', size='xx-large')
            ax.text(x=j, y=i+.3,s=(str(np.around(cf_matrix_pred[i, j]*100,4))+'%'), va='center', ha='center', size='xx-large', color = 'green')
            ax.text(x=j, y=i-.3,s=(str(np.around(cf_matrix_true[i, j]*100,4))+'%'), va='center', ha='center', size='xx-large', color = 'red')

            if (i == j):
                correct_preds += cf_matrix[i, j]

    correct_preds /= np.sum(cf_matrix)
    
    plt.xlabel('Prediction', fontsize=18, color = 'green')
    plt.ylabel('Actual', fontsize=18, color = 'red')
    plt.title('TLLTT Confusion Matrix (Accuracy - ' + str(np.around(correct_preds*100,2)) + '%)', fontsize=18)
    if (base):
        plt.savefig((vizualization_dir + "confusion_matrices/" + EXP_NAME + 'base_'+ str(data_amount) + 'percent_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    else:
        plt.savefig((vizualization_dir + "confusion_matrices/" + EXP_NAME + 'TLLTT_'+ str(data_amount) + 'percent_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()




    # plt.show()

    return np.around(correct_preds*100,2)
# y_predict_class_plot = np.argmax(y_predict,axis=1)
# y_predict_class_plot_low = y_predict_class_plot.astype('float64')
# y_predict_class_plot_avg = y_predict_class_plot.astype('float64')
# y_predict_class_plot_high = y_predict_class_plot.astype('float64')

# y_predict_class_plot_low[np.where(y_predict_class_plot_low==2)[0]] = np.nan
# y_predict_class_plot_low[np.where(y_predict_class_plot_low==1)[0]] = np.nan

# y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==2)[0]] = np.nan
# y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==0)[0]] = np.nan

# y_predict_class_plot_high[np.where(y_predict_class_plot_high==1)[0]] = np.nan
# y_predict_class_plot_high[np.where(y_predict_class_plot_high==0)[0]] = np.nan

# plt.figure(figsize=(20,6))
# plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/365,y_predict_class_plot_low, s=1 )
# plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/365,y_predict_class_plot_avg, s=1 )
# plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/365,y_predict_class_plot_high, s=1 )
# plt.yticks([0,1,2])
# plt.title("Model Class by Year", fontsize=20)
# plt.xlabel("Year", fontsize=15)
# plt.ylabel("Class", fontsize=15)
# plt.xlim(0,50)
# plt.show()

##################################################################################################################################################################################################################

def mjo_lookup(best_samps, samps_bool, percent_samps = 100):

    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation
    for temp_class in [0,1,2]:

        phases = MJO_info['Phase']
        rmm1 = MJO_info['RMM1']
        rmm2 = MJO_info['RMM2']

        mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

        less_than_one = np.where(mjo_amp < 1)[0]

        phases[less_than_one] = 0

        # print(less_than_one)

        if(samps_bool):
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp])==temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = temp_best_samps
            # isamples = np.where((np.argmax(y_predict[best_samps],axis=1)==temp_class))[0]
            # print("this ran")
            # print(np.argmax(y_predict[0]))
            # print(int(np.argmax(y_predict[0])))
        else:
            isamples = np.where((np.argmax(y_predict,axis=1)==temp_class) & (np.argmax(y_predict,axis=1)==y_true))[0]





        # print("compare")
        # print(best_samps)
        # print(isamples)
        corr_phases = phases[isamples]

        isamples_full = np.where((np.argmax(y_predict,axis=1)==temp_class))[0]

        corr_phases_full = phases[isamples_full]

        name_phases = ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7", "Phase 8"]

        num_phases = []
        for i in np.arange(0,9,1):
            num_phases.append( np.count_nonzero(corr_phases == i))
        num_phases = np.asarray(num_phases)

        num_phases_full = []
        for i in np.arange(0,9,1):
            num_phases_full.append( np.count_nonzero(corr_phases_full == i))
        num_phases_full = np.asarray(num_phases_full)

        # print("anotha one")
        # print(num_phases_full.shape[0])
        # print(num_phases.shape[0])

        # print(num_phases_full)
        # print(num_phases)

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )

        plt.title(str(percent_samps) + "percent - Class " + str(temp_class) +" by phase (n=" +str(np.sum(num_phases)) + ")")
        plt.bar(name_phases, num_phases/np.sum(num_phases))
        plt.savefig((vizualization_dir + "mjo_figs/" + "_" + str(percent_samps) + "_" + 'phase' + str(temp_class)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Class " + str(temp_class) +" by phase (n=" +str(np.sum(num_phases)) + ")")
        # print("third test")
        # print(num_phases/num_phases_full)
        # print(num_phases)
        plt.bar(name_phases, num_phases/np.sum(num_phases) - num_phases_full/np.sum(num_phases_full))
        plt.savefig((vizualization_dir + "mjo_figs/" + "_" + str(percent_samps) + "_" + 'percent_phase' + str(temp_class)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)

        # plt.show()

        samp_proto_match = [0,0,0,0,0,0,0,0,0,0]

        num_proto = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

        for sample in isamples:

            y_predict_class = int(np.argmax(y_predict[sample]))
            # print("oofie")
            # print(sample)
            # print(y_predict_class)
            points = max_similarity_score[sample,:]*w[:,y_predict_class]
            # print(np.max(points))
            all_points = w*np.squeeze(max_similarity_score[sample,:])[:,np.newaxis]
            total_points = np.sum(all_points,axis=0)        


            #----------------------   

            prototype_points = np.sort(points)[-1]
            # for golf in np.arange(1,30):
            #     print(np.sort(points)[-golf])

            prototype_index = np.where(points == prototype_points)[0][0]
            # print(prototype_index)
            prototype_class = np.argmax(proto_class_mask[prototype_index])
            if(y_predict_class != prototype_class):
                print_warning = '\n- prototype not associated with predicted class -'
            else:
                print_warning = ''
            # print(prototype_class)



            phase_proto = phases[prototype_sample[prototype_index]]


            # print("proto index")
            # print(prototype_index)
            if(temp_class != 0):
                num_proto[prototype_index%(temp_class*10)] += 1
            else:
                num_proto[prototype_index] += 1

            phase_samp = phases[sample]


            #Make a range of phases
            close_phase_samp = []

            if(phase_samp != 0):
                
                close_phase_samp.append(phase_samp)

                #One phase lower
                if(phase_samp == 1):
                    close_phase_samp.append(8)
                else: 
                    close_phase_samp.append(phase_samp - 1)

                #One phase higher
                if(phase_samp == 8):
                    close_phase_samp.append(1)
                else: 
                    close_phase_samp.append(phase_samp + 1)

                close_phase_samp = np.asarray(close_phase_samp)
            else:
                close_phase_samp = np.asarray([0,0,0])
                



            if(phase_proto in close_phase_samp and (phase_proto != 0.0 and phase_samp != 0.0)):
                # print("bruh test")
                if(temp_class != 0):
                    samp_proto_match[prototype_index%(temp_class*10)] += 1
                else:
                    samp_proto_match[prototype_index] += 1

        # print(samp_proto_match)

        # print("num_proto: " + str(num_proto))

        num_proto = np.asarray(num_proto)

        name_proto = ["Proto" +str(temp_class)+ "0", "Proto" +str(temp_class)+ "1", "Proto" +str(temp_class)+ "2", "Proto" +str(temp_class)+ "3", "Proto" +str(temp_class)+ "4", "Proto" +str(temp_class)+ "5", "Proto" +str(temp_class)+ "6", "Proto" +str(temp_class)+ "7", "Proto" +str(temp_class)+ "8", "Proto" +str(temp_class)+ "9"]

        name_proto = np.asarray(name_proto)

        percent_corr = samp_proto_match/num_proto
        percent_corr[np.isnan(percent_corr)] = 0
        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Number of samples associated with prototype (class " + str(temp_class)+ ")")
        plt.bar(name_proto, num_proto)


        # print(phases[prototype_sample])
        weak_protos = []
        for i in np.arange(temp_class*10, (temp_class*10)+10, 1):
            phase_proto = phases[prototype_sample[i]]

            if phase_proto == 0:
                weak_protos.append(i)

        # print(weak_protos)
        for i in weak_protos:
            # print((i*2)%(temp_class*10))
            # print((i%(temp_class*10))*2)
            if(temp_class != 0):
                axs.xaxis.get_ticklines()[(i%(temp_class*10))*2].set_markeredgecolor("red")
            else:
                axs.xaxis.get_ticklines()[(i*2)].set_markeredgecolor("red")
            
        plt.savefig((vizualization_dir + "mjo_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'protos_phase'+str(temp_class)+'_mjo.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Percentage of time prototype and sample mjo match (class " + str(temp_class)+ ")")
        plt.bar(name_proto, percent_corr)


        for i in weak_protos:
            # print((i*2)%(temp_class*10))
            if(temp_class != 0):
                axs.xaxis.get_ticklines()[(i%(temp_class*10))*2].set_markeredgecolor("red")
            else:
                axs.xaxis.get_ticklines()[(i*2)].set_markeredgecolor("red")

        plt.savefig((vizualization_dir + "mjo_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'match_phase'+str(temp_class)+'_mjo.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()
##################################################################################################################################################################################################################

def enso_lookup(best_samps, samps_bool, percent_samps = 100):

    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation
    for temp_class in [0,1,2]:

        phases = np.loadtxt(DATA_DIR + 'ENSO_200_years.txt')

        # print(less_than_one)

        if(samps_bool):
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp])==temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = temp_best_samps
            # isamples = np.where((np.argmax(y_predict[best_samps],axis=1)==temp_class))[0]
            # print("this ran")
            # print(np.argmax(y_predict[0]))
            # print(int(np.argmax(y_predict[0])))
        else:
            isamples = np.where((np.argmax(y_predict,axis=1)==temp_class) & (np.argmax(y_predict,axis=1)==y_true))[0]





        # print("compare")
        # print(best_samps)
        # print(isamples)
        corr_phases = phases[isamples]

        isamples_full = np.where((np.argmax(y_predict,axis=1)==temp_class))[0]

        corr_phases_full = phases[isamples_full]

        name_phases = ["La Nina", "Neutral", "El Nino"]

        num_phases = []
        for i in np.arange(-1,2,1):
            num_phases.append( np.count_nonzero(corr_phases == i))
        num_phases = np.asarray(num_phases)

        num_phases_full = []
        for i in np.arange(-1,2,1):
            num_phases_full.append( np.count_nonzero(corr_phases_full == i))
        num_phases_full = np.asarray(num_phases_full)

        # print("anotha one")
        # print(num_phases_full.shape[0])
        # print(num_phases.shape[0])

        # print(num_phases_full)
        # print(num_phases)

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )

        plt.title(str(percent_samps) + "percent - Class " + str(temp_class) +" by state (n=" +str(np.sum(num_phases)) + ")")
        plt.bar(name_phases, num_phases/np.sum(num_phases))
        plt.savefig((vizualization_dir + "enso_figs/" + "_" + str(percent_samps) + "_" + 'state_' + str(temp_class)+ '.png'), bbox_inches='tight', dpi=dpiFig)

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Class " + str(temp_class) +" by state (n=" +str(np.sum(num_phases)) + ")")
        # print("third test")
        # print(num_phases/num_phases_full)
        # print(num_phases)
        plt.bar(name_phases, num_phases/np.sum(num_phases) - num_phases_full/np.sum(num_phases_full))
        plt.savefig((vizualization_dir + "enso_figs/" + "_" + str(percent_samps) + "_" + 'percent_state_' + str(temp_class)+ '.png'), bbox_inches='tight', dpi=dpiFig)

        # plt.show()

        samp_proto_match = [0,0,0,0,0,0,0,0,0,0]

        num_proto = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

        for sample in isamples:

            y_predict_class = int(np.argmax(y_predict[sample]))
            # print("oofie")
            # print(sample)
            # print(y_predict_class)
            points = max_similarity_score[sample,:]*w[:,y_predict_class]
            # print(np.max(points))
            all_points = w*np.squeeze(max_similarity_score[sample,:])[:,np.newaxis]
            total_points = np.sum(all_points,axis=0)        


            #----------------------   

            prototype_points = np.sort(points)[-1]
            # for golf in np.arange(1,30):
            #     print(np.sort(points)[-golf])

            prototype_index = np.where(points == prototype_points)[0][0]
            # print(prototype_index)
            prototype_class = np.argmax(proto_class_mask[prototype_index])
            if(y_predict_class != prototype_class):
                print_warning = '\n- prototype not associated with predicted class -'
            else:
                print_warning = ''
            # print(prototype_class)



            phase_proto = phases[prototype_sample[prototype_index]]

            if(temp_class != 0):
                num_proto[prototype_index%(temp_class*10)] += 1
            else:
                num_proto[prototype_index] += 1

            phase_samp = phases[sample]


            if(phase_proto  == phase_samp):
                # print("bruh test")
                if(temp_class != 0):
                    samp_proto_match[prototype_index%(temp_class*10)] += 1
                else:
                    samp_proto_match[prototype_index] += 1

        # print(samp_proto_match)

        # print("num_proto: " + str(num_proto))

        num_proto = np.asarray(num_proto)

        name_proto = ["Proto" +str(temp_class)+ "0", "Proto" +str(temp_class)+ "1", "Proto" +str(temp_class)+ "2", "Proto" +str(temp_class)+ "3", "Proto" +str(temp_class)+ "4", "Proto" +str(temp_class)+ "5", "Proto" +str(temp_class)+ "6", "Proto" +str(temp_class)+ "7", "Proto" +str(temp_class)+ "8", "Proto" +str(temp_class)+ "9"]

        name_proto = np.asarray(name_proto)

        percent_corr = samp_proto_match/num_proto
        percent_corr[np.isnan(percent_corr)] = 0
        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Number of samples associated with prototype (class " + str(temp_class)+ ")")
        plt.bar(name_proto, num_proto)


        # # print(phases[prototype_sample])
        # weak_protos = []
        # for i in np.arange(temp_class*10, (temp_class*10)+10, 1):
        #     phase_proto = phases[prototype_sample[i]]

        #     if phase_proto == 0:
        #         weak_protos.append(i)

        # # print(weak_protos)
        # for i in weak_protos:
        #     # print((i*2)%(temp_class*10))
        #     # print((i%(temp_class*10))*2)
        #     if(temp_class != 0):
        #         axs.xaxis.get_ticklines()[(i%(temp_class*10))*2].set_markeredgecolor("red")
        #     else:
        #         axs.xaxis.get_ticklines()[(i*2)].set_markeredgecolor("red")
            
        plt.savefig((vizualization_dir + "enso_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'protos_state'+str(temp_class)+'.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()

        fig, axs = plt.subplots(1,
                                1, 
                                figsize=(21/2.5,12/2.5)
                            )
            
        plt.title(str(percent_samps) + "percent - Percentage of time prototype and sample state match (class " + str(temp_class)+ ")")
        plt.bar(name_proto, percent_corr)


        # for i in weak_protos:
        #     print((i*2)%(temp_class*10))
        #     if(temp_class != 0):
        #         axs.xaxis.get_ticklines()[(i%(temp_class*10))*2].set_markeredgecolor("red")
        #     else:
        #         axs.xaxis.get_ticklines()[(i*2)].set_markeredgecolor("red")

        plt.savefig((vizualization_dir + "enso_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'match_state_'+str(temp_class)+'.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()
##################################################################################################################################################################################################################

def precip_comps(best_samps, samps_bool, percent_samps = 100):
    for temp_class in np.arange(0,3,1):
        if(samps_bool):
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp])==temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = np.asarray(temp_best_samps)
        else:
            isamples = np.where((np.argmax(y_predict,axis=1)==temp_class) & (np.argmax(y_predict,axis=1)==y_true))[0]
        
        if(isamples.shape[0] > 0):
            img = 0
            for i in np.arange(isamples.shape[0]):
                img += np.squeeze(input_data[0][0][isamples[i],:,:,0])
            img = img / isamples.shape[0]
            # img = np.squeeze(input_data[0][0][isamples[40],:,:,0])
            # print(img.shape)

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

            plt.set_cmap('cmr.fusion')
            # sub1.plot(46,136,'go')
            img = sub1.contourf(np.asarray(lon), np.asarray(lat), img, np.linspace(-.75, .75, 41), transform=ccrs.PlateCarree())
            # plt.xticks(np.arange(-180,181,30), np.concatenate((np.arange(0,181,30),np.arange(-160,1,30)), axis = None))
            # sub1.set_xticks(np.arange(-180,181,30), np.arange(-180,181,30))
            sub1.set_xticks(np.arange(-180,181,30))
            sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
            sub1.set_yticks(np.arange(-90,91,15))
            sub1.set_xlim(-140,120)
            sub1.set_ylim(-30,30)
            sub1.set_xlabel("Longitude (degrees)",fontsize=25)
            sub1.set_ylabel("Latitude (degrees)",fontsize=25)
            sub1.set_title(str(percent_samps) + " percent - Tropical Precipitation (Class " + str(temp_class) + ")",fontsize=25)
            cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)
            cbar.set_label("mm/day", fontsize=25)

            sub1.coastlines()
            plt.savefig((vizualization_dir + "precip_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'PRECIPphase' + str(temp_class)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)

# plt.show()

def timeseries_predictions():
    y_predict_class_plot = np.argmax(y_predict,axis=1)
    y_predict_class_plot_low = y_predict_class_plot.astype('float64')
    y_predict_class_plot_avg = y_predict_class_plot.astype('float64')
    y_predict_class_plot_high = y_predict_class_plot.astype('float64')

    y_predict_class_plot_low[np.where(y_predict_class_plot_low==2)[0]] = np.nan
    y_predict_class_plot_low[np.where(y_predict_class_plot_low==1)[0]] = np.nan

    y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==2)[0]] = np.nan
    y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==0)[0]] = np.nan

    y_predict_class_plot_high[np.where(y_predict_class_plot_high==1)[0]] = np.nan
    y_predict_class_plot_high[np.where(y_predict_class_plot_high==0)[0]] = np.nan
    
    for decade in np.arange(10, 120,10):
        plt.figure(figsize=(20,6))
        plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
        plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
        plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
        plt.yticks([0,1,2])
        plt.title("Model Class by Year"  + str(decade), fontsize=20)
        plt.xlabel("Year", fontsize=15)
        plt.ylabel("Class", fontsize=15)
        plt.xlim(decade-10,decade)
        plt.savefig((vizualization_dir + "timeseries/" + EXP_NAME + 'decade_timeseries_' + str(decade)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()
        plt.close()

    for half_decade in np.arange(5, 120,5):
        plt.figure(figsize=(20,6))
        plt.scatter(np.arange(1,len(y_predict_class_plot_low)+1,1)/120,y_predict_class_plot_low, s=1 )
        plt.scatter(np.arange(1,len(y_predict_class_plot_avg)+1,1)/120,y_predict_class_plot_avg, s=1 )
        plt.scatter(np.arange(1,len(y_predict_class_plot_high)+1,1)/120,y_predict_class_plot_high, s=1 )
        plt.yticks([0,1,2])
        plt.title("Model Class by Year: " + str(half_decade), fontsize=20)
        plt.xlabel("Year", fontsize=15)
        plt.ylabel("Class", fontsize=15)
        plt.xlim(half_decade-5,half_decade)
        plt.savefig((vizualization_dir + "timeseries/" + EXP_NAME + 'half_decade_timeseries_' + str(half_decade)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.show()
        plt.close()

##################################################################################################################################################################################################################
def top_points_protos():
    phase = 0
    isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]
    # print(isamples)
    # print(max_similarity_score[isamples,:].shape)
    points = max_similarity_score[isamples,:]*w[:,phase]
    # print(points.shape)
    k = np.where(proto_class_mask[:,phase]==0)[0]
    points[:,k] = 0.
    # print(points.shape)
    # print(points)
    # print(hq.nlargest(10, points))
    maxvals = np.max(points,axis=1)
    # print(hq.nlargest(10,maxvals))

    topscore = np.asarray(hq.nlargest(10, maxvals))

    # print(np.where(np.in1d(maxvals,topscore))[0])

    toplocs = np.where(np.in1d(maxvals,topscore))[0]

    # print(maxvals[toplocs])
    # print(np.argsort(maxvals[toplocs]))

    argstest = np.argsort(maxvals[toplocs])

    # print(toplocs[argstest][::-1])

    # print(maxvals[toplocs][argstest][::-1])

    # print(isamples[toplocs[argstest][::-1]])

    bestsamps = isamples[toplocs[argstest][::-1]]
    # print("best samps:"  + str(bestsamps))
    return bestsamps
##################################################################################################################################################################################################################

def top_confidence_protos(percentage, predictions):
    #temp_classes = [0,1]
    temp_classes = [0,1,2]

    total_maxvals = []
    all_isamples = []
    for temp_class in temp_classes:

        isamples = np.where((np.argmax(predictions,axis=1)==temp_class))[0]
        # print(isamples)
        # print(max_similarity_score[isamples,:].shape)
        
        high_scores = predictions[isamples,temp_class]
        # print(points.shape)
        
        # print(points.shape)
        # print(points)
        # print(hq.nlargest(10, points))
        maxvals = high_scores
        # print(hq.nlargest(10,maxvals))

        total_maxvals.append(maxvals)

        all_isamples.append(isamples)

        # print("BIG TESTTTTTTTTTT:  =" + str(maxvals.shape[0]))

    total_maxvals = np.concatenate((total_maxvals[0], total_maxvals[1], total_maxvals[2]))

    total_maxvals = np.asarray(total_maxvals)

    all_isamples = np.concatenate((all_isamples[0], all_isamples[1], all_isamples[2]))

    all_isamples = np.asarray(all_isamples)
    
    topscore = np.asarray(hq.nlargest(int(total_maxvals.shape[0]*percentage), total_maxvals))

    # print(topscore)

    # print(total_maxvals)

    # print(np.where(np.in1d(maxvals,topscore))[0])

    toplocs = np.where(np.in1d(total_maxvals,topscore))[0]

    # print(maxvals[toplocs])
    # print(np.argsort(maxvals[toplocs]))

    argstest = np.argsort(total_maxvals[toplocs])

    # print(toplocs[argstest][::-1])

    # print(maxvals[toplocs][argstest][::-1])

    # print(isamples[toplocs[argstest][::-1]])

    # print(total_maxvals[toplocs[argstest[-1]]])
    # quit()
    bestsamps = all_isamples[toplocs[argstest][::-1]]
    # print("best samps:"  + str(bestsamps))
    return bestsamps


##################################################################################################################################################################################################################

def top_30per_scoring_protos_softmax():
    temp_classes = [0,1]

    total_maxvals = []
    all_isamples = []
    for temp_class in temp_classes:

        isamples = np.where((np.argmax(y_predict,axis=1)==temp_class))[0]
        # print(isamples)
        # print(max_similarity_score[isamples,:].shape)
        high_scores = y_predict[isamples,temp_class]

        # print("SCORES TEST:  #############################")
        # print(high_scores.shape)
        # print(high_scores)
        
        maxvals = high_scores

        total_maxvals.append(maxvals)

        all_isamples.append(isamples)

        # print("BIG TESTTTTTTTTTT:  =" + str(maxvals.shape[0]))

    total_maxvals = np.concatenate((total_maxvals[0], total_maxvals[1]))

    total_maxvals = np.asarray(total_maxvals)

    all_isamples = np.concatenate((all_isamples[0], all_isamples[1]))

    all_isamples = np.asarray(all_isamples)
    
    topscore = np.asarray(hq.nlargest(int(total_maxvals.shape[0]*.3), total_maxvals))

    # print(np.where(np.in1d(maxvals,topscore))[0])

    toplocs = np.where(np.in1d(total_maxvals,topscore))[0]

    # print(maxvals[toplocs])
    # print(np.argsort(maxvals[toplocs]))

    argstest = np.argsort(total_maxvals[toplocs])

    # print(toplocs[argstest][::-1])

    # print(maxvals[toplocs][argstest][::-1])

    # print(isamples[toplocs[argstest][::-1]])

    bestsamps = all_isamples[toplocs[argstest][::-1]]
    # print("best samps:"  + str(bestsamps))
    return bestsamps
##################################################################################################################################################################################################################

# Prototype examples
def examine_proto(good_samp, era5_flag):
    # print(lon)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    imp.reload(plots)

    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation

    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']

    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

    less_than_one = np.where(mjo_amp < 1)[0]
    # print("test")
    # print(less_than_one)
    phases[less_than_one] = 0
    # print("help us")
    y_predict_class = np.argmax(y_predict,axis=1)
    igrab_samples = np.where(y_predict_class==1)[0]    

    # for phase in [0,1,2]:
    #     isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]
    #     points = max_similarity_score[isamples,:]*w[:,phase]
    #     k = np.where(proto_class_mask[:,phase]==0)[0]
    #     points[:,k] = 0.

    #     maxvals = np.max(points,axis=1)

    #     topscore = np.asarray(hq.nlargest(10, maxvals))

    #     toplocs = np.where(np.in1d(maxvals,topscore))[0]

    #     argstest = np.argsort(maxvals[toplocs])

    #     bestsamps = isamples[toplocs[argstest][::-1]]


    # phase 1 = 6758, var = 1
    # phase 4 = 1159, var = 2
    # phase 7 = 6231, var = 0
    # print(y_predict.shape)
#     3087, 2523
    # SAMPLES = bestsamps
    SAMPLES = [good_samp] #4055 %4570
    # SAMPLES = (2523, 1159, 265) #6231
#     SAMPLES = (2200,2200,2200) #1159, 1904
#     SAMPLES = np.random.choice(igrab_samples,size=3)
    
    # VAR_INDEX = (1,2,0)
    VAR_INDEX = [0]
    #SORTED_VALUE = (1,1,1)   #(1,2,1) 
    SORTED_VALUE = (1,1,1)
    colors = ('tab:purple','tab:orange')
    FS = 13

    #------------------------------
    fig = plt.figure(figsize=(10,9.5), constrained_layout=True)
    grid_per_col = 7
    spec = gridspec.GridSpec(ncols=3*grid_per_col, nrows=5, figure=fig)

    for isample, sample in enumerate(SAMPLES):

        y_predict_class = int(np.argmax(y_predict[sample]))
        points = max_similarity_score[sample,:]*w[:,y_predict_class]
        # print(np.max(points))
        all_points = w*np.squeeze(max_similarity_score[sample,:])[:,np.newaxis]
        total_points = np.sum(all_points,axis=0)        
        
        # if(np.argmax(y_predict[sample]) != y_true[sample]):
        #     print("oh no")
        #     print(sample_date[sample])
        #     print(sample)
        #     # continue
        # print("anyway")
        #-------------------------------    
        base_col = isample*grid_per_col
        
        #var_index = VAR_INDEX[isample]
        var_index = 0
        if(var_index==0):
            # var_name = 'olr'
            var_name = 'Precipitation'
            letters = ('(a)','(b)','(c)','(d)')
        elif(var_index==1):
            var_name = 'u200'
            letters = ('(a)','(d)','(g)')            
        elif(var_index==2):
            var_name = 'u850'
            letters = ('(b)','(e)','(h)')      
        
        #----------------------   

        prototype_points = np.sort(points)[-(SORTED_VALUE[isample])]
        # for golf in np.arange(1,30):
        #     print(np.sort(points)[-golf])

        prototype_index = np.where(points == prototype_points)[0][0]
        # print(prototype_index)
        prototype_class = np.argmax(proto_class_mask[prototype_index])
        if(y_predict_class != prototype_class):
            print_warning = '\n- prototype not associated with predicted class -'
        else:
            print_warning = ''
        # print(y_predict.shape)
        # print("test" + str(y_predict_class))
        # print("true : " + str(labels[(70*365) + sample]))
        # print("isamp: "+ str(isample))
        #-------------------------------        
        # PLOT THE SAMPLE
        ax_samp = fig.add_subplot(spec[0,base_col:base_col+grid_per_col], projection=mapProj)            
        similarity_map  = similarity_scores[sample,:,:,prototype_index]
        # print(similarity_map)
        # print(prototype_index)
        j,k             = np.unravel_index(np.argmax(similarity_map), shape=similarity_map.shape)
        rf              = receptive_field.computeMask(j,k)   
        rf              = np.abs(rf-1.)
        rf[rf==0] = np.nan

        # print(input_data[0][0].shape)
        # print(input_train[0][0].shape)
        # quit()

        rf_save = rf            
        img = np.squeeze(input_data[0][0][sample,:,:,var_index])
        p = plots.plot_sample_shaded(ax_samp, img, globe=True, lat=lat, lon=lon, mapProj=mapProj, rf=rf)
        #p = plots.plot_sample(ax_samp, img, globe=True, lat=lat, lon=lon, mapProj=mapProj)

        # cbar = plt.colorbar(img,shrink=.5, aspect=20*0.8)

        #ax_samp.set_title(letters[0] + ' ' + var_name + ' of Sample ' + str(sample), fontsize=FS)
        ax_samp.set_title(letters[0] + ' ' + var_name + ' of Sample ', fontsize=FS)
        ax_samp.text(0.99, 1.0, 
            str(sample_date[sample]),
            fontfamily='monospace', 
            fontsize=FS, 
            va='bottom',
            ha='right',
            transform = ax_samp.transAxes,
        )

        class_text = "blah"
        if(y_true[sample] == 0):
            class_text = "cold class"
        elif(y_true[sample] == 1):
            class_text = "neutral class"
        else:
            class_text = "warm class"

        ax_samp.text(0.01, 1.0, 
            class_text,
            fontfamily='monospace', 
            fontsize=FS, 
            va='bottom',
            ha='left',
            transform = ax_samp.transAxes,
        )

        # ax_samp.text(0.66, 1.0, 
        #             'Phase: ' + str(phases[sample]),
        #             fontfamily='monospace', 
        #             fontsize=FS, 
        #             va='bottom',
        #             ha='right',
        #             transform = ax_samp.transAxes,
        #     )                 
        # print(sample)
        #-------------------------------        
        # PLOT THE PROTOTYPES
        ax = fig.add_subplot(spec[1,base_col:base_col+grid_per_col], projection=mapProj)
        rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])

        if(settings['plot_ERA5_convert'] == True):
            img = np.squeeze(input_val[0][0][prototype_sample[prototype_index],:,:,var_index])*rf 
        else:
            img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf
        #img[img == 0] = np.nan
        p = plots.plot_sample(ax, img, globe=True, lat=lat, lon=lon, mapProj=mapProj)
        

        #p = plots.plot_sample_shaded(ax, img, globe=True, lat=lat, lon=lon, mapProj=mapProj, rf =rf_save)
#         ax.set_title(letters[1] + ' ' + var_name + ' of Prototype ' + str(prototype_index) + ' (' + str(np.round(prototype_points,1)) + ' points)', fontsize=FS)
        ax.set_title(letters[1] + ' ' + var_name + ' of Prototype ' + str(prototype_index), fontsize=FS)
        ax.text(0.99, 1.0, 
            str(prototype_date[prototype_index]),
            fontfamily='monospace', 
            fontsize=FS, 
            va='bottom',
            ha='right',
            transform = ax.transAxes,
        ) 

        # class_text = "blah"
        # if(y_true[sample] == 0):
        #     class_text = "low class"
        # elif(y_true[sample] == 1):
        #     class_text = "neutral class"
        # else:
        #     class_text = "high class"

        class_text = "blah"
        print(sample)
        print(prototype_sample[prototype_index])
        if(y_train[prototype_sample[prototype_index]] == 0):
            class_text = "cold class"
        elif(y_train[prototype_sample[prototype_index]] == 1):
            class_text = "neutral class"
        else:
            class_text = "warm class"

        ax.text(0.01, 1.0, 
            str(class_text),
            fontfamily='monospace', 
            fontsize=FS, 
            va='bottom',
            ha='left',
            transform = ax.transAxes,
        )

        # ax.text(0.66, 1.0, 
        #             'Phase: ' + str(phases[prototype_sample[prototype_index]]),
        #             fontfamily='monospace', 
        #             fontsize=FS, 
        #             va='bottom',
        #             ha='right',
        #             transform = ax.transAxes,
        #     )   
        # print(prototype_sample[prototype_index])
        #-------------------------------        
        # PLOT THE MASKS
        ax = fig.add_subplot(spec[2,base_col:base_col+grid_per_col], projection=mapProj)
        ax.set_aspect("auto")
        img = local_mask[:,:,prototype_index] 
        # img = np.flipud(img)            
        p = plots.plot_mask(ax,img)
        p.set_clim(1.,np.max(img))
        ax.set_title(letters[2] + ' ' + 'Prototype ' + str(prototype_index) + ' Location Scaling', fontsize=FS)

        #-------------------------------        
        # PLOT THE POINTS
        ax = fig.add_subplot(spec[3,base_col+1:base_col+grid_per_col])
        plt.axhline(y=0,color='.75',linewidth=.5)    
        plot_colors = []
        for phase in np.arange(0,3):

            i = np.where(proto_class_mask[:,phase]==0)[0]
            plt.plot(np.ones(len(i))*phase,all_points[i,phase],
                    marker='o',
                    markeredgecolor='.5',
                    markerfacecolor='w', 
                    markersize=3,
                    markeredgewidth=.25,
                    )

            i = np.where(proto_class_mask[:,phase]==1)[0]
            p = plt.plot(np.ones(len(i))*phase,all_points[i,phase],'.')

            clr = p[0].get_color()
            plot_colors.append(clr)
            
            plt.text(phase,np.ceil(np.max(total_points))+.1,#6.1, 
                    str(np.round(total_points[phase],1)),
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    color=clr,
                    fontsize=12,
        #              weight='bold',
        #              transform=ax.transAxes, 
                    )
        # ax.set_title(letters[3] + ' ' + 'Points ' + str(prototype_index) + ' assigned per Class', fontsize=FS)
            
        
        plt.yticks((-1,0,1,2,3,4,5,6),('-1','0','1','2','3','4','5','6'))

        plt.ylim(-1,np.ceil(np.max(total_points))+.1)                     
        plt.xlim(-.5, 2.5)

        # plt.xticks(np.arange(0,2),("below","above"))
        plt.xticks(np.arange(0,3),("cold","neutral", "warm"))


        plt.xlabel('Temperature Class', fontsize=14)
        plt.ylabel('Points', fontsize=14)
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)  
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        [t.set_color(i) for (i,t) in
        zip(np.asarray(plot_colors),ax.xaxis.get_ticklabels())]    
        
        for t in plt.xticks()[1]:
            t.set_fontsize(12)
            
        for t in plt.yticks()[1]:
            t.set_fontsize(12)
        
        #-------------------------------        
        # sub3 = fig.add_subplot(spec[4,base_col+1:base_col+grid_per_col-1], projection=mapProj)
        # plt.set_cmap('cmr.fusion_r')
        # img = sub3.contourf(np.asarray(t_lon), np.asarray(t_lat), np.asarray(temp_test[sample,:,:]), np.linspace(-20, 20, 41), transform=ccrs.PlateCarree())
        # coast = cfeature.GSHHSFeature(scale='intermediate')
        # state = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale='10m', facecolor="none")
        # countries = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_0_boundary_lines_land', scale='10m', facecolor="none")
        # sub3.add_feature(coast, edgecolor='0')
        # sub3.add_feature(state, edgecolor='0')
        # sub3.add_feature(countries, edgecolor='0')
        # # sub1.coastlines()

        # sub3.tick_params(axis='both', labelsize=12)
        # sub3.set_xticks(np.arange(-180,181,30))
        # sub3.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
        # sub3.set_yticks(np.arange(-90,91,15))
        # sub3.set_xlim(10,75)
        # sub3.set_ylim(30,75)
        # sub3.set_xlabel("Longitude (degrees)",fontsize=14)
        # sub3.set_ylabel("Latitude (degrees)",fontsize=14)
        # # print(temp_anoms[8,64,88].values)
        # cbar = plt.colorbar(img,shrink=.75, aspect=20*0.8)
        # cbar.set_label("Temperature Anomaly (K)", fontsize=14)
        # cbar.ax.tick_params(labelsize=12) 
        # PLOT THE HISTO
        # bounds = 8
        # train_years = 70
        # sub3 = fig.add_subplot(spec[4,base_col+1:base_col+grid_per_col-1])

        # # print(temp.shape)

        # # print(avgtemp_day[SAMPLES[0] % 365, ].shape)

        # full_loc_temp = np.asarray(full_loc_temp)

        # sub3.hist(full_loc_temp, np.arange(np.round(np.median(full_loc_temp))-bounds,np.round(np.median(full_loc_temp))+bounds+1,.25))
        # sub3.axvline(np.percentile(full_loc_temp, 33.33), color='k', linestyle='dashed', linewidth=1)
        # sub3.axvline(np.percentile(full_loc_temp, 66.66), color='k', linestyle='dashed', linewidth=1)
        # # sub3.axvline(temp[(SAMPLES[0] + (365 * 70)+2, 46, 136, 0)] - avgtemp_day[(SAMPLES[0] % 365)+3, 46, 136, 0], color='orange', linestyle='solid', linewidth=1)
        # sub3.axvline(full_loc_temp[(sample + 14 + (365 * train_years))], color='orange', linestyle='solid', linewidth=1)
        # # sub3.axvline(full_loc_psl[prototype_sample[24]+3] - avgtemp_day[(prototype_sample[24] % 365)+3], color='red', linestyle='solid', linewidth=1)
        # # for i in range(len(best_points)):
        # #     sub3.axvline(best_points[i], color='r', linestyle='solid', linewidth=1)

        # true_val = labels[(train_years*365) + sample]
        # common_val = y_predict_class

        # print(true_val)
        # print(common_val)

        # if((true_val == 2) and (true_val == common_val)):
        #     sub3.axvspan(np.percentile(full_loc_temp, 66.66), bounds, color='thistle', alpha=0.5, lw=0)
        # elif((true_val == 2) and (true_val != common_val)):
        #     if(common_val == 0):
        #         sub3.axvspan(-1 * bounds, np.percentile(full_loc_temp, 33.33), color='gold', alpha=0.5, lw=0)
        #     elif(common_val == 2):
        #         sub3.axvspan(np.percentile(full_loc_temp, 66.66), bounds, color='gold', alpha=0.5, lw=0)
        #     else:
        #         sub3.axvspan(np.percentile(full_loc_temp, 33.33), np.percentile(full_loc_temp, 66.66), color='gold', alpha=0.5, lw=0)
        # elif((true_val == 1) and (true_val == common_val)):
        #     sub3.axvspan(np.percentile(full_loc_temp, 33.33), np.percentile(full_loc_temp, 66.66), color='thistle', alpha=0.5, lw=0)
        # elif((true_val == 1) and (true_val != common_val)):
        #     if(common_val == 0):
        #         sub3.axvspan(-1 * bounds, np.percentile(full_loc_temp, 33.33), color='gold', alpha=0.5, lw=0)
        #     elif(common_val == 2):
        #         sub3.axvspan(np.percentile(full_loc_temp, 66.66), bounds, color='gold', alpha=0.5, lw=0)
        #     else:
        #         sub3.axvspan(np.percentile(full_loc_temp, 33.33), np.percentile(full_loc_temp, 66.66), color='gold', alpha=0.5, lw=0)
        # elif((true_val == 0) and (true_val == common_val)):
        #     sub3.axvspan(-1 * bounds, np.percentile(full_loc_temp, 33.33), color='thistle', alpha=0.5, lw=0)
        # elif((true_val == 0) and (true_val != common_val)):
        #     if(common_val == 0):
        #         sub3.axvspan(-1 * bounds, np.percentile(full_loc_temp, 33.33), color='gold', alpha=0.5, lw=0)
        #     elif(common_val == 2):
        #         sub3.axvspan(np.percentile(full_loc_temp, 66.66), bounds, color='gold', alpha=0.5, lw=0)
        #     else:
        #         sub3.axvspan(np.percentile(full_loc_temp, 33.33), np.percentile(full_loc_temp, 66.66), color='gold', alpha=0.5, lw=0)


        # # # sub3.axvline(np.percentile(avg_slp_diff,5), color='k', linestyle='dashed', linewidth=1)
        # # # sub3.axvline(np.percentile(avg_slp_diff,95), color='k', linestyle='dashed', linewidth=1)
        # # sub3.set_title(first_month_str+" "+str(sel_day-total_day+1)+", year," + str(years) + ": Temp anomaly for the best 10 samples", fontsize=12)
        # sub3.set_ylabel("Number of samples", fontsize=10)
        # sub3.set_xlabel("Temp anomaly", fontsize=10)
        # sub3.set_xticks(np.arange(np.round(np.median(full_loc_temp))-bounds,np.round(np.median(full_loc_temp))+bounds,3))


    # plt.close()   
#     plt.tight_layout()
    if(era5_flag == 0):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_GCM.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    elif(era5_flag == 1):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_ERA5projected.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    else:
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_ERA5transfered_fixed.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    # plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_new.png'), bbox_inches='tight', dpi=dpiFig)
    #plt.show()


    # print(full)

##################################################################################################################################################################################################################
# All prototypes for each phase
# def show_all_protos(era5_flag, percentage):
#     print("Hello")

##################################################################################################################################################################################################################

# All prototypes for each phase
def show_all_protos(era5_flag, percentage):
    # print("HELOOOOOOOOOOOOOOOOOOOOO????")
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 12


    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation

    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']

    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

    less_than_one = np.where(mjo_amp < 1)[0]

    # phases[less_than_one] = 0


    for phase in np.arange(0,3):
        fig, axs = plt.subplots(10,
                            2, 
                            figsize=(18.3,22),
                            subplot_kw={'projection': mapProj}
                        )
        # fig, axs = plt.subplots(5,
        #                     2, 
        #                     figsize=(12,12),
        #                     subplot_kw={'projection': mapProj}
        #                    )
        top_samps = top_confidence_protos(percentage, y_predict)

        print(len(top_samps))
        print(y_predict.shape)
        print(y_true.shape)
        print(y_predict[top_samps].shape)
        
        #isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]

        isamples_top = np.where((np.argmax(y_predict[top_samps],axis=1)==phase) & (np.argmax(y_predict[top_samps],axis=1)==y_true[top_samps]))[0]

        isamples = top_samps[isamples_top]

        # print(isamples)
        # print(max_similarity_score[isamples,:].shape)
        points = max_similarity_score[isamples,:]*w[:,phase]
        # print(points.shape)
        k = np.where(proto_class_mask[:,phase]==0)[0]
        points[:,k] = 0.
        

        
        winning_prototype = np.argmax(points,axis=1)
        
        # print(isamples)
        # print(winning_prototype)
        
        
        points_avg = np.mean(points,axis=0)
        # proto_vector = np.where(points_avg != 0)[0] 
        proto_vector = np.where(proto_class_mask[:,phase]==1)[0]
        proto_points_vector = points_avg[proto_vector]
        sorted_index = np.argsort(proto_points_vector)
        
        # print(proto_vector[np.flipud(sorted_index)])

        points_var = np.var(points,axis=0)
        # print(points_var)
        
        # print(points.shape)
        # print(points_avg.shape)
        # print(max_similarity_score.shape)
        for ivar, var_index in enumerate([0]):
            if(var_index==0):
                var_name = 'Precipitation'
            elif(var_index==1):
                var_name = 'Precipitation'
            elif(var_index==2):
                var_name = 'u850'        
            
            # print(axs.shape)ß

            phase0_protos = []
            
            for iprototype, prototype_index in enumerate(proto_vector[np.flipud(sorted_index)]):
                # print("proto index")
                # print(prototype_index)

                #-------------------------------        
                # PLOT THE PROTOTYPES
                ax = axs[iprototype,ivar]
                ax.set_aspect("auto")
                # print(prototype_index)
                # print(prototype_indices)
                # print(prototype_indices[prototype_index,0])
                # print(prototype_indices[prototype_index,1])
                rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])
                img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf

                ######## TEST
                if phase == 0:
                    phase0_protos.append(img)
                img[img == 0] = np.nan
                p = plots.plot_sample(ax,
                                    img,
                                    globe=True,
                                    lat=lat,
                                    lon=lon,
                                    mapProj=mapProj,
                                    )
                # print(lat)
                # print(lon)
                # print(img)
                # p.set_clim(-7,7)

                class_text = "blah"
                if(phase == 0):
                    class_text = "cold class"
                elif(phase == 1):
                    class_text = "neutral class"
                else:
                    class_text = "warm class"
                ax.set_title(var_name + ' of Prototype ' + str(prototype_index), fontsize=FS*1.25)
                ax.text(0.01, 1.0, 
                    str(class_text),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='left',
                    transform = ax.transAxes,
                )

                # ax.text(0.49, 1.0, 
                #     'Phase: ' + str(phases[prototype_sample[prototype_index]]),
                #     fontfamily='monospace', 
                #     fontsize=FS, 
                #     va='bottom',
                #     ha='right',
                #     transform = ax.transAxes,
                # )   

                ax.text(0.99, 1.0, 
                    str(prototype_date[prototype_index]),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='right',
                    transform = ax.transAxes,
                )            
                #-------------------------------        
                # PLOT THE MASKS
                if var_index==0:
                    iwin = np.where(winning_prototype==prototype_index)[0]
                    if(len(winning_prototype>1)):
                        win_frac = np.round(len(iwin)/len(winning_prototype)*100, 2)
                    else:
                        win_frac = 0
                    
                    ax = axs[iprototype,1]
                    ax.set_aspect("auto")
                    img = local_mask[:,:,prototype_index]
                    # img = np.flipud(img)
                    p = plots.plot_mask(ax,img)
                    p.set_clim(1.,np.max(img[:]))
                    # print(np.max(img[:]))
                    ax.set_title('Prototype ' + str(prototype_index) + ' Location Scaling', fontsize=FS*1.25)
                    
                    ax.text(0.00, 1.0, 
                        str(np.round(points_avg[prototype_index],1)) + ' pts.',
                        fontfamily='monospace', 
                        fontsize=FS, 
                        va='bottom',
                        ha='left',
                        transform = ax.transAxes,
                        )
                    ax.text(1.0, 1.0, 
                        'σ² = ' + str(np.round(points_var[prototype_index],3)),
                        fontfamily='monospace', 
                        fontsize=FS, 
                        va='bottom',
                        ha='right',
                        transform = ax.transAxes,
                        )            

        if phase == 0:
            phase0_protos = np.asarray(phase0_protos)
            np.save(exp_data_dir + "_"+ EXP_NAME + 'ERA5_phase0_protos.npy', phase0_protos)
            print("test")
        if(era5_flag == 0):
            plt.savefig((vizualization_dir + str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            # plt.savefig((vizualization_dir + str(learning_rate) +  "_" +str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        elif(era5_flag == 1):
            plt.savefig((vizualization_dir + 'era5_figs/' + str(percentage*100) + "_" + "_" + EXP_NAME + '_translated_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        else:
            plt.savefig((vizualization_dir + 'era5_figs/' + str(percentage*100) + "_" + "_" + EXP_NAME + '_convert_allPrototypes_phase' + str(phase) + '_fixed.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
            
########################################################################################################################

def comps_by_proto(era5_flag):
    # print("HELOOOOOOOOOOOOOOOOOOOOO????")
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 12


    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation

    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']

    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

    less_than_one = np.where(mjo_amp < 1)[0]

    # phases[less_than_one] = 0


    for phase in np.arange(0,3):
        fig, axs = plt.subplots(10,
                            2, 
                            figsize=(18.3,22),
                            subplot_kw={'projection': mapProj}
                        )
        # fig, axs = plt.subplots(5,
        #                     2, 
        #                     figsize=(12,12),
        #                     subplot_kw={'projection': mapProj}
        #                    )

        
        isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]

        points = max_similarity_score[isamples,:]*w[:,phase]

        k = np.where(proto_class_mask[:,phase]==0)[0]
        points[:,k] = 0.
        

        
        winning_prototype = np.argmax(points,axis=1)
        
        
        points_avg = np.mean(points,axis=0)
        # proto_vector = np.where(points_avg != 0)[0] 
        proto_vector = np.where(proto_class_mask[:,phase]==1)[0]
        proto_points_vector = points_avg[proto_vector]
        sorted_index = np.argsort(proto_points_vector)
        
        print(proto_vector[np.flipud(sorted_index)])

        points_var = np.var(points,axis=0)
        print(points_var)
        
        # print(points.shape)
        # print(points_avg.shape)
        # print(max_similarity_score.shape)
        for ivar, var_index in enumerate([0]):
            if(var_index==0):
                var_name = 'Precipitation'
            elif(var_index==1):
                var_name = 'Precipitation'
            elif(var_index==2):
                var_name = 'u850'        
            
            # print(axs.shape)ß
            
            for iprototype, prototype_index in enumerate(proto_vector[np.flipud(sorted_index)][:4]):
                # print("proto index")
                # print(prototype_index)

                
                #-------------------------------        
                # PLOT THE PROTOTYPES
                ax = axs[iprototype,ivar]
                ax.set_aspect("auto")
                # print(prototype_index)
                # print(prototype_indices)
                # print(prototype_indices[prototype_index,0])
                # print(prototype_indices[prototype_index,1])
                rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])
                img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf
                img[img == 0] = np.nan
                p = plots.plot_sample(ax,
                                    img,
                                    globe=True,
                                    lat=lat,
                                    lon=lon,
                                    mapProj=mapProj,
                                    )
                # print(lat)
                # print(lon)
                # print(img)
                # p.set_clim(-7,7)

                class_text = "blah"
                if(phase == 0):
                    class_text = "cold class"
                elif(phase == 1):
                    class_text = "neutral class"
                else:
                    class_text = "warm class"
                ax.set_title(var_name + ' of Prototype ' + str(prototype_index), fontsize=FS*1.25)
                ax.text(0.01, 1.0, 
                    str(class_text),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='left',
                    transform = ax.transAxes,
                )

                # ax.text(0.49, 1.0, 
                #     'Phase: ' + str(phases[prototype_sample[prototype_index]]),
                #     fontfamily='monospace', 
                #     fontsize=FS, 
                #     va='bottom',
                #     ha='right',
                #     transform = ax.transAxes,
                # )   

                ax.text(0.99, 1.0, 
                    str(prototype_date[prototype_index]),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='right',
                    transform = ax.transAxes,
                )            
                #-------------------------------        
                # PLOT THE MASKS
                if var_index==0:
                    
                    samp_proto_index = np.where(winning_prototype == prototype_index)[0]
                    
                    samp_proto_locs = isamples[samp_proto_index]
                    
                    img = np.zeros(np.squeeze(input_data[0][0][samp_proto_locs[0],:,:,var_index]).shape)
                    
                    for samp_proto_loc in samp_proto_locs:
                        
                        samp_img = np.squeeze(input_data[0][0][samp_proto_loc,:,:,var_index])
                        
                        img = img + samp_img
                   
                        
                    img = img / samp_proto_locs.shape[0]
                    print(samp_proto_locs.shape[0])
                    
                    ax = axs[iprototype,1]
                    ax.set_aspect("auto")
                    # img = local_mask[:,:,prototype_index]
                    # img = np.flipud(img)
                    p = plots.plot_sample(ax, img * 5, globe=True, lat=lat, lon=lon, mapProj=mapProj)

                    # p.set_clim(1.,np.max(img[:]))
                    # print(np.max(img[:]))
                    ax.set_title('Prototype ' + str(prototype_index) + ' Sample Composite', fontsize=FS*1.25)
                    
                    ax.text(0.00, 1.0, 
                        'n = ' + str(samp_proto_locs.shape[0]),
                        fontfamily='monospace', 
                        fontsize=FS, 
                        va='bottom',
                        ha='left',
                        transform = ax.transAxes,
                        )

        if(era5_flag == 0):
            plt.savefig((vizualization_dir + "_" + EXP_NAME + '_Prototype_comps_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            # plt.savefig((vizualization_dir + str(learning_rate) +  "_" +str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        elif(era5_flag == 1):
            plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_translated_Prototype_comps_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        else:
            plt.savefig((vizualization_dir + 'era5_figs/' + "_" + EXP_NAME + '_convert_Prototype_comps_phase' + str(phase) + '_fixed.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()

def useless_func(era5_flag):
    # print("HELOOOOOOOOOOOOOOOOOOOOO????")
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 12


    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' # use this one for historical and SSP simulations with CESM2-WACCM


    #f = '/Users/nicojg/Documents/Work/2021_Fall_IAI/Data/Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'

    MJO_info = pd.read_pickle(f)

    # the indexing from [:180*2] is so that we only grab the winds and not precip for the correlation

    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']

    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

    less_than_one = np.where(mjo_amp < 1)[0]

    # phases[less_than_one] = 0


    for phase in np.arange(0,3):
        fig, axs = plt.subplots(10,
                            2, 
                            figsize=(18.3,22),
                            subplot_kw={'projection': mapProj}
                        )
        # fig, axs = plt.subplots(5,
        #                     2, 
        #                     figsize=(12,12),
        #                     subplot_kw={'projection': mapProj}
        #                    )

        
        isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]

        points = max_similarity_score[isamples,:]*w[:,phase]

        k = np.where(proto_class_mask[:,phase]==0)[0]
        points[:,k] = 0.
        

        
        winning_prototype = np.argmax(points,axis=1)
        
        
        points_avg = np.mean(points,axis=0)
        # proto_vector = np.where(points_avg != 0)[0] 
        proto_vector = np.where(proto_class_mask[:,phase]==1)[0]
        proto_points_vector = points_avg[proto_vector]
        sorted_index = np.argsort(proto_points_vector)
        
        print(proto_vector[np.flipud(sorted_index)])

        points_var = np.var(points,axis=0)
        print(points_var)
        
        # print(points.shape)
        # print(points_avg.shape)
        # print(max_similarity_score.shape)
        for ivar, var_index in enumerate([0]):
            if(var_index==0):
                var_name = 'precip'
            elif(var_index==1):
                var_name = 'precip'
            elif(var_index==2):
                var_name = 'u850'        
            
            # print(axs.shape)ß
            
            for iprototype, prototype_index in enumerate(proto_vector[np.flipud(sorted_index)][:3]):
                # print("proto index")
                # print(prototype_index)

                #-------------------------------        
                # PLOT THE PROTOTYPES
                ax = axs[iprototype,ivar]
                ax.set_aspect("auto")
                # print(prototype_index)
                # print(prototype_indices)
                # print(prototype_indices[prototype_index,0])
                # print(prototype_indices[prototype_index,1])
                rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])
                img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf
                img[img == 0] = np.nan
                p = plots.plot_sample(ax,
                                    img,
                                    globe=True,
                                    lat=lat,
                                    lon=lon,
                                    mapProj=mapProj,
                                    )
                # print(lat)
                # print(lon)
                # print(img)
                # p.set_clim(-7,7)

                class_text = "blah"
                if(phase == 0):
                    class_text = "cold class"
                elif(phase == 1):
                    class_text = "neutral class"
                else:
                    class_text = "warm class"
                ax.set_title(var_name + ' of Prototype ' + str(prototype_index), fontsize=FS*1.25)
                ax.text(0.01, 1.0, 
                    str(class_text),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='left',
                    transform = ax.transAxes,
                )

                # ax.text(0.49, 1.0, 
                #     'Phase: ' + str(phases[prototype_sample[prototype_index]]),
                #     fontfamily='monospace', 
                #     fontsize=FS, 
                #     va='bottom',
                #     ha='right',
                #     transform = ax.transAxes,
                # )   

                ax.text(0.99, 1.0, 
                    str(prototype_date[prototype_index]),
                    fontfamily='monospace', 
                    fontsize=FS, 
                    va='bottom',
                    ha='right',
                    transform = ax.transAxes,
                )            
                #-------------------------------        
                # PLOT THE MASKS
                if var_index==0:
                    
                    samp_proto_index = np.where(winning_prototype == prototype_index)[0]
                    
                    samp_proto_locs = isamples[samp_proto_index]
                    
                    img = np.zeros(np.squeeze(input_data[0][0][samp_proto_locs[0],:,:,var_index]).shape)
                    
                    for samp_proto_loc in samp_proto_locs:
                        
                        samp_img = np.squeeze(input_data[0][0][samp_proto_loc,:,:,var_index])
                        
                        img = img + samp_img
                   
                        
                    img = img / samp_proto_locs.shape[0]
                    print(samp_proto_locs.shape[0])
                    
                    print(samp_proto_locs[iprototype])
                    img = np.squeeze(input_data[0][0][samp_proto_locs[iprototype],:,:,var_index])

                    ax = axs[iprototype,1]
                    ax.set_aspect("auto")
                    # img = local_mask[:,:,prototype_index]
                    # img = np.flipud(img)
                    p = plots.plot_sample(ax, img , globe=True, lat=lat, lon=lon, mapProj=mapProj)

                    # p.set_clim(1.,np.max(img[:]))
                    # print(np.max(img[:]))
                    ax.set_title('Prototype ' + str(prototype_index) + ' Sample Composite', fontsize=FS*1.25)
                    
                    ax.text(0.00, 1.0, 
                        'n = ' + str(samp_proto_locs.shape[0]),
                        fontfamily='monospace', 
                        fontsize=FS, 
                        va='bottom',
                        ha='left',
                        transform = ax.transAxes,
                        )

        if(era5_flag == 0):
            plt.savefig((vizualization_dir + "_" + EXP_NAME + '_useless' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            # plt.savefig((vizualization_dir + str(learning_rate) +  "_" +str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        elif(era5_flag == 1):
            plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_translated_Prototype_comps_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        else:
            plt.savefig((vizualization_dir + 'era5_figs/' + "_" + EXP_NAME + 'useless_ERA5' + str(phase) + '_fixed.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()

def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge", label=("Prototype " + str(i)))
    plt.xticks(_X, X)


def proto_rankings(best_samps, samps_bool, percent_samps = 100):
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 5

    protos_num = NPROTOTYPES/ NCLASSES
    # proto_ranks = np.zeros((NPROTOTYPES, int(protos_num)))

    for temp_class in np.arange(0,3):

        proto_ranks = np.zeros((int(protos_num), int(protos_num)))

        proto_ranks_full = np.zeros((int(protos_num), int(protos_num)))

        fig, axs = plt.subplots(10,
                            2, 
                            figsize=(12,22),
                            subplot_kw={'projection': mapProj}
                        )
        # fig, axs = plt.subplots(5,
        #                     2, 
        #                     figsize=(12,12),
        #                     subplot_kw={'projection': mapProj}
        #                    )


        if(samps_bool):
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp])==temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = temp_best_samps
        else:
            isamples = np.where((np.argmax(y_predict,axis=1)==temp_class) & (np.argmax(y_predict,axis=1)==y_true))[0]

        points = max_similarity_score[isamples,:]*w[:,temp_class]
        k = np.where(proto_class_mask[:,temp_class]==0)[0]
        points[:,k] = 0.

        # print(temp_class)

        # print(int(temp_class*protos_num))

        # print(int((temp_class*protos_num)+protos_num))

        for i in np.arange(0, points.shape[0], 1):
            class_points = points[int(i)][int(temp_class*protos_num):int((temp_class*protos_num)+protos_num)]
            class_scores = np.argsort(class_points)[::-1]
            for j in np.arange(0, class_points.shape[0], 1):
                # print(i%(temp_class*protos_num))
                proto_ranks[class_scores[int(j)]][int(j)] += 1

        isamples_full = np.where((np.argmax(y_predict,axis=1)==temp_class))[0]

        points_full = max_similarity_score[isamples_full,:]*w[:,temp_class]
        k_full = np.where(proto_class_mask[:,temp_class]==0)[0]
        points_full[:,k_full] = 0.

        for i in np.arange(0, points_full.shape[0], 1):
            class_points_full = points_full[int(i)][int(temp_class*protos_num):int((temp_class*protos_num)+protos_num)]
            class_scores_full = np.argsort(class_points_full)[::-1]
            for j in np.arange(0, class_points_full.shape[0], 1):
                # print(i%(temp_class*protos_num))
                proto_ranks_full[class_scores_full[int(j)]][int(j)] += 1


        # print(proto_ranks)

        rank_names = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']

        rank_names = np.asarray(rank_names)
        fig, axs = plt.subplots(1,
                        1, 
                        figsize=(21/2.5,12/2.5)
                    )
        # for i in np.arange(0, proto_ranks.shape[0], 1):

        #     # fig, axs = plt.subplots(1,
        #     #                         1, 
        #     #                         figsize=(21/2.5,12/2.5)
        #     #                     )
                
        #     plt.title("Prototype " + str(temp_class*10 + i) + " Ranking " + "(n=" + str(np.sum(proto_ranks[i]))+")")
        #     plt.bar(rank_names, proto_ranks[int(i)])

        n = len(proto_ranks)
        _rank_names = np.arange(len(rank_names))
        # _rank_names = [2*i for i in _rank_names]

        width = .8

        for i in range(n):
            plt.bar(_rank_names - width + i/float(n)*width, proto_ranks[i], 
                    width=width/float(n), align="edge", label=("Prototype " + str(i+(temp_class*10))))
        plt.xticks(_rank_names, rank_names)


        # subcategorybar(rank_names, proto_ranks)

        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + "percent - Class" + str(temp_class) +" Prototype Rankings " + "(n=" + str(points.shape[0])+")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) +"class_"+ str(temp_class)+"_counts"+ '.png'), bbox_inches='tight', dpi=dpiFig)

        fig, axs = plt.subplots(1,
                        1, 
                        figsize=(21/2.5,12/2.5)
                    )

        for i in range(n):
            plt.bar(_rank_names - width + i/float(n)*width, proto_ranks[i]/points.shape[0], 
                    width=width/float(n), align="edge", label=("Prototype " + str(i+(temp_class*10))))
        plt.xticks(_rank_names, rank_names)


        # subcategorybar(rank_names, proto_ranks)

        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + "percent - Class" + str(temp_class) +" Prototype Rankings " + "(n=" + str(points.shape[0])+")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) +"class_"+ str(temp_class)+"_freq"+ '.png'), bbox_inches='tight', dpi=dpiFig)

        fig, axs = plt.subplots(1,
                        1, 
                        figsize=(21/2.5,12/2.5)
                    )

        for i in range(n):
            plt.bar(_rank_names - width + i/float(n)*width, proto_ranks[i]/points.shape[0] - proto_ranks_full[i]/points_full.shape[0], 
                    width=width/float(n), align="edge", label=("Prototype " + str(i+(temp_class*10))))
        plt.xticks(_rank_names, rank_names)


        # subcategorybar(rank_names, proto_ranks)

        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + "percent - Class" + str(temp_class) +" Prototype Rankings " + "(n=" + str(points.shape[0])+")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) +"class_"+ str(temp_class)+"_DIFF_freq"+ '.png'), bbox_inches='tight', dpi=dpiFig)


def compare_accuracies():
    run_seeds = [112,117]
    run_name = "GCM_alas_wint_550yrs_shuf_bal_seed"
    normal_fn = "normal_" + run_name + "125_TLLTT_accuracy.txt"
    normal_accuracies = np.loadtxt("/barnes-engr-scratch1/nicojg/data/" + run_name + "125/" + normal_fn)#.astype(float)

    plt.figure(figsize=(10,6))
    plt.plot(np.arange(10, 101, 5)[::-1], normal_accuracies, label = "ProtoLNet", color = '#f42c94')

    for run_seed in run_seeds:
        normal_fn = "normal_" + run_name + str(run_seed) + "_TLLTT_accuracy.txt"
        normal_accuracies = np.loadtxt("/barnes-engr-scratch1/nicojg/data/" + run_name + str(run_seed) + "/" + normal_fn)#.astype(float)
        plt.plot(np.arange(10, 101, 5)[::-1], normal_accuracies, label = "Base CNN", color = '#f89c04')
    # plt.title("Model Accuracy by percentage of most confident samples", fontsize=20)
    plt.title("Discard test", fontsize=20)
    plt.xlabel("Percentage samples not discarded", fontsize=15)
    plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1])
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.axhspan(0, 33, color='0.75', alpha=0.5, lw=0)

    plt.ylim(bottom=30)

    plt.savefig(("figures/misc/failure_plot.png"), bbox_inches='tight', dpi=400)

def plot_alaksa_temp(samp, era5_flag):
    # mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))

    temp_anoms, t_lon, t_lat = data_functions_schooner.get_temp_anoms(DATA_DIR)
    # fig, ax = plt.subplots(1,
    #                         1, 
    #                         figsize=(10,9.5),
    #                         subplot_kw={'projection': mapProj}
    #                     )
    
    # p = plots.plot_sample(ax, temp_anoms[8,:,:] , globe=True, lat=t_lat, lon=t_lon, mapProj=mapProj)

    fig = plt.figure(figsize=(10, 9.5))
    fig.tight_layout()

    spec = fig.add_gridspec(4, 5)

    plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

    sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

    plt.set_cmap('cmr.fusion_r')
    img = sub1.contourf(np.asarray(t_lon), np.asarray(t_lat), np.asarray(temp_test[samp,:,:]), np.linspace(-20, 20, 41), transform=ccrs.PlateCarree(), extend='both')
    coast = cfeature.GSHHSFeature(scale='intermediate')
    state = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale='10m', facecolor="none")
    countries = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_0_boundary_lines_land', scale='10m', facecolor="none")
    sub1.add_feature(coast, edgecolor='0')
    sub1.add_feature(state, edgecolor='0')
    sub1.add_feature(countries, edgecolor='0')
    # sub1.coastlines()

    sub1.tick_params(axis='both', labelsize=12)
    sub1.set_xticks(np.arange(-180,181,30))
    sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
    sub1.set_yticks(np.arange(-90,91,15))
    sub1.set_xlim(10,75)
    sub1.set_ylim(30,75)
    sub1.set_xlabel("Longitude (degrees)",fontsize=14)
    sub1.set_ylabel("Latitude (degrees)",fontsize=14)
    sub1.set_title("(e) 14-day lead time temperature anomalies", fontsize=14)
    # print(temp_anoms[8,64,88].values)
    cbar = plt.colorbar(img,shrink=.75, aspect=20*0.8)
    cbar.set_label("Temperature (K)", fontsize=14)
    cbar.ax.tick_params(labelsize=12) 
    if(era5_flag == 0):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_real_temp_' + str(samp) + '_' + 'coast_plot.png'), bbox_inches='tight', dpi=dpiFig)
        # plt.savefig((vizualization_dir + str(learning_rate) +  "_" +str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    elif(era5_flag == 1):
        plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_translated_real_temp' + str(samp) + '_' + 'coast_plot.png'), bbox_inches='tight', dpi=dpiFig)
    else:
        plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_convert_real_temp' + str(samp) + '_' + 'coast_plot_fixed.png'), bbox_inches='tight', dpi=dpiFig)

# plot_alaksa_temp()
# quit()
# compare_accuracies()


accuracies = []
base_accuracies = []

accuracies_val = []
base_accuracies_val = []

era5_flag_set = 0
if(settings['plot_ERA5_translated'] == True):
    era5_flag_set = 1 
if(settings['plot_ERA5_convert'] == True):
    era5_flag_set = 2

# comps_by_proto(era5_flag_set)
# useless_func(era5_flag_set)

top_samps = top_confidence_protos(1, y_predict)[:5]
# for decent_samp in top_samps:
#     plot_alaksa_temp(decent_samp, era5_flag_set)
#     examine_proto(decent_samp, era5_flag_set)

# quit()    

# for i in np.arange(10, 101, 5):
#     accuracies.append(make_confuse_matrix(y_predict[top_confidence_protos(i/100.)], y_true[top_confidence_protos(i/100.)], i, False))

# for i in np.arange(10, 101, 5):
#     base_accuracies.append(make_confuse_matrix(base_y_predict[top_confidence_protos(i/100.)], y_true[top_confidence_protos(i/100.)], i, True))
# for i in np.arange(10, 101, 5):
#     precip_comps(top_confidence_protos(i/100.), True, i)

# timeseries_predictions()


for i in np.arange(10, 101, 5):
    # precip_comps(top_confidence_protos(i/100., y_predict), True, i)
    if(settings['pretrain'] == True):
        base_accuracies.append(make_confuse_matrix(base_y_predict[top_confidence_protos(i/100., base_y_predict_test)], y_true[top_confidence_protos(i/100., base_y_predict_test)], i, True))
    accuracies.append(make_confuse_matrix(y_predict[top_confidence_protos(i/100., y_predict)], y_true[top_confidence_protos(i/100., y_predict)], i, False))

    # precip_comps(top_confidence_protos(i/100., y_predict_val), True, i)
    if(settings['pretrain'] == True):
        base_accuracies_val.append(make_confuse_matrix(base_y_predict_val[top_confidence_protos(i/100., base_y_predict_val)], y_val[top_confidence_protos(i/100., base_y_predict_val)], i, True))
    accuracies_val.append(make_confuse_matrix(y_predict_val[top_confidence_protos(i/100., y_predict_val)], y_val[top_confidence_protos(i/100., y_predict_val)], i, False))
    # mjo_lookup(top_confidence_protos(i/100.), True, i)
    # proto_rankings(top_confidence_protos(i/100.), True, i)
# plt.close()


plt.figure(figsize=(10,6))
plt.plot(np.arange(10, 101, 5)[::-1], accuracies, label = "ProtoLNet", color = '#f42c94')
# plt.plot(np.arange(10, 101, 5)[::-1], accuracies_val, label = "TLLTT - val")
if(settings['pretrain'] == True):
    plt.plot(np.arange(10, 101, 5)[::-1], base_accuracies, label = "Base CNN", color = '#f89c04')
    # plt.plot(np.arange(10, 101, 5)[::-1], base_accuracies_val, label = "Base CNN - val")
# plt.title("Model Accuracy by percentage of most confident samples", fontsize=20)
# plt.title("Discard test", fontsize=20)

plt.xlabel("Percentage samples not discarded", fontsize=15)
plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=15)
plt.axhspan(0, 33, color='0.75', alpha=0.5, lw=0)

print(np.min(accuracies) >= 31)

# print(np.min(base_accuracies) >= 31)
if(settings['pretrain'] == True):
    if((np.min(accuracies) >= 31) and (np.min(base_accuracies) >= 31)):
        print("srtting ylim")
        plt.ylim(bottom=30)
    else:
        plt.ylim(bottom=20)
    plt.legend()
else:
    if((np.min(accuracies) >= 31)):
        print("srtting ylim")
        plt.ylim(bottom=30)
    else:
        plt.ylim(bottom=20)
    plt.legend()

if(not era5_plots):
    plt.savefig((vizualization_dir + EXP_NAME + '_forecast_of_opportunity.png'), bbox_inches='tight', dpi=dpiFig)
    # plt.savefig((vizualization_dir  + str(learning_rate) + '_' +  EXP_NAME + '_forecast_of_opportunity.png'), bbox_inches='tight', dpi=dpiFig)
else:
    plt.savefig((vizualization_dir + 'era5_figs/' + EXP_NAME + '_forecast_of_opportunity.png'), bbox_inches='tight', dpi=dpiFig)
# plt.show()

# if(not did_not_train):
#     show_all_protos(era5_flag_set, 1)
#     show_all_protos(era5_flag_set, .2)

translated_fn = exp_data_dir + "translated_era5_"+ EXP_NAME + '_TLLTT_accuracy.txt'
convert_fn = exp_data_dir + "convert_era5_"+ EXP_NAME + '_TLLTT_accuracy.txt'
normal_fn = exp_data_dir + "normal_"+ EXP_NAME + '_TLLTT_accuracy.txt'

if(settings['plot_ERA5_translated']):
    np.savetxt(translated_fn, accuracies, fmt='%1.5f')
elif(settings['plot_ERA5_convert']):
    np.savetxt(convert_fn, accuracies, fmt='%1.5f')
elif(EXP_NAME[:3] == 'GCM' or EXP_NAME[:3] == 'ERA'):
    np.savetxt(normal_fn, accuracies, fmt='%1.5f')
    
if (os.path.exists(translated_fn) and os.path.exists(convert_fn) and os.path.exists(normal_fn)):
    
    translated_accuracies = np.loadtxt(translated_fn)#.astype(float)
    convert_accuracies = np.loadtxt(convert_fn)#.astype(float)
    normal_accuracies = np.loadtxt(normal_fn)#.astype(float)
    
    plt.figure(figsize=(10,6))
    plt.xlim(10,100)
    plt.plot(np.arange(10, 101, 5)[::-1], normal_accuracies, label = "CESM2-ProtoLNet", color = '#f42c94', linewidth=2)
    plt.plot(np.arange(10, 101, 5)[::-1], translated_accuracies, label = "CESM2 onto ERA5", color = '#f89c04', linewidth=2) 
    plt.plot(np.arange(10, 101, 5)[::-1], convert_accuracies, label = "Transferred-ProtoLNet", color = '#3c8cdc', linewidth=2)
    # plt.title("Model Accuracy by percentage of most confident samples", fontsize=20)
    plt.xlabel("Percent Most Confident (%)", fontsize=15)
    plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1], fontsize =12)
    plt.yticks(fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.axhspan(0, 33, color='.75', alpha=0.5, lw=0)
    
    if(settings['plot_ERA5_convert'] == True):
        print(nino_final_accur)
        plt.axhline(nino_final_accur*100, color='black', label="Niño3.4 Baseline", linewidth=2)
    
    print(convert_accuracies)
    if((np.min(normal_accuracies) >= 31)):
        print("srtting ylim")
        plt.ylim(bottom=30)
    else:
        plt.ylim(bottom=20)
    plt.legend(fontsize = 12)


    plt.savefig((vizualization_dir + 'era5_figs/' + EXP_NAME + '_ERA5_additions_forecast_of_opportunity_fixed.png'), bbox_inches='tight', dpi=dpiFig)

# if(not did_not_train):
#     show_all_protos(era5_flag_set, 1)
#     show_all_protos(era5_flag_set, .2)
    
print((np.argmax(y_predict,axis=1) == y_true).astype(int))
print(np.max(y_predict,axis=1))

# brier_score = 0
# o_t = (np.argmax(y_predict,axis=1) == y_true).astype(int)
# f_t = np.max(y_predict,axis=1)
# for i in np.arange(0, len(y_predict), 1):
#     print(o_t.shape)
#     o_ti = o_t[i]
#     f_ti = f_t[i]
#     brier_score += np.square(f_ti - o_ti)
# brier_score = brier_score/len(y_predict)

# # Step 1: One-hot encode y_true
# num_classes = y_predict.shape[1]
# y_true_one_hot = np.eye(num_classes)[y_true]  # shape: (num_samples, num_classes)

# Step 2: Compute the Brier score for all classes

# print(y_true.shape)
# print(y_true)
print("break")
print(y_predict)
y_true_one_hot = np.eye(NCLASSES)[y_true.astype(int)]
print(y_true_one_hot)
brier_score = np.mean(np.sum((y_true_one_hot - y_predict) ** 2, axis=1)/3)

print(brier_score)
np.savetxt(exp_data_dir + "_"+ EXP_NAME + 'brier_score.txt', [brier_score], fmt='%f')

# Compute cumulative distributions
y_true_cum = np.cumsum(y_true_one_hot, axis=1)
y_pred_cum = np.cumsum(y_predict, axis=1)

# Compute squared differences and average RPS
rps_score = np.mean(np.sum((y_true_cum - y_pred_cum) ** 2, axis=1))/2

# Print and save
print(rps_score)
print("above rps")
np.savetxt(exp_data_dir + "_" + EXP_NAME + 'rps_score.txt', [rps_score], fmt='%f')


top20_pred = y_predict[top_confidence_protos(.2, y_predict)]
top20_true = y_true[top_confidence_protos(.2, y_predict)]

# brier_score = 0
# o_t = (np.argmax(top20_pred,axis=1) == top20_true).astype(int)
# f_t = np.max(top20_pred,axis=1)
# for i in np.arange(0, len(top20_pred), 1):
#     o_ti = o_t[i]
#     f_ti = f_t[i]
#     brier_score += np.square(f_ti - o_ti)
# brier_score = brier_score/len(top20_pred)
# print(brier_score)

top20_y_true_one_hot = np.eye(NCLASSES)[top20_true.astype(int)]
brier_score = np.mean(np.sum((top20_y_true_one_hot - top20_pred ) ** 2, axis=1)/3)
print(brier_score)

np.savetxt(exp_data_dir + "_"+ EXP_NAME + 'brier_score_top20.txt', [brier_score], fmt='%f')

# Compute cumulative distributions
top20_y_true_cum = np.cumsum(top20_y_true_one_hot, axis=1)
top_20y_pred_cum = np.cumsum(top20_pred, axis=1)

# Compute squared differences and average RPS
top20_rps_score = np.mean(np.sum((top20_y_true_cum - top_20y_pred_cum) ** 2, axis=1))/2

print(top20_rps_score)
print("above")

np.savetxt(exp_data_dir + "_" + EXP_NAME + 'rps_score_top20.txt', [top20_rps_score], fmt='%f')


print("end")
quit()
top_samps = top_confidence_protos(1, y_predict)[:5]
for decent_samp in top_samps:
    # mjo_correlation(decent_samp)
    examine_proto(decent_samp, era5_flag_set)
    plot_alaksa_temp(decent_samp, era5_flag_set)