# This Looks Like That There - S2S Release Version


# Visualization code
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

import network
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

__author__ = "Nicolas Gordillo"
__version__ = "April 2024"

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
    # EXP_NAME = 'GCM_alas_lr_wint_550yrs_seed144_nopre_lrtest_epochs'
    EXP_NAME = 'GCM_alas_lr_wint_550redo_seed'+str(num) #+ '_nopre' #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'
    import experiment_settings_multiple_seeds_lr_redo as experiment_settings
    # import experiment_settings_shuf_550bal_seeds as experiment_settings
    file_lon = 89
    file_lat = 64
else:
    file_lon = int(sys.argv[2])
    file_lat = int(sys.argv[1])
    EXP_NAME = 'GCM_'+ str(file_lon) + '_' + str(file_lat) +'_wint_550yrs_shuf_bal_seed134_redo' #Modify seeds as needed
    import experiment_settings_coast_550_lr_adjust_134_redo as experiment_settings


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
                                                                                            # Load training, validation, and test data based on the experiment name
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
    # Load tropical data for the winter season
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon  = data_functions_schooner.load_tropic_data_winter(DATA_DIR, file_lon, file_lat, False)
    # Process the tropical data for training, validation, and testing
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
    print("Experiment name is not recognized.")
    quit()

# Create a mask for class identities based on prototypes
proto_class_mask = network.createClassIdentity(PROTOTYPES_PER_CLASS)

# Initialize arrays to hold prototypes for each class
prototypes_of_correct_class_train = np.zeros((len(y_train), NPROTOTYPES))
for i in range(0, prototypes_of_correct_class_train.shape[0]):
    prototypes_of_correct_class_train[i, :] = proto_class_mask[:, int(y_train[i])]
    
prototypes_of_correct_class_val   = np.zeros((len(y_val), NPROTOTYPES))    
for i in range(0, prototypes_of_correct_class_val.shape[0]):
    prototypes_of_correct_class_val[i, :] = proto_class_mask[:, int(y_val[i])]

prototypes_of_correct_class_test   = np.zeros((len(y_test), NPROTOTYPES))    
for i in range(0, prototypes_of_correct_class_test.shape[0]):
    prototypes_of_correct_class_test[i, :] = proto_class_mask[:, int(y_test[i])]
    
# Define the base model filename based on the experiment name
base_model_filename = model_dir + 'pretrained_model_' + EXP_NAME

if(EXP_NAME[:9] == 'GCM_SGold'):
    base_model_filename = './saved_models/GCM_alas_wint_583yrs_gold_redo/' + 'pretrained_model_' + 'GCM_alas_wint_583yrs_gold_redo'

# Load the pretrained model if specified in settings
if(settings['pretrain'] == True):
    base_model = common_functions.load_model(base_model_filename)

# Load the model for the current stage
model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(STAGE)
model = common_functions.load_model(model_filename)

# Get specific layers from the model
model_final_conv_layer = network.get_model_final_conv_layer(model)
model_prototype_layer  = network.get_model_prototype_layer(model)

# Calculate the local mask from the model's weights
local_mask = np.exp(model.layers[-3].get_weights()[1])

# Reload the network module to ensure the latest changes are applied
imp.reload(network)

# Build a CNN model with specified parameters
model_cnn_only = network.build_model(
    nlayers=NLAYERS,
    nfilters=NFILTERS,
    input_shape=X_train.shape[1:],
    output_shape=NCLASSES,
    prototypes_per_class=PROTOTYPES_PER_CLASS,
    network_seed=RANDOM_SEED,
    cnn_only=True,
    double_conv=DOUBLE_CONV,    
)

# Get the CNN-only convolution layer from the model
model_cnn_only = network.get_model_cnn_only_conv_layer(model_cnn_only)

# Calculate the receptive field of the model
print('Running receptive field calculation...')
receptive_field = network.ReceptiveField(model_cnn_only)
print('Receptive field calculation complete.')

# Prepare input for pushing prototypes
input_train = [[X_train, prototypes_of_correct_class_train]]

# Push prototypes into the model
imp.reload(push_prototypes)
model, push_info = push_prototypes.push(model, 
                                        input_train[0], 
                                        prototypes_of_correct_class_train,
                                        perform_push=True,
                                        batch_size=BATCH_SIZE_PREDICT,
                                        verbose=0,
                                    )
# Extract information from the push operation
prototype_sample  = push_info[0]
prototype_indices = push_info[-1]
similarity_scores = push_info[-2]

print("Push Info:")

# Save the prototype samples to a text file
np.savetxt(exp_data_dir + "_1_" + EXP_NAME + 'viz_push_protos.txt', prototype_sample, fmt='%d')

# Check if ERA5 plots are to be generated
era5_plots = (settings['plot_ERA5_translated'])

if(era5_plots):
    # Load and process ERA5 tropical data
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR, file_lon, file_lat, False)
    X_train, y_train, time_train_era5, X_val, y_val, time_val, X_test, y_test, time_test, temp_train, temp_val, temp_test = data_functions_schooner.get_and_process_tropic_data_winter_ERA5(labels,
                                                                                            data,
                                                                                            time,
                                                                                            rng,
                                                                                            train_yrs_era5,
                                                                                            val_yrs_era5,
                                                                                            test_years_era5,
                                                                                            temp_anoms,
                                                                                            translation=era5_plots,
                                                                                            colored=settings['colored'],
                                                                                            standardize=settings['standardize'],
                                                                                            shuffle=settings['shuffle'],
                                                                                            bal_data=settings['balance_data'],
                                                                                            r_seed=RANDOM_SEED,
                                                                                        )

    # Create class identity mask for ERA5 data
    proto_class_mask = network.createClassIdentity(PROTOTYPES_PER_CLASS)

    # Initialize arrays for prototypes of correct class for ERA5 data
    prototypes_of_correct_class_train = np.zeros((len(y_train), NPROTOTYPES))
    for i in range(0, prototypes_of_correct_class_train.shape[0]):
        prototypes_of_correct_class_train[i, :] = proto_class_mask[:, int(y_train[i])]
        
    prototypes_of_correct_class_val   = np.zeros((len(y_val), NPROTOTYPES))    
    for i in range(0, prototypes_of_correct_class_val.shape[0]):
        prototypes_of_correct_class_val[i, :] = proto_class_mask[:, int(y_val[i])]

    prototypes_of_correct_class_test   = np.zeros((len(y_test), NPROTOTYPES))    
    for i in range(0, prototypes_of_correct_class_test.shape[0]):
        prototypes_of_correct_class_test[i, :] = proto_class_mask[:, int(y_test[i])]
else:
    era5_plots = (settings['plot_ERA5_convert'])

# Print the model summary
model.summary()

# Prepare validation samples for prediction
input_val  = [[X_val, prototypes_of_correct_class_val]]

# Run predictions on validation data
y_predict_val = model.predict(input_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

# Evaluate model performance on validation data
model.evaluate(input_val, y_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

# Prepare testing samples if plotting is enabled
if(settings['plot_ERA5_convert'] == True):
    # Remove prototype samples from test data
    X_test = np.delete(X_test, np.unique(prototype_sample), axis=0)
    y_test = np.delete(y_test, np.unique(prototype_sample), axis=0)
    nino_time_test = np.delete(time_test, np.unique(prototype_sample), axis=0)

    # Initialize prototypes for the correct class in the test set
    prototypes_of_correct_class_test = np.zeros((len(y_test), NPROTOTYPES))    
    for i in range(0, prototypes_of_correct_class_test.shape[0]):
        prototypes_of_correct_class_test[i, :] = proto_class_mask[:, int(y_test[i])]

# Prepare input for testing
input_test = [[X_test, prototypes_of_correct_class_test]]

# Run predictions on test data
y_predict_test = model.predict(input_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)

# Evaluate model performance on test data
model.evaluate(input_test, y_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)

# Print accuracies by class
print('Accuracies by class: ')

# Process NINO data if plotting is enabled
if(settings['plot_ERA5_convert'] == True):
    nino_y_test = []
    era_years = [x.year for x in nino_time_test.values.astype("datetime64[D]").tolist()]
    era_months = [x.month for x in nino_time_test.values.astype("datetime64[D]").tolist()]
    nino_table = data_functions_schooner.process_nino_data(DATA_DIR)

    # Determine NINO phase for each test sample
    for i in np.arange(0, len(era_years), 1):
        nino_index = nino_table[era_years[i]-1870][era_months[i]]
        if(nino_index < -.4):
            nino_y_test.append(0)
        elif(nino_index > .4):
            nino_y_test.append(2)
        else:
            nino_y_test.append(1)

    # Calculate accuracy for NINO predictions
    nino_test_accur = []
    for i in np.arange(0, len(y_test), 1):
        if(nino_y_test[i] == y_test[i]):
            nino_test_accur.append(1)
        else:
            nino_test_accur.append(0)

    nino_final_accur = np.sum(nino_test_accur) / len(nino_test_accur)

# Initialize flag for training status
did_not_train = False

# Calculate and print accuracy for each class
for c in np.arange(0, NCLASSES):
    i = np.where(y_test == c)[0]
    j = np.where(y_test[i] == np.argmax(y_predict_test[i], axis=1))[0]
    acc = np.round(len(j) / len(i), 3)

    if(acc >= .995):
        did_not_train = True
    print(np.argmax(y_predict_test[i], axis=1))
    print('   phase ' + str(c) + ' = ' + str(acc))

# If pretraining is enabled, run base model predictions
if(settings['pretrain'] == True):
    base_y_predict_test = base_model.predict(X_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)
    base_y_predict_val = base_model.predict(X_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

    # Evaluate base model performance
    base_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)

    # Print accuracies by class for base model
    for c in np.arange(0, NCLASSES):
        i = np.where(y_test == c)[0]
        j = np.where(y_test[i] == np.argmax(base_y_predict_test[i], axis=1))[0]
        acc = np.round(len(j) / len(i), 3)
        print(np.argmax(base_y_predict_test[i], axis=1))
        print('   phase ' + str(c) + ' = ' + str(acc))

    base_y_predict = base_y_predict_test

# Prepare data for similarity map generation
y_predict = y_predict_test
y_true = y_test
input_data = input_test

# Reload push_prototypes module
imp.reload(push_prototypes)

# Get similarity maps
inputs_to_prototype_layer = model_final_conv_layer.predict(input_data)
prototypes = model.layers[-3].get_weights()[0]
similarity_scores = push_prototypes.get_similarity_maps(inputs_to_prototype_layer, 
                                                        prototypes, 
                                                        local_mask,
                                                        batch_size=BATCH_SIZE_PREDICT,
                                                    )

# Get winning similarity scores across maps for each prototype and sample
max_similarity_score = model_prototype_layer.predict(input_data)

# Get final weights from the model
w = np.round(model.layers[-2].get_weights()[0], 3)

# Set up plotting parameters for prototypes and samples
plt.rc('text', usetex=False)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']}) 
plt.rc('savefig', facecolor='white')
plt.rc('axes', facecolor='white')
plt.rc('axes', labelcolor='dimgrey')
plt.rc('xtick', color='dimgrey')
plt.rc('ytick', color='dimgrey')

# Function to adjust plot spines
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

# Prepare dates for prototypes and samples
prototype_date = time_train.dt.strftime("%b %d %Y").values[prototype_sample]
if(settings['plot_ERA5_convert'] == True):
    prototype_date = time_val.dt.strftime("%b %d %Y").values[prototype_sample]
sample_date = time_test.dt.strftime("%b %d %Y").values

# Function to generate confusion matrix
def make_confuse_matrix(y_predict, y_test, data_amount, base):
    # Generate the confusion matrix
    y_predict_class = np.argmax(y_predict, axis=1)
    cf_matrix = confusion_matrix(y_test, y_predict_class)
    cf_matrix_pred = confusion_matrix(y_test, y_predict_class, normalize='pred')
    cf_matrix_true = confusion_matrix(y_test, y_predict_class, normalize='true')
    cf_matrix = np.around(cf_matrix, 3)
    cf_matrix_pred = np.around(cf_matrix_pred, 3)
    cf_matrix_true = np.around(cf_matrix_true, 3)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf_matrix, cmap=plt.cm.Blues, alpha=0.3)

    correct_preds = 0
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            ax.text(x=j, y=i, s=cf_matrix[i, j], va='center', ha='center', size='xx-large')
            ax.text(x=j, y=i + .3, s=(str(np.around(cf_matrix_pred[i, j] * 100, 4)) + '%'), va='center', ha='center', size='xx-large', color='green')
            ax.text(x=j, y=i - .3, s=(str(np.around(cf_matrix_true[i, j] * 100, 4)) + '%'), va='center', ha='center', size='xx-large', color='red')

            if (i == j):
                correct_preds += cf_matrix[i, j]

    correct_preds /= np.sum(cf_matrix)
    
    plt.xlabel('Prediction', fontsize=18, color='green')
    plt.ylabel('Actual', fontsize=18, color='red')
    plt.title('TLLTT Confusion Matrix (Accuracy - ' + str(np.around(correct_preds * 100, 2)) + '%)', fontsize=18)
    
    # Save confusion matrix plot
    if (base):
        plt.savefig((vizualization_dir + "confusion_matrices/" + EXP_NAME + 'base_' + str(data_amount) + 'percent_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    else:
        plt.savefig((vizualization_dir + "confusion_matrices/" + EXP_NAME + 'TLLTT_' + str(data_amount) + 'percent_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()

    return np.around(correct_preds * 100, 2)


def precip_comps(best_samps, samps_bool, percent_samps = 100):
    # Loop through each temperature class (0 to 2)
    for temp_class in np.arange(0,3,1):
        if(samps_bool):
            # If samples boolean is true, filter best samples based on predicted class
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp])==temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = np.asarray(temp_best_samps)
        else:
            # Otherwise, find samples where predicted class matches true class
            isamples = np.where((np.argmax(y_predict,axis=1)==temp_class) & (np.argmax(y_predict,axis=1)==y_true))[0]
        
        if(isamples.shape[0] > 0):
            # Initialize image accumulation
            img = 0
            for i in np.arange(isamples.shape[0]):
                img += np.squeeze(input_data[0][0][isamples[i],:,:,0])
            img = img / isamples.shape[0]  # Average the images

            # Set up the figure for plotting
            fig = plt.figure(figsize=(20, 16))
            fig.tight_layout()
            spec = fig.add_gridspec(4, 5)
            plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

            # Create a subplot with a specific projection
            sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))
            plt.set_cmap('cmr.fusion')

            # Plot the contour of the averaged image
            img = sub1.contourf(np.asarray(lon), np.asarray(lat), img, np.linspace(-.75, .75, 41), transform=ccrs.PlateCarree())
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

            # Add coastlines and save the figure
            sub1.coastlines()
            plt.savefig((vizualization_dir + "precip_figs/" + "_" + str(percent_samps) + "_" + EXP_NAME + 'PRECIPphase' + str(temp_class)+ '_mjo.png'), bbox_inches='tight', dpi=dpiFig)

def timeseries_predictions():
    # Prepare arrays for plotting predictions over time
    y_predict_class_plot = np.argmax(y_predict,axis=1)
    y_predict_class_plot_low = y_predict_class_plot.astype('float64')
    y_predict_class_plot_avg = y_predict_class_plot.astype('float64')
    y_predict_class_plot_high = y_predict_class_plot.astype('float64')

    # Set NaN for specific classes to filter out in plots
    y_predict_class_plot_low[np.where(y_predict_class_plot_low==2)[0]] = np.nan
    y_predict_class_plot_low[np.where(y_predict_class_plot_low==1)[0]] = np.nan

    y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==2)[0]] = np.nan
    y_predict_class_plot_avg[np.where(y_predict_class_plot_avg==0)[0]] = np.nan

    y_predict_class_plot_high[np.where(y_predict_class_plot_high==1)[0]] = np.nan
    y_predict_class_plot_high[np.where(y_predict_class_plot_high==0)[0]] = np.nan
    
    # Plot predictions for each decade
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
        plt.close()

    # Plot predictions for each half-decade
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
        plt.close()


##################################################################################################################################################################################################################

def top_confidence_protos(percentage, predictions):
    temp_classes = [0,1,2]

    total_maxvals = []
    all_isamples = []
    for temp_class in temp_classes:

        isamples = np.where((np.argmax(predictions,axis=1)==temp_class))[0]
        
        high_scores = predictions[isamples,temp_class]

        maxvals = high_scores

        total_maxvals.append(maxvals)

        all_isamples.append(isamples)

    total_maxvals = np.concatenate((total_maxvals[0], total_maxvals[1], total_maxvals[2]))

    total_maxvals = np.asarray(total_maxvals)

    all_isamples = np.concatenate((all_isamples[0], all_isamples[1], all_isamples[2]))

    all_isamples = np.asarray(all_isamples)
    # Function to get best samples based on top scores
    topscore = np.asarray(hq.nlargest(int(total_maxvals.shape[0]*percentage), total_maxvals))
    toplocs = np.where(np.in1d(total_maxvals,topscore))[0]
    argstest = np.argsort(total_maxvals[toplocs])
    bestsamps = all_isamples[toplocs[argstest][::-1]]
    return bestsamps

    ##################################################################################################################################################################################################################

# Prototype examples
def examine_proto(good_samp, era5_flag):
    # Initialize map projection and reload plots
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    imp.reload(plots)

    # Load MJO information from a pickle file
    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'
    MJO_info = pd.read_pickle(f)

    # Extract MJO phases and amplitudes
    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']
    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))

    # Filter phases based on amplitude
    less_than_one = np.where(mjo_amp < 1)[0]
    phases[less_than_one] = 0
    y_predict_class = np.argmax(y_predict, axis=1)
    igrab_samples = np.where(y_predict_class == 1)[0]    

    # Define sample indices and variable settings
    SAMPLES = [good_samp]
    VAR_INDEX = [0]
    SORTED_VALUE = (1, 1, 1)
    colors = ('tab:purple', 'tab:orange')
    FS = 13

    # Create figure and grid for subplots
    fig = plt.figure(figsize=(10, 9.5), constrained_layout=True)
    grid_per_col = 7
    spec = gridspec.GridSpec(ncols=3 * grid_per_col, nrows=5, figure=fig)

    # Loop through each sample to plot
    for isample, sample in enumerate(SAMPLES):
        # Predict class and calculate similarity scores
        y_predict_class = int(np.argmax(y_predict[sample]))
        points = max_similarity_score[sample, :] * w[:, y_predict_class]
        all_points = w * np.squeeze(max_similarity_score[sample, :])[:, np.newaxis]
        total_points = np.sum(all_points, axis=0)        
        base_col = isample * grid_per_col
        
        # Determine variable name and letters for plots
        var_index = 0
        if var_index == 0:
            var_name = 'Precipitation'
            letters = ('(a)', '(b)', '(c)', '(d)')
        elif var_index == 1:
            var_name = 'u200'
            letters = ('(a)', '(d)', '(g)')            
        elif var_index == 2:
            var_name = 'u850'
            letters = ('(b)', '(e)', '(h)')      


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

        # Plot the sample
        ax_samp = fig.add_subplot(spec[0, base_col:base_col + grid_per_col], projection=mapProj)            
        similarity_map = similarity_scores[sample, :, :, prototype_index]
        j, k = np.unravel_index(np.argmax(similarity_map), shape=similarity_map.shape)
        rf = receptive_field.computeMask(j, k)   
        rf = np.abs(rf - 1.)
        rf[rf == 0] = np.nan

        # Plotting details for the sample
        img = np.squeeze(input_data[0][0][sample, :, :, var_index])
        p = plots.plot_sample_shaded(ax_samp, img, globe=True, lat=lat, lon=lon, mapProj=mapProj, rf=rf)
        ax_samp.set_title(letters[0] + ' ' + var_name + ' of Sample ', fontsize=FS)
        ax_samp.text(0.99, 1.0, str(sample_date[sample]), fontfamily='monospace', fontsize=FS, va='bottom', ha='right', transform=ax_samp.transAxes)

        # Class text based on true labels
        class_text = "cold class" if y_true[sample] == 0 else "neutral class" if y_true[sample] == 1 else "warm class"
        ax_samp.text(0.01, 1.0, class_text, fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax_samp.transAxes)

        # Plot the prototypes
        ax = fig.add_subplot(spec[1, base_col:base_col + grid_per_col], projection=mapProj)
        rf = receptive_field.computeMask(prototype_indices[prototype_index, 0], prototype_indices[prototype_index, 1])
        img = np.squeeze(input_val[0][0][prototype_sample[prototype_index], :, :, var_index]) * rf if settings['plot_ERA5_convert'] else np.squeeze(input_train[0][0][prototype_sample[prototype_index], :, :, var_index]) * rf
        p = plots.plot_sample(ax, img, globe=True, lat=lat, lon=lon, mapProj=mapProj)
        ax.set_title(letters[1] + ' ' + var_name + ' of Prototype ' + str(prototype_index), fontsize=FS)
        ax.text(0.99, 1.0, str(prototype_date[prototype_index]), fontfamily='monospace', fontsize=FS, va='bottom', ha='right', transform=ax.transAxes)

        # Class text for prototype
        class_text = "cold class" if y_train[prototype_sample[prototype_index]] == 0 else "neutral class" if y_train[prototype_sample[prototype_index]] == 1 else "warm class"
        ax.text(0.01, 1.0, str(class_text), fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax.transAxes)

        # Plot the masks
        ax = fig.add_subplot(spec[2, base_col:base_col + grid_per_col], projection=mapProj)
        ax.set_aspect("auto")
        img = local_mask[:, :, prototype_index] 
        p = plots.plot_mask(ax, img)
        p.set_clim(1., np.max(img))
        ax.set_title(letters[2] + ' ' + 'Prototype ' + str(prototype_index) + ' Location Scaling', fontsize=FS)

        # Plot the points
        ax = fig.add_subplot(spec[3, base_col + 1:base_col + grid_per_col])
        plt.axhline(y=0, color='.75', linewidth=.5)    
        plot_colors = []
        for phase in np.arange(0, 3):
            i = np.where(proto_class_mask[:, phase] == 0)[0]
            plt.plot(np.ones(len(i)) * phase, all_points[i, phase], marker='o', markeredgecolor='.5', markerfacecolor='w', markersize=3, markeredgewidth=.25)
            i = np.where(proto_class_mask[:, phase] == 1)[0]
            p = plt.plot(np.ones(len(i)) * phase, all_points[i, phase], '.')
            clr = p[0].get_color()
            plot_colors.append(clr)
            plt.text(phase, np.ceil(np.max(total_points)) + .1, str(np.round(total_points[phase], 1)), verticalalignment='bottom', horizontalalignment='center', color=clr, fontsize=12)

        # Set plot limits and labels
        plt.yticks((-1, 0, 1, 2, 3, 4, 5, 6), ('-1', '0', '1', '2', '3', '4', '5', '6'))
        plt.ylim(-1, np.ceil(np.max(total_points)) + .1)                     
        plt.xlim(-.5, 2.5)
        plt.xticks(np.arange(0, 3), ("cold", "neutral", "warm"))
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
        [t.set_color(i) for (i, t) in zip(np.asarray(plot_colors), ax.xaxis.get_ticklabels())]    
        
        for t in plt.xticks()[1]:
            t.set_fontsize(12)
            
        for t in plt.yticks()[1]:
            t.set_fontsize(12)
        
        # Plot temperature
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

    if(era5_flag == 0):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_GCM.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    elif(era5_flag == 1):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_ERA5projected.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    else:
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_' + str(SAMPLES[0]) + '_' + 'class' + str(y_predict_class) +'_3samples_prototypes_ERA5transfered_fixed.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()

##################################################################################################################################################################################################################

# Function to display all prototypes for each phase
def show_all_protos(era5_flag, percentage):
    # Import necessary libraries
    from scipy import stats
    imp.reload(plots)  # Reload plots module
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))  # Set up map projection
    FS = 12  # Font size for plots

    # Load MJO information from a pickle file
    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl'
    MJO_info = pd.read_pickle(f)

    # Extract phases and RMM values from MJO information
    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']

    # Calculate MJO amplitude
    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))
    less_than_one = np.where(mjo_amp < 1)[0]  # Identify samples with amplitude less than 1

    # Loop through each phase to create subplots
    for phase in np.arange(0,3):
        fig, axs = plt.subplots(10, 2, figsize=(18.3,22), subplot_kw={'projection': mapProj})

        # Get top samples based on confidence
        top_samps = top_confidence_protos(percentage, y_predict)

        # Print debugging information
        print(len(top_samps))
        print(y_predict.shape)
        print(y_true.shape)
        print(y_predict[top_samps].shape)

        # Identify samples that match the current phase
        isamples_top = np.where((np.argmax(y_predict[top_samps],axis=1)==phase) & (np.argmax(y_predict[top_samps],axis=1)==y_true[top_samps]))[0]
        isamples = top_samps[isamples_top]

        # Calculate points based on similarity scores and weights
        points = max_similarity_score[isamples,:]*w[:,phase]
        k = np.where(proto_class_mask[:,phase]==0)[0]
        points[:,k] = 0.  # Set points to zero where the mask is not active

        # Determine the winning prototype for each sample
        winning_prototype = np.argmax(points,axis=1)
        points_avg = np.mean(points,axis=0)  # Average points across samples
        proto_vector = np.where(proto_class_mask[:,phase]==1)[0]  # Get active prototypes
        proto_points_vector = points_avg[proto_vector]
        sorted_index = np.argsort(proto_points_vector)  # Sort prototypes by average points

        points_var = np.var(points,axis=0)  # Calculate variance of points

        # Loop through each variable to plot prototypes
        for ivar, var_index in enumerate([0]):
            # Set variable name based on index
            if(var_index==0):
                var_name = 'Precipitation'
            elif(var_index==1):
                var_name = 'Precipitation'
            elif(var_index==2):
                var_name = 'u850'        

            phase0_protos = []  # Initialize list to store phase 0 prototypes

            # Loop through sorted prototypes for plotting
            for iprototype, prototype_index in enumerate(proto_vector[np.flipud(sorted_index)]):
                # Plot the prototypes
                ax = axs[iprototype,ivar]
                ax.set_aspect("auto")
                rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])
                img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf

                if phase == 0:
                    phase0_protos.append(img)  # Store phase 0 prototypes
                img[img == 0] = np.nan  # Set zero values to NaN for plotting
                p = plots.plot_sample(ax, img, globe=True, lat=lat, lon=lon, mapProj=mapProj)

                # Set title and text for the plot
                class_text = "blah"
                if(phase == 0):
                    class_text = "cold class"
                elif(phase == 1):
                    class_text = "neutral class"
                else:
                    class_text = "warm class"
                ax.set_title(var_name + ' of Prototype ' + str(prototype_index), fontsize=FS*1.25)
                ax.text(0.01, 1.0, str(class_text), fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax.transAxes)

                # Display prototype date
                ax.text(0.99, 1.0, str(prototype_date[prototype_index]), fontfamily='monospace', fontsize=FS, va='bottom', ha='right', transform=ax.transAxes)

                # Plot the masks for the prototypes
                if var_index==0:
                    iwin = np.where(winning_prototype==prototype_index)[0]
                    if(len(winning_prototype>1)):
                        win_frac = np.round(len(iwin)/len(winning_prototype)*100, 2)
                    else:
                        win_frac = 0
                    
                    ax = axs[iprototype,1]
                    ax.set_aspect("auto")
                    img = local_mask[:,:,prototype_index]
                    p = plots.plot_mask(ax,img)
                    p.set_clim(1.,np.max(img[:]))
                    ax.set_title('Prototype ' + str(prototype_index) + ' Location Scaling', fontsize=FS*1.25)
                    
                    # Display average points and variance for the prototype
                    ax.text(0.00, 1.0, str(np.round(points_avg[prototype_index],1)) + ' pts.', fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax.transAxes)
                    ax.text(1.0, 1.0, 'σ² = ' + str(np.round(points_var[prototype_index],3)), fontfamily='monospace', fontsize=FS, va='bottom', ha='right', transform=ax.transAxes)

        # Save phase 0 prototypes if applicable
        if phase == 0:
            phase0_protos = np.asarray(phase0_protos)
            np.save(exp_data_dir + "_"+ EXP_NAME + 'ERA5_phase0_protos.npy', phase0_protos)
            print("test")

        # Save the figure based on the era5_flag
        if(era5_flag == 0):
            plt.savefig((vizualization_dir + str(percentage*100) + "_" + "_" + EXP_NAME + '_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        elif(era5_flag == 1):
            plt.savefig((vizualization_dir + 'era5_figs/' + str(percentage*100) + "_" + "_" + EXP_NAME + '_translated_allPrototypes_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        else:
            plt.savefig((vizualization_dir + 'era5_figs/' + str(percentage*100) + "_" + "_" + EXP_NAME + '_convert_allPrototypes_phase' + str(phase) + '_fixed.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
            
########################################################################################################################

def comps_by_proto(era5_flag):
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 12

    # Load MJO information from a pickle file
    f = DATA_DIR + 'Index_EOFS/MJO_CESM2-piControl_intialTEST.pkl' 
    MJO_info = pd.read_pickle(f)

    # Extract MJO phases and RMM values
    phases = MJO_info['Phase']
    rmm1 = MJO_info['RMM1']
    rmm2 = MJO_info['RMM2']
    mjo_amp = np.sqrt(np.square(rmm1) + np.square(rmm2))
    less_than_one = np.where(mjo_amp < 1)[0]

    # Loop through each MJO phase
    for phase in np.arange(0,3):
        fig, axs = plt.subplots(10, 2, figsize=(18.3,22), subplot_kw={'projection': mapProj})

        # Identify samples for the current phase
        isamples = np.where((np.argmax(y_predict,axis=1)==phase) & (np.argmax(y_predict,axis=1)==y_true))[0]
        points = max_similarity_score[isamples,:]*w[:,phase]
        k = np.where(proto_class_mask[:,phase]==0)[0]
        points[:,k] = 0.

        # Determine the winning prototype for each sample
        winning_prototype = np.argmax(points,axis=1)
        points_avg = np.mean(points,axis=0)
        proto_vector = np.where(proto_class_mask[:,phase]==1)[0]
        proto_points_vector = points_avg[proto_vector]
        sorted_index = np.argsort(proto_points_vector)
        points_var = np.var(points,axis=0)

        # Loop through each prototype to create plots
        for ivar, var_index in enumerate([0]):
            if(var_index==0):
                var_name = 'Precipitation'
            elif(var_index==1):
                var_name = 'Precipitation'
            elif(var_index==2):
                var_name = 'u850'        

            for iprototype, prototype_index in enumerate(proto_vector[np.flipud(sorted_index)][:4]):
                # Plot the prototypes
                ax = axs[iprototype,ivar]
                ax.set_aspect("auto")
                rf = receptive_field.computeMask(prototype_indices[prototype_index,0], prototype_indices[prototype_index,1])
                img = np.squeeze(input_train[0][0][prototype_sample[prototype_index],:,:,var_index])*rf
                img[img == 0] = np.nan
                p = plots.plot_sample(ax, img, globe=True, lat=lat, lon=lon, mapProj=mapProj)

                # Set title and text for the plot
                class_text = "blah"
                if(phase == 0):
                    class_text = "cold class"
                elif(phase == 1):
                    class_text = "neutral class"
                else:
                    class_text = "warm class"
                ax.set_title(var_name + ' of Prototype ' + str(prototype_index), fontsize=FS*1.25)
                ax.text(0.01, 1.0, str(class_text), fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax.transAxes)

                # Plot the masks for the current variable
                if var_index==0:
                    samp_proto_index = np.where(winning_prototype == prototype_index)[0]
                    samp_proto_locs = isamples[samp_proto_index]
                    img = np.zeros(np.squeeze(input_data[0][0][samp_proto_locs[0],:,:,var_index]).shape)

                    # Average the samples for the prototype
                    for samp_proto_loc in samp_proto_locs:
                        samp_img = np.squeeze(input_data[0][0][samp_proto_loc,:,:,var_index])
                        img = img + samp_img
                    img = img / samp_proto_locs.shape[0]

                    # Plot the averaged sample
                    ax = axs[iprototype,1]
                    ax.set_aspect("auto")
                    p = plots.plot_sample(ax, img * 5, globe=True, lat=lat, lon=lon, mapProj=mapProj)
                    ax.set_title('Prototype ' + str(prototype_index) + ' Sample Composite', fontsize=FS*1.25)
                    ax.text(0.00, 1.0, 'n = ' + str(samp_proto_locs.shape[0]), fontfamily='monospace', fontsize=FS, va='bottom', ha='left', transform=ax.transAxes)

        # Save the figure based on the era5_flag
        if(era5_flag == 0):
            plt.savefig((vizualization_dir + "_" + EXP_NAME + '_Prototype_comps_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        elif(era5_flag == 1):
            plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_translated_Prototype_comps_phase' + str(phase) + '.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()
        else:
            plt.savefig((vizualization_dir + 'era5_figs/' + "_" + EXP_NAME + '_convert_Prototype_comps_phase' + str(phase) + '_fixed.png'), bbox_inches='tight', dpi=dpiFig)
            plt.close()

def proto_rankings(best_samps, samps_bool, percent_samps = 100):
    from scipy import stats
    imp.reload(plots)
    mapProj = ccrs.PlateCarree(central_longitude = np.mean(lon))
    FS = 5

    # Calculate number of prototypes per class
    protos_num = NPROTOTYPES / NCLASSES

    for temp_class in np.arange(0, 3):
        # Initialize ranking matrices
        proto_ranks = np.zeros((int(protos_num), int(protos_num)))
        proto_ranks_full = np.zeros((int(protos_num), int(protos_num)))

        # Create subplots for visualizing rankings
        fig, axs = plt.subplots(10, 2, figsize=(12, 22), subplot_kw={'projection': mapProj})

        # Select samples based on conditions
        if(samps_bool):
            temp_best_samps = []
            for spec_samp in best_samps:
                if (np.argmax(y_predict[spec_samp]) == temp_class):
                    temp_best_samps.append(spec_samp)
            isamples = temp_best_samps
        else:
            isamples = np.where((np.argmax(y_predict, axis=1) == temp_class) & (np.argmax(y_predict, axis=1) == y_true))[0]

        # Calculate points for ranking
        points = max_similarity_score[isamples, :] * w[:, temp_class]
        k = np.where(proto_class_mask[:, temp_class] == 0)[0]
        points[:, k] = 0.

        # Rank prototypes based on similarity scores
        for i in np.arange(0, points.shape[0], 1):
            class_points = points[int(i)][int(temp_class * protos_num):int((temp_class * protos_num) + protos_num)]
            class_scores = np.argsort(class_points)[::-1]
            for j in np.arange(0, class_points.shape[0], 1):
                proto_ranks[class_scores[int(j)]][int(j)] += 1

        # Repeat ranking for full sample set
        isamples_full = np.where((np.argmax(y_predict, axis=1) == temp_class))[0]
        points_full = max_similarity_score[isamples_full, :] * w[:, temp_class]
        k_full = np.where(proto_class_mask[:, temp_class] == 0)[0]
        points_full[:, k_full] = 0.

        for i in np.arange(0, points_full.shape[0], 1):
            class_points_full = points_full[int(i)][int(temp_class * protos_num):int((temp_class * protos_num) + protos_num)]
            class_scores_full = np.argsort(class_points_full)[::-1]
            for j in np.arange(0, class_points_full.shape[0], 1):
                proto_ranks_full[class_scores_full[int(j)]][int(j)] += 1

        # Define rank names for visualization
        rank_names = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        rank_names = np.asarray(rank_names)

        # Plot rankings
        fig, axs = plt.subplots(1, 1, figsize=(21 / 2.5, 12 / 2.5))
        n = len(proto_ranks)
        _rank_names = np.arange(len(rank_names))
        width = .8

        # Bar plot for prototype rankings
        for i in range(n):
            plt.bar(_rank_names - width + i / float(n) * width, proto_ranks[i], 
                    width=width / float(n), align="edge", label=("Prototype " + str(i + (temp_class * 10))))
        plt.xticks(_rank_names, rank_names)
        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + " percent - Class " + str(temp_class) + " Prototype Rankings " + "(n=" + str(points.shape[0]) + ")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) + "class_" + str(temp_class) + "_counts" + '.png'), bbox_inches='tight', dpi=dpiFig)

        # Plot frequency of rankings
        fig, axs = plt.subplots(1, 1, figsize=(21 / 2.5, 12 / 2.5))
        for i in range(n):
            plt.bar(_rank_names - width + i / float(n) * width, proto_ranks[i] / points.shape[0], 
                    width=width / float(n), align="edge", label=("Prototype " + str(i + (temp_class * 10))))
        plt.xticks(_rank_names, rank_names)
        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + " percent - Class " + str(temp_class) + " Prototype Rankings " + "(n=" + str(points.shape[0]) + ")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) + "class_" + str(temp_class) + "_freq" + '.png'), bbox_inches='tight', dpi=dpiFig)

        # Plot difference in frequency of rankings
        fig, axs = plt.subplots(1, 1, figsize=(21 / 2.5, 12 / 2.5))
        for i in range(n):
            plt.bar(_rank_names - width + i / float(n) * width, proto_ranks[i] / points.shape[0] - proto_ranks_full[i] / points_full.shape[0], 
                    width=width / float(n), align="edge", label=("Prototype " + str(i + (temp_class * 10))))
        plt.xticks(_rank_names, rank_names)
        plt.legend(fontsize=5)
        plt.title(str(percent_samps) + " percent - Class " + str(temp_class) + " Prototype Rankings " + "(n=" + str(points.shape[0]) + ")")
        plt.savefig((vizualization_dir + "rank_figs/" + str(percent_samps) + "class_" + str(temp_class) + "_DIFF_freq" + '.png'), bbox_inches='tight', dpi=dpiFig)

def compare_accuracies():
    # Define seeds for different runs
    run_seeds = [112,117]
    run_name = "GCM_alas_wint_550yrs_shuf_bal_seed"
    normal_fn = "normal_" + run_name + "125_TLLTT_accuracy.txt"
    
    # Load normal accuracies from file
    normal_accuracies = np.loadtxt("/barnes-engr-scratch1/nicojg/data/" + run_name + "125/" + normal_fn)

    # Create a plot for accuracies
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(10, 101, 5)[::-1], normal_accuracies, label = "ProtoLNet", color = '#f42c94')

    # Loop through each run seed to load and plot base CNN accuracies
    for run_seed in run_seeds:
        normal_fn = "normal_" + run_name + str(run_seed) + "_TLLTT_accuracy.txt"
        normal_accuracies = np.loadtxt("/barnes-engr-scratch1/nicojg/data/" + run_name + str(run_seed) + "/" + normal_fn)
        plt.plot(np.arange(10, 101, 5)[::-1], normal_accuracies, label = "Base CNN", color = '#f89c04')

    # Set plot title and labels
    plt.title("Discard test", fontsize=20)
    plt.xlabel("Percentage samples not discarded", fontsize=15)
    plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1])
    plt.ylabel("Accuracy (%)", fontsize=15)
    
    # Highlight area under the curve
    plt.axhspan(0, 33, color='0.75', alpha=0.5, lw=0)
    plt.ylim(bottom=30)

    # Save the figure
    plt.savefig(("figures/misc/failure_plot.png"), bbox_inches='tight', dpi=400)

def plot_alaksa_temp(samp, era5_flag):
    # Get temperature anomalies and coordinates
    temp_anoms, t_lon, t_lat = data_functions_schooner.get_temp_anoms(DATA_DIR)

    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 9.5))
    fig.tight_layout()

    # Set up a grid for subplots
    spec = fig.add_gridspec(4, 5)
    plt.subplots_adjust(wspace= 0.35, hspace= 0.25)

    # Create a subplot with a specific projection
    sub1 = fig.add_subplot(111, projection = ccrs.PlateCarree(central_longitude=180))

    # Plot temperature anomalies
    plt.set_cmap('cmr.fusion_r')
    img = sub1.contourf(np.asarray(t_lon), np.asarray(t_lat), np.asarray(temp_test[samp,:,:]), np.linspace(-20, 20, 41), transform=ccrs.PlateCarree(), extend='both')
    
    # Add geographical features
    coast = cfeature.GSHHSFeature(scale='intermediate')
    state = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale='10m', facecolor="none")
    countries = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_0_boundary_lines_land', scale='10m', facecolor="none")
    sub1.add_feature(coast, edgecolor='0')
    sub1.add_feature(state, edgecolor='0')
    sub1.add_feature(countries, edgecolor='0')

    # Set subplot parameters
    sub1.tick_params(axis='both', labelsize=12)
    sub1.set_xticks(np.arange(-180,181,30))
    sub1.set_xticklabels(np.concatenate((np.arange(0,181,30),np.arange(-160,1,30))))
    sub1.set_yticks(np.arange(-90,91,15))
    sub1.set_xlim(10,75)
    sub1.set_ylim(30,75)
    sub1.set_xlabel("Longitude (degrees)",fontsize=14)
    sub1.set_ylabel("Latitude (degrees)",fontsize=14)
    sub1.set_title("(e) 14-day lead time temperature anomalies", fontsize=14)
    
    # Add colorbar for temperature
    cbar = plt.colorbar(img,shrink=.75, aspect=20*0.8)
    cbar.set_label("Temperature (K)", fontsize=14)
    cbar.ax.tick_params(labelsize=12) 
    
    # Save the figure based on era5_flag
    if(era5_flag == 0):
        plt.savefig((vizualization_dir + "individual_protos/" + EXP_NAME + '_real_temp_' + str(samp) + '_' + 'coast_plot.png'), bbox_inches='tight', dpi=dpiFig)
        plt.close()
    elif(era5_flag == 1):
        plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_translated_real_temp' + str(samp) + '_' + 'coast_plot.png'), bbox_inches='tight', dpi=dpiFig)
    else:
        plt.savefig((vizualization_dir + 'era5_figs/'  + "_" + EXP_NAME + '_convert_real_temp' + str(samp) + '_' + 'coast_plot_fixed.png'), bbox_inches='tight', dpi=dpiFig)

# Initialize accuracy lists
accuracies = []
base_accuracies = []
accuracies_val = []
base_accuracies_val = []

# Set era5_flag based on settings
era5_flag_set = 0
if(settings['plot_ERA5_translated'] == True):
    era5_flag_set = 1 
if(settings['plot_ERA5_convert'] == True):
    era5_flag_set = 2

# Get top samples based on confidence
top_samps = top_confidence_protos(1, y_predict)[:5]

# Loop through percentages to compute accuracies
for i in np.arange(10, 101, 5):
    if(settings['pretrain'] == True):
        base_accuracies.append(make_confuse_matrix(base_y_predict[top_confidence_protos(i/100., base_y_predict_test)], y_true[top_confidence_protos(i/100., base_y_predict_test)], i, True))
    accuracies.append(make_confuse_matrix(y_predict[top_confidence_protos(i/100., y_predict)], y_true[top_confidence_protos(i/100., y_predict)], i, False))

    if(settings['pretrain'] == True):
        base_accuracies_val.append(make_confuse_matrix(base_y_predict_val[top_confidence_protos(i/100., base_y_predict_val)], y_val[top_confidence_protos(i/100., base_y_predict_val)], i, True))
    accuracies_val.append(make_confuse_matrix(y_predict_val[top_confidence_protos(i/100., y_predict_val)], y_val[top_confidence_protos(i/100., y_predict_val)], i, False))

# Plotting accuracies
plt.figure(figsize=(10,6))
plt.plot(np.arange(10, 101, 5)[::-1], accuracies, label = "ProtoLNet", color = '#f42c94')
if(settings['pretrain'] == True):
    plt.plot(np.arange(10, 101, 5)[::-1], base_accuracies, label = "Base CNN", color = '#f89c04')

# Set plot labels and limits
plt.xlabel("Percentage samples not discarded", fontsize=15)
plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=15)
plt.axhspan(0, 33, color='0.75', alpha=0.5, lw=0)

# Adjust y-limits based on accuracy conditions
if(settings['pretrain'] == True):
    if((np.min(accuracies) >= 31) and (np.min(base_accuracies) >= 31)):
        plt.ylim(bottom=30)
    else:
        plt.ylim(bottom=20)
else:
    if((np.min(accuracies) >= 31)):
        plt.ylim(bottom=30)
    else:
        plt.ylim(bottom=20)

# Save the final plot based on era5_flag
if(not era5_plots):
    plt.savefig((vizualization_dir + EXP_NAME + '_forecast_of_opportunity.png'), bbox_inches='tight', dpi=dpiFig)
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

np.savetxt(exp_data_dir + "_" + EXP_NAME + 'rps_score_top20.txt', [top20_rps_score], fmt='%f')


quit()
top_samps = top_confidence_protos(1, y_predict)[:5]
for decent_samp in top_samps:
    # mjo_correlation(decent_samp)
    examine_proto(decent_samp, era5_flag_set)
    plot_alaksa_temp(decent_samp, era5_flag_set)