# import py3nvml
# py3nvml.grab_gpus(num_gpus=1, gpu_select=[2])


# This Looks Like That There - S2S Release Version
# Pretrain CNN Only

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

import sys
import time
import imp 
import numpy, warnings
numpy.warnings = warnings

import numpy as np
from icecream import ic
import scipy.io as sio

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns


import tensorflow as tf

import random

from sklearn.metrics import confusion_matrix

import network as network
import data_functions_schooner
import common_functions


__author__ = "Nicolas Gordillo"
__version__ = "April 2024"

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
dpiFig = 300.

# ## Print the detailed system info

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"tensorflow version = {tf.__version__}")

# ## Define experiment settings and directories
file_lon = 0
file_lat = 0

if len(sys.argv) < 2:
    EXP_NAME = 'GCM_alas_wint_550yrs_shuf_bal_seed125_redux'
    file_lon = 89
    file_lat = 64
    import experiment_settings_shuf_550bal_seeds as experiment_settings

elif len(sys.argv) == 2:
    num = int(sys.argv[1])
    EXP_NAME = 'GCM_alas_lr_wint_550redo_seed'+str(num) #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'
    import experiment_settings_multiple_seeds_lr_redo as experiment_settings

else:
    file_lon = int(sys.argv[2])
    file_lat = int(sys.argv[1])
    EXP_NAME = 'GCM_'+ str(file_lon) + '_' + str(file_lat) +'_wint_550yrs_shuf_bal_seed131'
    import experiment_settings_coast_550_lr_adjust_131 as experiment_settings

print(EXP_NAME)

imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

imp.reload(common_functions)
#model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_schooner(EXP_NAME)
# model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories(EXP_NAME)
model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_falco(EXP_NAME)

# ## Define the network parameters


RANDOM_SEED          = settings['random_seed']
BATCH_SIZE           = settings['batch_size']
NLAYERS              = settings['nlayers']
NFILTERS             = settings['nfilters']   
DOUBLE_CONV          = settings['double_conv']   
assert(len(NFILTERS)==NLAYERS)

NCLASSES             = settings['nclasses']
PROTOTYPES_PER_CLASS = settings['prototypes_per_class']
NPROTOTYPES          = np.sum(PROTOTYPES_PER_CLASS) 

NEPOCHS              = settings['nepochs_pretrain']
LR_INIT              = settings['lr_pretrain']
LR_CALLBACK_EPOCH    = settings['lr_cb_epoch_pretrain']
PATIENCE             = 100


# ## Initialize
gpus = tf.config.list_physical_devices('GPU')

# the next line will restrict tensorflow to the first GPU 
# you can select other gpus from the list instead
tf.config.set_visible_devices(gpus[1], 'GPU')

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
print(DATA_DIR)

train_yrs = settings['train_yrs']
val_yrs = settings['val_yrs']
test_years = settings['test_yrs']

train_yrs_era5 = settings['train_yrs_era5']
val_yrs_era5 = settings['val_yrs_era5']
test_years_era5 = settings['test_yrs_era5']
if(EXP_NAME[:3]=='ERA'):   

    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR, file_lon, file_lat, False)

    X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test = data_functions_schooner.get_and_process_tropic_data_winter_ERA5(labels,
                                                                                            data,
                                                                                            time,
                                                                                            rng,
                                                                                            train_yrs_era5,
                                                                                            val_yrs_era5,
                                                                                            test_years_era5,
                                                                                            colored=settings['colored'],
                                                                                            standardize=settings['standardize'],
                                                                                            shuffle=settings['shuffle'],
                                                                                            bal_data = settings['balance_data'],
                                                                                            r_seed = RANDOM_SEED,
                                                                                        )

elif(EXP_NAME[:3] == 'GCM'):

    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter(DATA_DIR, file_lon, file_lat, False)

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
# Check if the experiment name is recognized
else:
    print("Expermient name is not recognized.")
    quit()

# Create a class identity mask for prototypes
proto_class_mask = network.createClassIdentity(PROTOTYPES_PER_CLASS)

# Initialize arrays for prototypes of the correct class for training, validation, and testing
prototypes_of_correct_class_train = np.zeros((len(y_train),NPROTOTYPES))
for i in range(0,prototypes_of_correct_class_train.shape[0]):
    prototypes_of_correct_class_train[i,:] = proto_class_mask[:,int(y_train[i])]
    
prototypes_of_correct_class_val   = np.zeros((len(y_val),NPROTOTYPES))    
for i in range(0,prototypes_of_correct_class_val.shape[0]):
    prototypes_of_correct_class_val[i,:] = proto_class_mask[:,int(y_val[i])]

prototypes_of_correct_class_test   = np.zeros((len(y_test),NPROTOTYPES))    
for i in range(0,prototypes_of_correct_class_test.shape[0]):
    prototypes_of_correct_class_test[i,:] = proto_class_mask[:,int(y_test[i])]

# ## Define the training callbacks and metrics

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < LR_CALLBACK_EPOCH:
        return np.round(lr,8)
    else:
        if(epoch % 2 == 0):
            return lr/2.
        else:
            return lr

# Define learning rate callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)    
    
# Early stopping callback to prevent overfitting
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', 
    mode='max',
    patience=settings['pretrain_patience'], 
    restore_best_weights=True, 
    verbose=1
)

# List of callbacks to be used during training
callbacks_list = [
#     lr_callback,
    es_callback,
]            

# List of metrics to evaluate the model
metrics_list = [
    tf.keras.metrics.SparseCategoricalAccuracy(),
]

# Reload experiment settings
imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

# Reload common functions and get directories for saving models and diagnostics
imp.reload(common_functions)
model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_falco(EXP_NAME)

# Set random seed and batch sizes from settings
RANDOM_SEED          = settings['random_seed']
BATCH_SIZE_PREDICT   = settings['batch_size_predict']
BATCH_SIZE           = settings['batch_size']
NLAYERS              = settings['nlayers']
NFILTERS             = settings['nfilters']   
DOUBLE_CONV          = settings['double_conv']   
assert(len(NFILTERS)==NLAYERS)

# Set number of classes and prototypes
NCLASSES             = settings['nclasses']
PROTOTYPES_PER_CLASS = settings['prototypes_per_class']
NPROTOTYPES          = np.sum(PROTOTYPES_PER_CLASS)

# Set training parameters
NEPOCHS              = settings['nepochs_pretrain']
LR_INIT              = settings['lr_pretrain']
LR_CALLBACK_EPOCH    = settings['lr_cb_epoch_pretrain']
PATIENCE             = 100

# ## Instantiate the model

# Reload the network module and clear the Keras session
__ = imp.reload(network)
tf.keras.backend.clear_session()

# Print debug information
print("BLAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print(X_train.shape)

# Build the model with specified parameters
model = network.build_model(
    cnn_only=True,    
    nlayers=NLAYERS,
    nfilters=NFILTERS,
    input_shape= X_train.shape[1:],
    output_shape=NCLASSES,
    prototypes_per_class=PROTOTYPES_PER_CLASS,
    network_seed=RANDOM_SEED,
    double_conv=DOUBLE_CONV,
    dense_nodes=settings['dense_nodes'],
    prototype_channels=settings['prototype_channels'],
    kernel_l1_coeff=settings['kernel_l1_coeff'],
    kernel_l2_coeff=settings['kernel_l2_coeff'],
    drop_rate=settings['drop_rate'],
    drop_rate_final=settings['drop_rate_final'],    
    coeff_cluster=0.0,#settings['coeff_cluster'],
    coeff_separation=0.0,#settings['coeff_separation'],
    coeff_l1=0.0,#settings['coeff_l1'],    
)
# Display the model summary
model.summary()

# Print the proportion of class 0 in the training set
__ = ic(len(np.where(y_train==0)[0])/len(y_train))

# Compile the model with optimizer, loss function, and metrics
print('learning rate = ' + str(LR_INIT))

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LR_INIT,
#         beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    ),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = metrics_list,
)

# Train the model with the training data
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val,y_val),
    batch_size=BATCH_SIZE,
    epochs=NEPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=callbacks_list
)

# Save the trained model to the specified filename
model_filename = model_dir + 'pretrained_model_' + EXP_NAME
common_functions.save_model(model, model_filename)

#-------------------------------
# Display the results
best_epoch = np.argmin(history.history['val_loss'])

#---- plot loss and errors ----
trainColor = (117/255., 112/255., 179/255., 1.)
valColor = (231/255., 41/255., 138/255., 1.)
FS = 7
MS = 4

# Create subplots for loss and accuracy
plt.subplots(1,2,figsize=(10, 3))

# Plot training and validation loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'o-', color=trainColor, label='training loss', markersize=MS)
plt.plot(history.history['val_loss'], 'o-', color=valColor, label='validation loss', markersize=MS)
# plt.axvline(x=best_epoch, linestyle = '--', color='tab:gray')
plt.title("Loss Function")
plt.ylabel('average loss')
plt.xlabel('epoch')
plt.grid(False)
plt.legend(frameon=True, fontsize=FS)
plt.xlim(-.1, 30+1)

# Plot training and validation accuracy
plt.subplot(1,2,2)
plt.plot(history.history['sparse_categorical_accuracy'], 'o-', color=trainColor, label='training loss', markersize=MS)
plt.plot(history.history['val_sparse_categorical_accuracy'], 'o-', color=valColor, label='validation loss', markersize=MS)
# plt.axvline(x=best_epoch, linestyle = '--', color='tab:gray')
plt.title("Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(False)
plt.legend(frameon=True, fontsize=FS)
plt.xlim(-.1, 30+1)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(model_diagnostics_dir + 'loss_history_pretrained_model_' + EXP_NAME + '.png', dpi=dpiFig)

# Set up for plotting with a white background
########################################################################################################################
plt.rc('text',usetex=False)
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='white')
plt.rc('axes',facecolor='white')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('xtick',color='dimgrey')
plt.rc('ytick',color='dimgrey')
################################  
################################  
# Function to adjust the spines of the plot
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

# Load pretrained weights for convolutional layers
for layer in range(1,len(model.layers)):
    if(model.layers[layer].name[:4]=='conv'):
        print('   loading pretrained weights for --> ' + model.layers[layer].name)

# Make predictions on the test dataset
print('running model.predict()...')
y_predict_test = model.predict(X_test, batch_size=BATCH_SIZE_PREDICT, verbose=1)
print('model.predict() complete.')

# Make predictions on the training dataset
print('running model.predict()...')
y_predict_train = model.predict(X_train, batch_size=BATCH_SIZE_PREDICT, verbose=1)
print('model.predict() complete.')

# Evaluate the model on the test dataset
model.evaluate(X_test,y_test,batch_size=BATCH_SIZE_PREDICT, verbose=1)

# Print accuracies by class for the test dataset
print('Accuracies by class: ')

for c in np.arange(0,NCLASSES):
    i = np.where(y_test==c)[0]
    j = np.where(y_test[i]==np.argmax(y_predict_test[i],axis=1))[0]
    acc = np.round(len(j)/len(i),3)
    # print(np.argmax(y_predict_test[i],axis=1))
    
    print('   phase ' + str(c) + ' = ' + str(acc))

# Print accuracies by class for the training dataset
for c in np.arange(0,NCLASSES):
    i = np.where(y_train==c)[0]
    j = np.where(y_train[i]==np.argmax(y_predict_train[i],axis=1))[0]
    acc = np.round(len(j)/len(i),3)
    # print(np.argmax(y_predict_train[i],axis=1))
    
    print('   phase ' + str(c) + ' = ' + str(acc))
    

#-------------
# Set predictions and true labels for confusion matrix
y_predict  = y_predict_test
y_true     = y_test

# y_predict  = y_predict_train
y_true_train    = y_train

# time       = time_val
# input_data = input_val
#-------------

# Get predicted classes from the predictions
y_predict_class = np.argmax(y_predict,axis=1)

# Compute confusion matrices
cf_matrix = confusion_matrix(y_test, y_predict_class)
cf_matrix_pred = confusion_matrix(y_test, y_predict_class, normalize='pred')
cf_matrix_true = confusion_matrix(y_test, y_predict_class, normalize='true')
cf_matrix = np.around(cf_matrix,3)
cf_matrix_pred = np.around(cf_matrix_pred,3)
cf_matrix_true = np.around(cf_matrix_true,3)

# Create a plot for the confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cf_matrix, cmap=plt.cm.Blues, alpha=0.3)

# Annotate the confusion matrix with values
correct_preds = 0
for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(x=j, y=i,s=cf_matrix[i, j], va='center', ha='center', size='xx-large')
        ax.text(x=j, y=i+.3,s=(str(np.around(cf_matrix_pred[i, j]*100,4))+'\%'), va='center', ha='center', size='xx-large', color = 'green')
        ax.text(x=j, y=i-.3,s=(str(np.around(cf_matrix_true[i, j]*100,4))+'\%'), va='center', ha='center', size='xx-large', color = 'red')

        if (i == j):
            correct_preds += cf_matrix[i, j]

# Calculate overall accuracy from the confusion matrix
correct_preds /= np.sum(cf_matrix)

# Set labels and title for the confusion matrix plot
plt.xlabel('Prediction', fontsize=18, color = 'green')
plt.ylabel('Actual', fontsize=18, color = 'red')
plt.title('TLLTT Confusion Matrix (Accuracy - ' + str(np.around(correct_preds*100,2)) + '\%)', fontsize=18)
plt.savefig((vizualization_dir + EXP_NAME + 'BaseCNN_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
# plt.savefig((vizualization_dir + str(learning_rate) + "_" + EXP_NAME + 'BaseCNN_confmatrix.png'), bbox_inches='tight', dpi=dpiFig)
