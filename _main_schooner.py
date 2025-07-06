# This Looks Like That There - S2S Release Version
# 
# Main training notebook.

import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
import sys
import time
import imp #imp.reload(module)

import numpy, warnings
numpy.warnings = warnings
import numpy as np
from tqdm import trange
from icecream import ic

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

import tensorflow as tf
import random

import network
# import experiment_settings_coast_550
import data_functions_schooner
import push_prototypes
import plots
import common_functions

import heapq as hq

from sklearn.metrics import confusion_matrix



7
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
    EXP_NAME = 'GCM_alas_lr_wint_550yrs_seed128_redo'  #'GCM_alas_lr_wint_550yrs_seed144_nopre_fake128' #'GCM_alas_wint_550yrs_seed105_21days'#'smaller_test'#'quadrants_testcase'
    file_lon = 89
    file_lat = 64
    # import experiment_settings_shuf_550bal_seeds as experiment_settings
    import experiment_settings_multiple_seeds_lr as experiment_settings
elif len(sys.argv) == 2:
    num = int(sys.argv[1])
    EXP_NAME = 'GCM_alas_lr_wint_550redo_seed'+str(num) #+ '_nopre' #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'
    import experiment_settings_multiple_seeds_lr_redo as experiment_settings

    # learning_rate = float(sys.argv[1])
    # print(learning_rate)
    # EXP_NAME = 'GCM_alas_lr_wint_550yrs_seed144_nopre_lrtest_epochs' #GCM_alas_lr_wint_550yrs_seed147_nopre_lrtest'
    # import experiment_settings_multiple_seeds_lr as experiment_settings
    # import experiment_settings_shuf_550bal_seeds as experiment_settings
    file_lon = 89
    file_lat = 64


    # num = int(sys.argv[1])
    # EXP_NAME = 'GCM_alas_wint_550yrs_shuf_bal_seed'+str(num) #balanced_test'#initial_test'#'mjo'#'quadrants_testcase'
else:
    file_lon = int(sys.argv[2])
    file_lat = int(sys.argv[1])
    EXP_NAME = 'GCM_'+ str(file_lon) + '_' + str(file_lat) +'_wint_550yrs_shuf_bal_seed131'
    import experiment_settings_coast_550_lr_adjust_131 as experiment_settings

imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

imp.reload(common_functions)
# model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_schooner(EXP_NAME)
#model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories(EXP_NAME)
model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir = common_functions.get_exp_directories_falco(EXP_NAME)

# ## Define the network parameters

RANDOM_SEED          = settings['random_seed']
BATCH_SIZE_PREDICT   = settings['batch_size_predict']
BATCH_SIZE           = settings['batch_size']
NLAYERS              = settings['nlayers']
NFILTERS             = settings['nfilters']   
assert(len(NFILTERS)==NLAYERS)

NCLASSES             = settings['nclasses']
PROTOTYPES_PER_CLASS = settings['prototypes_per_class']
NPROTOTYPES          = np.sum(PROTOTYPES_PER_CLASS)

NEPOCHS              = settings['nepochs']
LR_INIT              = settings['lr']
LR_CALLBACK_EPOCH    = settings['lr_cb_epoch']
PATIENCE             = 100

EARLY_STOPPING       = settings['es_stop']

# LR_INIT = learning_rate

# ## Initialize

gpus = tf.config.list_physical_devices('GPU')

# the next line will restrict tensorflow to the first GPU 
# you can select other gpus from the list instead
tf.config.set_visible_devices(gpus[1], 'GPU')

tf.config.list_logical_devices('GPU')
tf.keras.backend.clear_session()
np.random.seed(RANDOM_SEED)

rng = np.random.default_rng(RANDOM_SEED)
# rng = np.random.default_rng(128)

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

train_yrs_era5 = settings['train_yrs_era5'],
val_yrs_era5 = settings['val_yrs_era5'],
test_years_era5 = settings['test_yrs_era5'],

if(EXP_NAME[:3]=='ERA'):   
    #labels, data, lat, lon, time = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR)

    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR, file_lon, file_lat, False)
    # Load training, validation, and test data based on the experiment name
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
    # Load tropical data for GCM experiments
    labels, data, lat, lon, time, temp_anoms, t_lat, t_lon = data_functions_schooner.load_tropic_data_winter(DATA_DIR, file_lon, file_lat, False)

    # Process the loaded data for training, validation, and testing
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
                                                                                            # r_seed = 128,
                                                                                )
else:
    print("Expermient name is bad")  # Handle invalid experiment name
    quit()

# Create a mask for class identities based on prototypes
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
        return np.round(lr,8)  # Return the learning rate as is for initial epochs
    else:
        if(epoch % 2 == 0):
            return lr/2.  # Reduce learning rate every two epochs
        else:
            return lr

# Define callbacks for learning rate scheduling and early stopping
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)    
    
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', 
    mode='max',
    patience=settings['patience'], 
    restore_best_weights=True, 
    verbose=1
)

callbacks_list = [
#     lr_callback,  
    es_callback,
]            

if(EARLY_STOPPING == False):
    callbacks_list.remove(es_callback)  # Remove early stopping if not needed

# Define metrics for model evaluation
metrics_list = [
    tf.keras.metrics.SparseCategoricalAccuracy(),
]

# ## Instantiate the model

__ = imp.reload(network)  # Reload the network module
tf.keras.backend.clear_session()  # Clear any previous session

# Build the model with specified parameters
model = network.build_model(
    nlayers              = NLAYERS,
    nfilters             = NFILTERS,
    input_shape          = X_train.shape[1:],
    output_shape         = NCLASSES,
    prototypes_per_class = PROTOTYPES_PER_CLASS,
    network_seed         = RANDOM_SEED,    
    prototype_channels   = settings['prototype_channels'],    
    coeff_cluster        = settings['coeff_cluster'],
    coeff_separation     = settings['coeff_separation'],
    coeff_l1             = settings['coeff_l1'],
    incorrect_strength   = settings['incorrect_strength'],
    double_conv          = settings['double_conv'],
    kernel_l1_coeff      = 0.0,#settings['kernel_l1_coeff'],
    kernel_l2_coeff      = 0.0,#settings['kernel_l2_coeff'],
    drop_rate            = 0.0,
    drop_rate_final      = 0.0,        
    
)
model.summary()  # Display model summary

# ## Load pre-trained weights into convolutional layers

if(settings['pretrain'] == True):

    # Determine the path for the pretrained model
    if(settings['pretrain_exp'] is None):
        PRETRAINED_MODEL = model_dir + 'pretrained_model_' + EXP_NAME

        if(EXP_NAME[:9] == 'GCM_SGold'):
            PRETRAINED_MODEL = './saved_models/GCM_alas_wint_583yrs_gold_redo/' + 'pretrained_model_' + 'GCM_alas_wint_583yrs_gold_redo'
    else:
        PRETRAINED_MODEL = './saved_models/' + settings['pretrain_exp']


    print('loading pretrained convolutional layers from ' + PRETRAINED_MODEL)  # Log loading of pretrained model
    pretrained_model = tf.keras.models.load_model(PRETRAINED_MODEL)  # Load the pretrained model

    # Set weights for convolutional layers from the pretrained model
    for layer in range(1,len(model.layers)):
        if(model.layers[layer].name[:4]=='conv'):
            print('   loading pretrained weights for --> ' + model.layers[layer].name)
            model.layers[layer].set_weights(pretrained_model.layers[layer].get_weights())
else:
    print('no pretrained model specified. keeping random initialized weights.')  # Log if no pretrained model is specified
    


# ***

# # Run Training Stages

imp.reload(network)  # Reload the network module
imp.reload(plots)  # Reload the plots module
imp.reload(push_prototypes)  # Reload the push_prototypes module
imp.reload(experiment_settings)  # Reload the experiment_settings module
settings = experiment_settings.get_settings(EXP_NAME)  # Get experiment settings

# Log the shapes of training data and prototypes
ic(np.shape(X_train))
ic(np.shape(prototypes_of_correct_class_train))
ic(np.shape(prototypes_of_correct_class_train))

imp.reload(push_prototypes)  # Reload the push_prototypes module
NEPOCHS    = settings['nepochs']  # Get number of epochs from settings
STAGE_LIST = (0,1,2,3,4,5,6,7,8,9)  # Define the list of training stages

for stage in STAGE_LIST:
    
    print('--------------------')
    print('TRAINING STAGE = ' + str(stage))  # Log the current training stage
    print('--------------------')

    # Load previously trained stage, unless it is the 0th stage
    if(stage != 0):
        tf.keras.backend.clear_session()  # Clear session for new training stage
        model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(stage-1)+ ".h5"
        # model_filename = model_dir + str(learning_rate) + "_" + 'model_' + EXP_NAME + '_stage' + str(stage-1)+ ".h5"
#         model = common_functions.load_model(model_filename)  # Load model if needed
        model.load_weights(model_filename)  # Load weights for the current stage
        
    # Learn layers (during even numbered stages)
    if(stage % 2 == 0):
        # Train prototypes layers (and possibly CNN layers)
        if(settings['pretrain']==False and settings['train_cnn_in_stage'] == True):
            model = network.set_trainable_layers(model, [True,True,True,False])            
        elif(settings['train_cnn_in_stage'] == False or stage==0):
            model = network.set_trainable_layers(model, [False,True,True,False])
        elif(settings['train_cnn_in_stage'] == True):
            model = network.set_trainable_layers(model, [True,True,True,False])            
        elif(stage >= settings['train_cnn_in_stage']):
            model = network.set_trainable_layers(model, [True,True,True,False])            
        else:
            model = network.set_trainable_layers(model, [False,True,True,False])
    else:
        #.......................................................
        # push the prototypes
        #.......................................................        
        model, push_info = push_prototypes.push(model, 
                                                [X_train,prototypes_of_correct_class_train], 
                                                prototypes_of_correct_class_train, 
                                                perform_push=True,
                                                batch_size=BATCH_SIZE_PREDICT,
                                                verbose=False,
                                                )
        print('Push complete.\n')            

        # Set the model to train only the last layer
        model = network.set_trainable_layers(model, [False,False,False,True])

        if(stage == 9):
            print("Writing Final Protos")
            # Save final prototypes and similarity scores
            np.savetxt(exp_data_dir + EXP_NAME + 'final_push_protos.txt', push_info[0], fmt='%d')
            np.savetxt(exp_data_dir + EXP_NAME + 'final_protos_loc.txt', push_info[-1], fmt='%d')
            np.save(exp_data_dir + EXP_NAME + 'similarity_scores.npy', push_info[-2])
            

    #.......................................................
    # Compile the model
    #.......................................................
    if(stage>=settings['cut_lr_stage']):
        lr_factor = 10.**(np.floor((stage-settings['cut_lr_stage']+2)/2))
    else:
        lr_factor = 1.
    if(LR_INIT/lr_factor<settings['min_lr']):
        lr_factor = LR_INIT/settings['min_lr']
    print('learning rate = ' + str(np.asarray(LR_INIT/lr_factor,dtype='float32')))

    # Compile the model with Adam optimizer and loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=np.asarray(LR_INIT/lr_factor,dtype='float32'), 
        ),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = metrics_list,
    )
    # Display minimum and maximum weights of the layer
    ic(np.min(model.layers[-3].get_weights()[1]),np.max(model.layers[-3].get_weights()[1]))

    #.......................................................
    # Train the model
    #.......................................................
    print('Training the model...')    
    
    tf.random.set_seed(RANDOM_SEED)   
    np.random.seed(RANDOM_SEED)    
    # Fit the model on training data
    history = model.fit(
        [X_train,prototypes_of_correct_class_train],
        y_train,
        validation_data=([[X_val,prototypes_of_correct_class_val]], [y_val]),
        batch_size=BATCH_SIZE,
        epochs=NEPOCHS[stage],
        shuffle=True,
        verbose=1,
        callbacks=callbacks_list
    )
    print('Training complete.\n')            
        
    # Save the model at this training stage
    model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(stage)
    common_functions.save_model(model, model_filename) 
    
    #.......................................................
    # Plot results
    #.......................................................  
    try:
        # Plot loss history of the model
        plots.plot_loss_history(history)
        plt.savefig(model_diagnostics_dir + EXP_NAME + '_loss_history_stage' + str(stage) + '.png', dpi=dpiFig)    
        plt.close()

        # Plot the weights of the model
        plots.plot_weights(model, PROTOTYPES_PER_CLASS)    
        plt.savefig(model_diagnostics_dir + EXP_NAME + '_weights_stage' + str(stage) + '.png', dpi=dpiFig)
        plt.close()
    except:
        print('not making plots...')
        plt.close()

# Function to get top confidence prototypes
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
    
    # Get top scores based on percentage
    topscore = np.asarray(hq.nlargest(int(total_maxvals.shape[0]*percentage), total_maxvals))
    toplocs = np.where(np.in1d(total_maxvals,topscore))[0]

    argstest = np.argsort(total_maxvals[toplocs])
    bestsamps = all_isamples[toplocs[argstest][::-1]]
    return bestsamps

# Function to generate confusion matrix
def make_confuse_matrix(y_predict, y_test, data_amount, base):
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
            ax.text(x=j, y=i+.3,s=(str(np.around(cf_matrix_pred[i, j]*100,4))+'\%'), va='center', ha='center', size='xx-large', color = 'green')
            ax.text(x=j, y=i-.3,s=(str(np.around(cf_matrix_true[i, j]*100,4))+'\%'), va='center', ha='center', size='xx-large', color = 'red')

            if (i == j):
                correct_preds += cf_matrix[i, j]

    correct_preds /= np.sum(cf_matrix)
    
    return np.around(correct_preds*100,2)

input_val  = [[X_val,prototypes_of_correct_class_val]]

# Predict on validation data
y_predict_val = model.predict(input_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

if(settings['pretrain'] == True):
    base_model_filename = model_dir + 'pretrained_model_' + EXP_NAME
    base_model = common_functions.load_model(base_model_filename)
    base_y_predict_val = base_model.predict(X_val, batch_size=BATCH_SIZE_PREDICT, verbose=1)

accuracies = []
base_accuracies = []

# Calculate accuracies for different percentages of confident samples
for i in np.arange(10, 101, 5):
    accuracies.append(make_confuse_matrix(y_predict_val[top_confidence_protos(i/100., y_predict_val)], y_val[top_confidence_protos(i/100., y_predict_val)], i, True))
    if(settings['pretrain'] == True):
        base_accuracies.append(make_confuse_matrix(base_y_predict_val[top_confidence_protos(i/100., base_y_predict_val)], y_val[top_confidence_protos(i/100., base_y_predict_val)], i, True))

# Plot accuracy results
plt.figure(figsize=(10,6))
plt.plot(np.arange(10, 101, 5)[::-1], accuracies, label = "TLLTT - val")

if(settings['pretrain'] == True):
    plt.plot(np.arange(10, 101, 5)[::-1], base_accuracies, label = "Base CNN - val")

plt.title("Discard plot for Val Only", fontsize=20)
plt.xlabel("Percentage of confident samples used", fontsize=15)
plt.xticks(ticks=np.arange(10, 101, 5), labels=np.arange(10, 101, 5)[::-1])
plt.ylabel("Accuracy", fontsize=15)
plt.axhspan(0, 33, color='y', alpha=0.5, lw=0)

# Save accuracy results to file
np.savetxt(exp_data_dir + EXP_NAME + '_TLLTT_Val_accuracy', accuracies, fmt='%1.5f')

if(settings['pretrain'] == True):
    np.savetxt(exp_data_dir + EXP_NAME + '_Base_Val_accuracy', base_accuracies, fmt='%1.5f')

# Adjust y-axis limits based on accuracy
if((np.min(accuracies) >= 31)):
    plt.ylim(bottom=30)
else:
    plt.ylim(bottom=20)
plt.legend()
plt.savefig((vizualization_dir + EXP_NAME + 'VALONLY_forecast_of_opportunity.png'), bbox_inches='tight', dpi=dpiFig)
plt.close()
