

# # This Looks Like That There
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
import experiment_settings 
import data_functions_schooner
import push_prototypes
import plots
import common_functions


7
__author__ = "Elizabeth A. Barnes and Randal J Barnes"
__version__ = "1 December 2021"

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
dpiFig = 300.

# ## Print the detailed system info

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"tensorflow version = {tf.__version__}")

# ## Define experiment settings and directories

EXP_NAME = 'alas_200year_winter_ternary_GCM_Falco'#balanced_test'#initial_test'#'mjo'#'quadrants_testcase'

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
      
# elif((EXP_NAME[:12]=='initial_test') or (EXP_NAME[:12]=='smaller_test') or (EXP_NAME[:13]=='balanced_test') or (EXP_NAME[:13]=='threeday_test') or (EXP_NAME[:12]=='zeroday_test') or (EXP_NAME[:16]=='fourteenday_test') or (EXP_NAME[:18]=='fourteenday_precip')
#      or (EXP_NAME[:19]=='seventeenday_precip') or (EXP_NAME[:16]=='elevenday_precip') or (EXP_NAME[:30]=='fixed_fourteenday_precip') or (EXP_NAME[:30]=='cold_fourteenday_precip') or (EXP_NAME[:30]=='mjo_fourteenday_precip') or (EXP_NAME[:30]=='shuffle_fourteenday_precip')
#      or (EXP_NAME[:30]=='cali_fourteenday_precip') or (EXP_NAME[:30]=='alas_fourteenday_precip') or (EXP_NAME[:30]=='alas_fourteenday_5proto') or (EXP_NAME[:30]=='alas_fourteenday_back') or (EXP_NAME[:30]=='alas_fourteenday_large') or (EXP_NAME[:30]=='vanc_fourteenday_precip')
#      or (EXP_NAME[:30]=='alas_fourteenday_precip_pre') or (EXP_NAME[:30]=='alas_14day_precip_schooner') or (EXP_NAME[:30]=='LA_14day_precip_schooner') or (EXP_NAME[:30]=='cres_14day_precip_schooner') or (EXP_NAME[:30]=='vanc_14day_precip_schooner')
#      or (EXP_NAME[:30]=='alas_14dayback_precip_schooner') or (EXP_NAME[:50]=='alas_14day_precip_large_schooner') or (EXP_NAME[:70]=='alas_14day_precip_5mean_large_schooner') or (EXP_NAME[:70]=='alas_14day_precip_5mean_schooner') 
#      or (EXP_NAME[:70]=='alas_14day_precip_5back_schooner') or (EXP_NAME[:70]=='alas_14day_precip_6back_schooner')):
labels, data, lat, lon, time = data_functions_schooner.load_tropic_data_winter(DATA_DIR)
X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test = data_functions_schooner.get_and_process_tropic_data_winter(labels,
                                                                                        data,
                                                                                        time,
                                                                                        rng, 
                                                                                        colored=settings['colored'],
                                                                                        standardize=settings['standardize'],
                                                                                        shuffle=settings['shuffle'],
                                                                                    )

                                                                                
# labels, data, lat, lon, time = data_functions_schooner.load_tropic_data_winter_ERA5(DATA_DIR)
# X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test = data_functions_schooner.get_and_process_tropic_data_winter_ERA5(labels,
#                                                                                         data,
#                                                                                         time,
#                                                                                         rng, 
#                                                                                         colored=settings['colored'],
#                                                                                         standardize=settings['standardize'],
#                                                                                         shuffle=settings['shuffle'],
#                                                                                         r_seed = RANDOM_SEED,
#                                                                                     )

# elif((EXP_NAME[:21]=='fourteenday_both_test') or ((EXP_NAME[:18]=='threeday_both_test'))):
#     print("bingo")
#     labels, data, lat, lon, time = data_functions_schooner.load_z500_precip_data(DATA_DIR)
#     X_train, y_train, time_train, X_val, y_val, time_val, X_test, y_test, time_test = data_functions_schooner.get_and_process_tropic_data(labels,
#                                                                                          data,
#                                                                                          time,
#                                                                                          rng, 
#                                                                                          colored=settings['colored'],
#                                                                                          standardize=settings['standardize'],
#                                                                                          shuffle=settings['shuffle'],
#                                                                                         )


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
    

# ## Define the training callbacks and metrics

# callbacks
def scheduler(epoch, lr):
    if epoch < LR_CALLBACK_EPOCH:
        return np.round(lr,8)
    else:
        if(epoch % 2 == 0):
            return lr/2.
        else:
            return lr

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
#     es_callback,
]            

# metrics
metrics_list = [
    tf.keras.metrics.SparseCategoricalAccuracy(),
]

# ## Instantiate the model

__ = imp.reload(network)
tf.keras.backend.clear_session()

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
model.summary()

# ## Load pre-trained weights into convolutional layers

if(settings['pretrain'] == True):

    if(settings['pretrain_exp'] is None):
        PRETRAINED_MODEL = model_dir + 'pretrained_model_' + EXP_NAME
    else:
        PRETRAINED_MODEL = './saved_models/' + settings['pretrain_exp'] 

    print('loading pretrained convolutional layers from ' + PRETRAINED_MODEL)
    pretrained_model = tf.keras.models.load_model(PRETRAINED_MODEL)

    for layer in range(1,len(model.layers)):
        if(model.layers[layer].name[:4]=='conv'):
            print('   loading pretrained weights for --> ' + model.layers[layer].name)
            model.layers[layer].set_weights(pretrained_model.layers[layer].get_weights())
else:
    print('no pretrained model specified. keeping random initialized weights.')
    

# raise ValueError('here')

# ***

# # Run Training Stages

imp.reload(network)
imp.reload(plots)
imp.reload(push_prototypes)
imp.reload(experiment_settings)
settings = experiment_settings.get_settings(EXP_NAME)

ic(np.shape(X_train))
ic(np.shape(prototypes_of_correct_class_train))
ic(np.shape(prototypes_of_correct_class_train))

imp.reload(push_prototypes)
NEPOCHS    = settings['nepochs']
STAGE_LIST = (0,1,2,3,4,5,6,7,8,9)#(0,1,2,3,4,5,6,7,8,9)#range(len(NEPOCHS))#(1,2,3,4,5)#range(len(NEPOCHS))

for stage in STAGE_LIST:
    
    print('--------------------')
    print('TRAINING STAGE = ' + str(stage))
    print('--------------------')

    # load previously trained stage, unless it is the 0th stage
    if(stage != 0):
        tf.keras.backend.clear_session()
        model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(stage-1)+ ".h5"
#         model = common_functions.load_model(model_filename)
        model.load_weights(model_filename)
        
    # learn layers (during even numbered stages)
    if(stage % 2 == 0):
        # train prototypes layers (and possibly CNN layers)
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

        # train weights layer only
        model = network.set_trainable_layers(model, [False,False,False,True])
        # print("Huge Print------------------------------------------------------------------------------------------")
        print(push_info[0])
        print(push_info[0].shape)
        if(stage == 9):
            print("Writing Final Protos")
            print(push_info[-2])
            print(push_info[-2].shape)
            np.savetxt(exp_data_dir + EXP_NAME + 'final_push_protos.txt', push_info[0], fmt='%d')
            np.savetxt(exp_data_dir + EXP_NAME + 'final_protos_loc.txt', push_info[-1], fmt='%d')
            np.save(exp_data_dir + EXP_NAME + 'similarity_scores.npy', push_info[-2])
            
        # print("Huge Print------------------------------------------------------------------------------------------")

    #.......................................................
    # compile the model
    #.......................................................
    if(stage>=settings['cut_lr_stage']):
        lr_factor = 10.**(np.floor((stage-settings['cut_lr_stage']+2)/2))
    else:
        lr_factor = 1.
    if(LR_INIT/lr_factor<settings['min_lr']):
        lr_factor = LR_INIT/settings['min_lr']
    print('learning rate = ' + str(np.asarray(LR_INIT/lr_factor,dtype='float32')))

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=np.asarray(LR_INIT/lr_factor,dtype='float32'), 
        ),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = metrics_list,
    )
#     model.summary()
    ic(np.min(model.layers[-3].get_weights()[1]),np.max(model.layers[-3].get_weights()[1]))

    #.......................................................
    # train the model
    #.......................................................
    print('Training the model...')    
    
    tf.random.set_seed(RANDOM_SEED)   
    np.random.seed(RANDOM_SEED)    
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
        

    # save the model at this training stage
    model_filename = model_dir + 'model_' + EXP_NAME + '_stage' + str(stage)
    common_functions.save_model(model, model_filename) 
    
    #.......................................................
    # plot results
    #.......................................................  
    try:
        # plot loss history of the model
        plots.plot_loss_history(history)
        plt.savefig(model_diagnostics_dir + EXP_NAME + '_loss_history_stage' + str(stage) + '.png', dpi=dpiFig)    
        plt.close()

        # plot the weights
        plots.plot_weights(model, PROTOTYPES_PER_CLASS)    
        plt.savefig(model_diagnostics_dir + EXP_NAME + '_weights_stage' + str(stage) + '.png', dpi=dpiFig)
        plt.close()
    except:
        print('not making plots...')
        plt.close()



