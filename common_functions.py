# This Looks Like That There - S2S Release Version

import os
import numpy as np
import tensorflow as tf

__author__ = "Nicolas Gordillo"
__date__ = "April 2024"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def save_model(model, model_filename):

    print('saving model and weights to ' + model_filename)
    
    # save the weights only
    model.save_weights(model_filename + '.h5')

    # save the entire model in TF2.5
    tf.keras.models.save_model(model, model_filename, overwrite=True)       

def load_model(model_filename):

    print('loading model from ' + model_filename)

    # Restore the weights
    # model.load_weights('saved_models/' + model_name + '.h5')
    
    # Load full model
    model = tf.keras.models.load_model(model_filename)
    return model


def get_exp_directories(exp_name):
    # make model and figure directories if they do not exist
    
    model_diagnostics_dir = './figures/' + exp_name + '/model_diagnostics/' 
    if not os.path.exists(model_diagnostics_dir):
        os.makedirs(model_diagnostics_dir)   
    
    vizualization_dir = './figures/' + exp_name + '/vizualization/' 
    if not os.path.exists(vizualization_dir):
        os.makedirs(vizualization_dir)
    
    model_dir = './saved_models/' + exp_name + '/' 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    exp_data_dir = './data/' + exp_name + '/' 
    if not os.path.exists(exp_data_dir):
        os.makedirs(exp_data_dir)    

    #Make extra figure directories. 
    #########################################################################
    conf_dir = './figures/' + exp_name + '/vizualization/confusion_matrices/'
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    indiv_dir = './figures/' + exp_name + '/vizualization/individual_protos/'
    if not os.path.exists(indiv_dir):
        os.makedirs(indiv_dir)
    return model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir

def get_exp_directories_schooner(exp_name):
    # make model and figure directories if they do not exist
    
    model_diagnostics_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/model_diagnostics/' 
    if not os.path.exists(model_diagnostics_dir):
        os.makedirs(model_diagnostics_dir)   
    
    vizualization_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/' 
    if not os.path.exists(vizualization_dir):
        os.makedirs(vizualization_dir)
    
    model_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/saved_models/' + exp_name + '/' 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    exp_data_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/data/' + exp_name + '/' 
    if not os.path.exists(exp_data_dir):
        os.makedirs(exp_data_dir)

    #Make extra figure directories. 
    #########################################################################
    conf_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/confusion_matrices/'
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    indiv_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/individual_protos/'
    if not os.path.exists(indiv_dir):
        os.makedirs(indiv_dir)

    mjo_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/mjo_figs/'
    if not os.path.exists(mjo_dir):
        os.makedirs(mjo_dir)
    
    precip_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/precip_figs/'
    if not os.path.exists(precip_dir):
        os.makedirs(precip_dir)

    timeseries_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/timeseries/'
    if not os.path.exists(timeseries_dir):
        os.makedirs(timeseries_dir)

    examples_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/example_protos/'
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    enso_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/enso_figs/'
    if not os.path.exists(enso_dir):
        os.makedirs(enso_dir)

    ranks_dir = '/ourdisk/hpc/ai2es/nicojg/TLLTT/figures/' + exp_name + '/vizualization/rank_figs/'
    if not os.path.exists(ranks_dir):
        os.makedirs(ranks_dir)

        
        
    return model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir

def get_exp_directories_falco(exp_name):
    # make model and figure directories if they do not exist
    
    model_diagnostics_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/model_diagnostics/' 
    if not os.path.exists(model_diagnostics_dir):
        os.makedirs(model_diagnostics_dir)   
    
    vizualization_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/' 
    if not os.path.exists(vizualization_dir):
        os.makedirs(vizualization_dir)
    
    model_dir = '/home/nicojg/TLLTT/saved_models/' + exp_name + '/' 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    exp_data_dir = '/barnes-engr-scratch1/nicojg/data/' + exp_name + '/' 
    if not os.path.exists(exp_data_dir):
        os.makedirs(exp_data_dir)

    #Make extra figure directories. 
    #########################################################################
    conf_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/confusion_matrices/'
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)

    indiv_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/individual_protos/'
    if not os.path.exists(indiv_dir):
        os.makedirs(indiv_dir)

    mjo_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/mjo_figs/'
    if not os.path.exists(mjo_dir):
        os.makedirs(mjo_dir)
    
    precip_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/precip_figs/'
    if not os.path.exists(precip_dir):
        os.makedirs(precip_dir)

    timeseries_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/timeseries/'
    if not os.path.exists(timeseries_dir):
        os.makedirs(timeseries_dir)

    examples_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/example_protos/'
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    enso_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/enso_figs/'
    if not os.path.exists(enso_dir):
        os.makedirs(enso_dir)

    ranks_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/rank_figs/'
    if not os.path.exists(ranks_dir):
        os.makedirs(ranks_dir)

    era5_dir = '/home/nicojg/TLLTT/figures/' + exp_name + '/vizualization/era5_figs/'
    if not os.path.exists(era5_dir):
        os.makedirs(era5_dir)
        
    return model_dir, model_diagnostics_dir, vizualization_dir, exp_data_dir