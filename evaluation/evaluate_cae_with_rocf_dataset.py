'''
Author: Juan Guerrero Martin.
Creation date: 20 december 2023.
'''

'''
Script structure:
Part 1. Imports.
Part 2. Argument parser.
Part 3. Global constants and variables.
Part 4. Main body.
'''

'''
Part 1. Imports.
'''

import argparse
# numpy recommended version: 1.21.2
import numpy as np
import os
import sys

from tensorflow import keras

# In order to use sibling modules.
sys.path.append( '..' )

import utils.machine_learning_utils as ml_utils

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'evaluate_cae_with_rocf_dataset' )

parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented.pickle", help = "Pickle with the ROCF dataset." )

# WARNING: make sure to choose the correct dataset.
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_augmented_fold_info_tvt_v01/", help = "Directory with fold information of ROCF dataset." )

parser.add_argument( "--trained_model_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/models/", help = "Directory where to save the partial models." )

parser.add_argument( "--stopping_epochs_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/stopping_epochs/", help = "Directory where to save the stopping epochs." )

parser.add_argument( "--problem_type", type = str, default = "recons", help = "Choosing the problem type." )
parser.add_argument( "--model_type", type = str, default = "CAE", help = "Choosing the model." )
parser.add_argument( "--dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the dataset to be used for training." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )

parser.add_argument( "--subset_to_evaluate", type = str, default = "val", help = "Choosing the dataset to be used for predicting." )

parser.add_argument( "--custom_gpu_id", type = str, default = '0', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path
fold_info_dir = args.fold_info_dir

trained_model_dir = args.trained_model_dir
stopping_epochs_dir = args.stopping_epochs_dir

problem_type = args.problem_type
model_type = args.model_type
dataset_type = args.dataset_type
execution_iteration = args.execution_iteration

subset_to_evaluate = args.subset_to_evaluate

custom_gpu_id = args.custom_gpu_id

# CUDA configuration.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = custom_gpu_id
# Dynamic GPU memory allocation (enable this to not occupy all GPU memory).
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

NUMBER_OF_FOLDS = 14
RESCALE_FACTOR = 1.0 / 255.0
STOPPING_EPOCHS_DELIMITER = ','
INDICES_DELIMITER = ';'

'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  # Defining model and stopping epochs directories.
  
  output_model_base_name = problem_type + '_' + model_type + '_with_' + dataset_type + '_i' + str( execution_iteration )
  
  experiment_trained_model_dir = trained_model_dir + output_model_base_name + '/'
  
  print( 'Loading models from: ', experiment_trained_model_dir )
  print( '\n' )
  
  experiment_stopping_epochs_dir = stopping_epochs_dir + output_model_base_name + '/'
  
  print( 'Loading stopping epochs from: ', experiment_stopping_epochs_dir )
  print( '\n' )
      
  # Loading stopping epochs.

  # Stopping epochs file stores the selected model for each fold (represented by its training epoch).
  # Example: [2,4,4,13,5,6,11,6,1,11,4,4,6,7]

  stopping_epochs_path = experiment_stopping_epochs_dir + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  input_stopping_epoch_list = np.genfromtxt( stopping_epochs_path, delimiter = STOPPING_EPOCHS_DELIMITER, dtype = int )
  
  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path )
  
  # Image data generators (for real-time data augmentation AKA RTDA).
    
  # Manually re-scaling.
  X = X * RESCALE_FACTOR
  
  eval_image_data_generator = keras.preprocessing.image.ImageDataGenerator( )
  
  # Evaluating.
  
  print( 'Evaluating.' )
  print( 'Experiment alias:', output_model_base_name )
  print( '\n' )
  
  loss_list = []
  
  for fold_index in range( 0, NUMBER_OF_FOLDS ):
    
    fold_counter = fold_index + 1
    
    current_fold_as_string = '{:02d}'.format( fold_counter )
    
    print( 'Fold ' + current_fold_as_string + '.' )
    print( '\n' )
        
    eval_indices_as_csv = None
    if subset_to_evaluate == 'test':
      eval_indices_as_csv = fold_info_dir + subset_to_evaluate + '.csv'
    else:
      eval_indices_as_csv = fold_info_dir + subset_to_evaluate + '_fold' + current_fold_as_string + '.csv'
    
    eval_indices = np.genfromtxt( eval_indices_as_csv, delimiter = INDICES_DELIMITER, dtype = int )
    
    print( 'Subset to evaluate:', subset_to_evaluate )
    print( 'Path:', eval_indices_as_csv )
    print( 'Indices:', eval_indices )
    print( '\n' )
        
    current_stopping_epoch_as_string = '{:04d}'.format( input_stopping_epoch_list[ fold_index ] )
    
    # Model loading.
    
    pretrained_model_path = experiment_trained_model_dir + 'fold' + current_fold_as_string +  '/' + current_stopping_epoch_as_string + '.hdf5'
    
    model = keras.models.load_model( pretrained_model_path )
    
    print( 'Loaded model:', pretrained_model_path )
    print( '\n' )
    
    # Model summary.
    #model.summary( )

    # Image data iterator.
    
    eval_data_iterator = eval_image_data_generator.flow(
      x = X[eval_indices],
      y = X[eval_indices],
      batch_size = 1,
      shuffle = False
    )
    
    step_size_eval = int( np.ceil( eval_data_iterator.n / eval_data_iterator.batch_size ) )
    
    # Model predict.

    cross_entropy = model.evaluate(
      x = eval_data_iterator,
      steps = step_size_eval,
      verbose = 1,
    )
    
    print( 'Cross-entropy:', cross_entropy )
    print( '\n' )
    
    loss_list.append( cross_entropy )
    
  loss_list_mean = np.mean( loss_list )
  loss_list_std = np.std( loss_list )
  
  print( f'Mean Cross Entropy: {loss_list_mean:.4f}, Standard Deviation Cross Entropy: {loss_list_std:.4f}' )
  print( '\n' )