'''
Author: Juan Guerrero Martin.
Creation date: 22 december 2023.
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

parser = argparse.ArgumentParser( description = 'extract_embeddings' )

parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented.pickle", help = "Pickle with the ROCF dataset." )

# WARNING: make sure to choose the correct dataset.
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_augmented_fold_info_tvt_v01/", help = "Directory with fold information of ROCF dataset." )

parser.add_argument( "--trained_model_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/models/", help = "Directory where to save the partial models." )

parser.add_argument( "--stopping_epochs_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/stopping_epochs/", help = "Directory where to save the stopping epochs." )

parser.add_argument( "--input_problem_type", type = str, default = "recons", help = "INPUT problem type." )
parser.add_argument( "--input_model_type", type = str, default = "CAE", help = "INPUT model." )
parser.add_argument( "--input_dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "INPUT dataset." )
parser.add_argument( "--input_execution_iteration", type = int, default = 0, help = "INPUT execution iteration." )

parser.add_argument( "--input_fold_index", type = int, default = 0, help = "Choosing the fold from which we will get the embeddings." )

parser.add_argument( "--output_dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the dataset from which we will get the embeddings." )
parser.add_argument( "--output_subset_name", type = str, default = "train", help = "Choosing the subset from which we will get the embeddings." )

parser.add_argument( "--output_embeddings_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/embeddings/", help = "Directory where to save the embeddings." )

parser.add_argument( "--custom_gpu_id", type = str, default = '0', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path
fold_info_dir = args.fold_info_dir

trained_model_dir = args.trained_model_dir
stopping_epochs_dir = args.stopping_epochs_dir

input_problem_type = args.input_problem_type
input_model_type = args.input_model_type
input_dataset_type = args.input_dataset_type
input_execution_iteration = args.input_execution_iteration
input_fold_index = args.input_fold_index

output_dataset_type = args.output_dataset_type
output_subset_name = args.output_subset_name

output_embeddings_dir = args.output_embeddings_dir

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
DO_SAVE_EMBEDDINGS = True

'''
Part 4. Main body.
'''

if __name__ == '__main__':
    
  # Defining model and dataset names.
  
  input_model_base_name = input_problem_type + '_' + input_model_type + '_with_' + input_dataset_type + '_i' + str( input_execution_iteration )
  output_dataset_base_name = output_dataset_type + '_' + output_subset_name
  
  # Creating experiment folders.
  
  experiment_trained_model_dir = trained_model_dir + input_model_base_name + '/'
  
  experiment_stopping_epochs_dir = stopping_epochs_dir + input_model_base_name + '/'
  
  experiment_output_embeddings_dir = output_embeddings_dir + 'emb_' + output_dataset_base_name + '_from_' + input_model_base_name + '/'
  if not( os.path.isdir( experiment_output_embeddings_dir ) ):
    os.mkdir( experiment_output_embeddings_dir )
    
  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path )
  
  # Creating the generator.
  
  X = X * RESCALE_FACTOR
  eval_image_data_generator = keras.preprocessing.image.ImageDataGenerator( )
  
  # Getting embedding information.
  
  fold_counter = input_fold_index + 1
  current_fold_as_string = '{:02d}'.format( fold_counter )
  
  print( 'Generating embeddings.' )
  print( 'Trained model:', input_model_base_name )
  print( 'Fold:', current_fold_as_string )
  print( 'Dataset:', output_dataset_type )
  print( 'Subset:', output_subset_name )
  print( '\n' )
    
  # Loading stopping epochs.

  # Stopping epochs file stores the selected model for each fold (represented by its training epoch).
  # Example: [2,4,4,13,5,6,11,6,1,11,4,4,6,7]

  stopping_epochs_path = experiment_stopping_epochs_dir + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  input_stopping_epoch_list = np.genfromtxt( stopping_epochs_path, delimiter = STOPPING_EPOCHS_DELIMITER, dtype = int )
  current_stopping_epoch_as_string = '{:04d}'.format( input_stopping_epoch_list[ input_fold_index ] )
  
  # Loading the model.
  
  pretrained_model_path = experiment_trained_model_dir + 'fold' + current_fold_as_string +  '/' + current_stopping_epoch_as_string + '.hdf5'
  
  model = keras.models.load_model( pretrained_model_path )
  
  print( 'Loaded model:', pretrained_model_path )
  print( '\n' )
  
  # Feedback.
  #model.summary( )
  #sys.exit( )
  
  # Building a partial model.
  
  last_layer_name = 'max_pooling2d_3'
  new_output = keras.layers.Flatten( name = 'flatten' )( model.get_layer( last_layer_name ).output )
  
  partial_model = keras.Model( inputs = model.input, outputs = new_output )
  
  # Feedback.
  #partial_model.summary( )
  #sys.exit( )
  
  # Getting subset indices.
  
  eval_indices_as_csv = None
  if output_subset_name == 'test':
    eval_indices_as_csv = fold_info_dir + output_subset_name + '.csv'
  else:
    eval_indices_as_csv = fold_info_dir + output_subset_name + '_fold' + current_fold_as_string + '.csv'
  
  eval_indices = np.genfromtxt( eval_indices_as_csv, delimiter = INDICES_DELIMITER, dtype = int )
  
  print( 'Indices path:', eval_indices_as_csv )
  print( 'Indices values:', eval_indices )
  print( '\n' )
  
  # Image data iterator.
  
  eval_data_iterator = eval_image_data_generator.flow(
    x = X[eval_indices],
    y = X[eval_indices],
    batch_size = 1,
    shuffle = False
  )
  
  step_size_eval = int( np.ceil( eval_data_iterator.n / eval_data_iterator.batch_size ) )
  
  # Model predict.

  embedding_matrix = partial_model.predict(
    x = eval_data_iterator,
    steps = step_size_eval,
    verbose = 1,
  )
  
  print( 'Embedding matrix.' )
  print( embedding_matrix.shape )
  print( embedding_matrix )
  print( '\n' )
  
  # Saving the embeddings to a NPY file.
  
  if DO_SAVE_EMBEDDINGS:
  
    embeddings_path = experiment_output_embeddings_dir + 'fold' + current_fold_as_string + '.npy'
    np.save( embeddings_path, embedding_matrix )
    
    print( embeddings_path, 'generated.' )
    print( '\n' )