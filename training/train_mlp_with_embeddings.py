'''
Author: Juan Guerrero Martin.
Creation date: 3 january 2024.
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
import tensorflow as tf
import time

from tensorflow import keras
print( '\n' )

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils
import utils.model_creator as model_creator

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'train_mlp_with_embeddings' )

parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented.pickle", help = "Pickle with the ROCF dataset." )
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_augmented_fold_info_tvt_v01/", help = "Directory with fold information of ROCF dataset." )

parser.add_argument( "--embeddings_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/embeddings/", help = "Directory where to save the embeddings." )
parser.add_argument( "--embed_problem_type", type = str, default = "recons", help = "INPUT problem type." )
parser.add_argument( "--embed_model_type", type = str, default = "CAE", help = "Choosing the INPUT model." )
parser.add_argument( "--embed_dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the INPUT dataset." )
parser.add_argument( "--embed_execution_iteration", type = int, default = 0, help = "INPUT execution iteration." )

parser.add_argument( "--output_model_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/models/", help = "Directory where to save the partial models." )
parser.add_argument( "--execution_times_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/execution_times/", help = "Directory where to save the execution times." )
parser.add_argument( "--histories_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/histories/", help = "Directory where to save the histories." )

parser.add_argument( "--problem_type", type = str, default = "mci_det", help = "Choosing the problem type." )
parser.add_argument( "--model_type", type = str, default = "MLP", help = "Choosing the model to be used for training." )
parser.add_argument( "--dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the dataset to be used for training." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )

parser.add_argument( "--custom_batch_size", type = int, default = 32, help = "Batch size." )
parser.add_argument( "--custom_learning_rate", type = float, default = 1e-3, help = "Learning rate." )
parser.add_argument( "--custom_training_epochs", type = int, default = 100, help = "Training epochs." )

parser.add_argument( "--custom_gpu_id", type = str, default = '0', help = "Choosing which GPU to use by its index." )
parser.add_argument( "--custom_random_seed", type = int, default = 333, help = "Random seed." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path
fold_info_dir = args.fold_info_dir

embeddings_dir = args.embeddings_dir
embed_problem_type = args.embed_problem_type
embed_model_type = args.embed_model_type
embed_dataset_type = args.embed_dataset_type
embed_execution_iteration = args.embed_execution_iteration

output_model_dir = args.output_model_dir
execution_times_dir = args.execution_times_dir
histories_dir = args.histories_dir

problem_type = args.problem_type
model_type = args.model_type
dataset_type = args.dataset_type
execution_iteration = args.execution_iteration

custom_batch_size = args.custom_batch_size
custom_learning_rate = args.custom_learning_rate
custom_training_epochs = args.custom_training_epochs

custom_gpu_id = args.custom_gpu_id
custom_random_seed = args.custom_random_seed

# Setting the random seed.
np.random.seed( custom_random_seed )
tf.random.set_seed( custom_random_seed )

# CUDA configuration.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = custom_gpu_id
# Dynamic GPU memory allocation (enable this to not occupy all GPU memory).
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

NUMBER_OF_FOLDS = 14
CSV_DELIMITER = ';'

CUSTOM_EMBEDDING_LENGTH = 9216
CUSTOM_NUMBER_OF_CLASSES = 1
CUSTOM_OUTPUT_ACTIVATION_FUNCTION = 'sigmoid'

'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  embed_model_base_name = embed_problem_type + '_' + embed_model_type + '_with_' + embed_dataset_type + '_i' + str( embed_execution_iteration )
  output_model_base_name = problem_type + '_' + model_type + '_with_' + dataset_type + '_i' + str( execution_iteration )
  
  # Creating experiment folders.
  
  experiment_output_model_dir = output_model_dir + output_model_base_name + '/'
  if not( os.path.isdir( experiment_output_model_dir ) ):
    os.mkdir( experiment_output_model_dir )
    
  experiment_execution_times_dir = execution_times_dir + output_model_base_name + '/'
  if not( os.path.isdir( experiment_execution_times_dir ) ):
    os.mkdir( experiment_execution_times_dir )
    
  experiment_histories_dir = histories_dir + output_model_base_name + '/'
  if not( os.path.isdir( experiment_histories_dir ) ):
    os.mkdir( experiment_histories_dir )

  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path )
    
  # Training.
  
  execution_time_list = np.empty( shape = ( NUMBER_OF_FOLDS ), dtype = object )

  print( 'Training.' )
  print( 'Experiment alias:', output_model_base_name )
  print( '\n' )
  
  for fold_index in range( 0, NUMBER_OF_FOLDS ):
    
    fold_counter = fold_index + 1
    
    current_fold_as_string = '{:02d}'.format( fold_counter )
    
    print( 'Fold ' + current_fold_as_string + '.' )
    print( '\n' )
    
    # Getting the indices.
    
    train_indices_as_csv = fold_info_dir + 'train_fold' + current_fold_as_string + '.csv'
    train_indices = np.genfromtxt( train_indices_as_csv, delimiter = ';', dtype = int )
    
    val_indices_as_csv = fold_info_dir + 'val_fold' + current_fold_as_string + '.csv'
    val_indices = np.genfromtxt( val_indices_as_csv, delimiter = ';', dtype = int )
    
    print( 'Validation set indices:', val_indices )
    print( '\n' )
    
    # Getting the embeddings.
    
    train_embeddings_dir = embeddings_dir + 'emb_' + dataset_type + '_' + 'train' + '_from_' + embed_model_base_name + '/'
    train_embeddings_path = train_embeddings_dir + 'fold' + current_fold_as_string + '.npy'
    
    print( 'Train set embeddings path:', train_embeddings_path )
    print( '\n' )
    
    train_embeddings = np.load( train_embeddings_path )
  
    print( 'train_embeddings' )
    print( train_embeddings.shape )
    print( train_embeddings )
    print( '\n' )
    
    val_embeddings_dir = embeddings_dir + 'emb_' + dataset_type + '_' + 'val' + '_from_' + embed_model_base_name + '/'
    val_embeddings_path = val_embeddings_dir + 'fold' + current_fold_as_string + '.npy'
    
    print( 'Validation set embeddings path:', val_embeddings_path )
    print( '\n' )
    
    val_embeddings = np.load( val_embeddings_path )
  
    print( 'val_embeddings' )
    print( val_embeddings.shape )
    print( val_embeddings )
    print( '\n' )
    
    # Model creation.
    
    model = model_creator.simpleNeuralNetwork(
      embedding_length = CUSTOM_EMBEDDING_LENGTH, 
      num_of_classes = CUSTOM_NUMBER_OF_CLASSES, 
      activation_function = CUSTOM_OUTPUT_ACTIVATION_FUNCTION )

    # Model summary.
    #model.summary()
    #sys.exit()
    
    # Model compile.

    '''
    Adam optimizer.
    beta_1: related to gradient.
    beta_2: related to squared gradient.
    '''
    custom_adam_optimizer = keras.optimizers.Adam( 
      learning_rate = custom_learning_rate, 
      beta_1 = 0.9, beta_2 = 0.999, amsgrad = False )

    model.compile(
      optimizer = custom_adam_optimizer,
      loss = 'binary_crossentropy',
      metrics = [ 'accuracy' ]
    )
        
    # Model checkpointing.
    
    experiment_output_model_dir_fold = experiment_output_model_dir + 'fold' + current_fold_as_string + '/'
    
    if not( os.path.isdir( experiment_output_model_dir_fold ) ):
      os.mkdir( experiment_output_model_dir_fold )

    experiment_output_model_dir_fold_checkpoints = experiment_output_model_dir_fold + '{epoch:04d}.hdf5'
    
    custom_model_checkpoint = keras.callbacks.ModelCheckpoint(
      filepath = experiment_output_model_dir_fold_checkpoints,
      monitor = 'val_loss',
      verbose = 0,
      save_best_only = True,
      save_weights_only = False,
      mode = 'min',
      save_freq = 'epoch'
    )
    
    # Model fit.
    
    # Timing BEGIN.
    start_time = time.time()

    history = model.fit(
      x = train_embeddings,
      y = y[train_indices],
      batch_size = custom_batch_size,
      epochs = custom_training_epochs,
      callbacks = [ custom_model_checkpoint ],
      validation_data = ( val_embeddings, y[val_indices] )
    )
    
    # Timing END.
    elapsed_time = time.time() - start_time
    
    # Save history.

    histories_path = experiment_histories_dir + 'fold' + current_fold_as_string + '.npy'
    np.save( histories_path, history.history )
    
    # Format execution time.

    execution_time_as_string = ml_utils.convertToDDHHMMSSFormatAndPrint( elapsed_time )
    execution_time_list[ fold_index ] = execution_time_as_string

  # Saving execution times.

  execution_times_as_csv = experiment_execution_times_dir + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  np.savetxt( execution_times_as_csv, execution_time_list, fmt = '%s', delimiter = ',' )
  print( 'Execution times saved to:', execution_times_as_csv )