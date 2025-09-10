'''
Author: Juan Guerrero Martin.
Creation date: 03 january 2023.
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
import pandas as pd
import sys

from tensorflow import keras

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils
import utils.model_creator as model_creator

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'predict_siamese_model_with_rocf_dataset' )

parser.add_argument( "--dataset_path", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented.pickle", help = "Pickle with the dataset." )

parser.add_argument( "--rocf_reference_path", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocf_reference.png", help = "ROCF reference." )

# WARNING: make sure to choose the correct dataset.
parser.add_argument( "--fold_info_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_augmented_fold_info_tvt_v01/", help = "Directory with fold information of ROCF dataset." )

parser.add_argument( "--trained_model_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/models/", help = "Directory where to save the partial models." )

parser.add_argument( "--stopping_epochs_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/stopping_epochs/", help = "Directory where to save the stopping epochs." )

parser.add_argument( "--problem_type", type = str, default = "mci_det", help = "Defining the problem." )
parser.add_argument( "--model_type", type = str, default = "CLSSaN", help = "Choosing the model." )
parser.add_argument( "--dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the dataset to be used for training." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )

parser.add_argument( "--subset_to_evaluate", type = str, default = "val", help = "Choosing the dataset to be used for predicting." )

parser.add_argument( "--custom_distance_threshold", type = float, default = 1e-5, help = "Distance threshold." )

parser.add_argument( "--output_predictions_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/predictions/", help = "Directory where to save the predictions." )
parser.add_argument( "--output_distances_to_anchor_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/distances_to_anchor/", help = "Directory where to save the plots of the distances to anchor." )

parser.add_argument( "--custom_gpu_id", type = str, default = '0', help = "Choosing which GPU to use by its index." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

dataset_path = args.dataset_path
rocf_reference_path = args.rocf_reference_path
fold_info_dir = args.fold_info_dir

trained_model_dir = args.trained_model_dir
stopping_epochs_dir = args.stopping_epochs_dir

problem_type = args.problem_type
model_type = args.model_type
dataset_type = args.dataset_type
execution_iteration = args.execution_iteration

custom_distance_threshold = args.custom_distance_threshold

subset_to_evaluate = args.subset_to_evaluate

output_predictions_dir = args.output_predictions_dir
output_embeddings_dir = args.output_embeddings_dir
output_distances_to_anchor_dir = args.output_distances_to_anchor_dir

custom_gpu_id = args.custom_gpu_id

# CUDA configuration.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = custom_gpu_id
# Dynamic GPU memory allocation (enable this to not occupy all GPU memory).
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Constants.

NUMBER_OF_FOLDS = 14
STOPPING_EPOCHS_DELIMITER = ','
RESCALE_FACTOR = 1.0 / 255.0

SAVE_PREDICTIONS = True
SAVE_DISTANCES_TO_ANCHOR = True

FOLD_CUSTOMIZED_THRESHOLD = False
# Accuracy.
FOLD_THRESHOLD_LIST = [1.0490e-05,9.2685e-06,1.0580e-05,1.3769e-05,1.0878e-05,2.2948e-05,9.7454e-06,9.4026e-06,2.0027e-05,2.2143e-05,1.2815e-05,1.5527e-05,1.5959e-05,1.9431e-05]
# F1.
#FOLD_THRESHOLD_LIST = [1.0490e-05,9.2685e-06,1.0580e-05,1.3769e-05,1.0878e-05,2.2948e-05,9.7454e-06,9.4026e-06,7.9125e-06,8.4788e-06,1.2815e-05,1.5527e-05,1.5959e-05,1.9431e-05]

SIAMESE_MODEL_IMAGE_HEIGHT = 384
SIAMESE_MODEL_IMAGE_WIDTH = 384

# SaN specific.
SAN_L6_KERNEL_SIZE = 13
SAN_L7_FILTERS = 256

'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  output_model_base_name = problem_type + '_' + model_type + '_with_' + dataset_type + '_i' + str( execution_iteration )
  
  # Defining model and stopping epochs directories.
  
  experiment_trained_model_dir = trained_model_dir + output_model_base_name + '/'
  
  print( 'Loading models from: ', experiment_trained_model_dir )
  print( '\n' )
  
  experiment_stopping_epochs_dir = stopping_epochs_dir + output_model_base_name + '/'
  
  print( 'Loading stopping epochs from: ', experiment_stopping_epochs_dir )
  print( '\n' )
  
  # Creating experiment folders.
  
  distance_threshold_as_string = str( custom_distance_threshold )
  distance_threshold_as_string = distance_threshold_as_string.replace( '.', '_' )
  
  experiment_output_predictions_dir = output_predictions_dir + output_model_base_name + '_' + subset_to_evaluate + '_dt' + distance_threshold_as_string + '/'
  if not( os.path.isdir( experiment_output_predictions_dir ) ):
    os.mkdir( experiment_output_predictions_dir )
    
  experiment_output_distances_to_anchor_dir = output_distances_to_anchor_dir + output_model_base_name + '_' + subset_to_evaluate + '/'
  if not( os.path.isdir( experiment_output_distances_to_anchor_dir ) ):
    os.mkdir( experiment_output_distances_to_anchor_dir ) 
    
  # Loading stopping epochs.

  # Stopping epochs file stores the selected model for each fold (represented by its training epoch).
  # Example: [2,4,4,13,5,6,11,6,1,11,4,4,6,7]

  stopping_epochs_path = experiment_stopping_epochs_dir + str( NUMBER_OF_FOLDS ) + 'folds.csv'
  input_stopping_epoch_list = np.genfromtxt( stopping_epochs_path, delimiter = STOPPING_EPOCHS_DELIMITER, dtype = int )
  
  # Unpickling dataset.
  
  X, y, paths = ml_utils.unpickleDataset( dataset_path )
    
  # Predicting.
  
  print( 'Predicting.' )
  print( 'Experiment alias:', output_model_base_name )
  print( '\n' )
    
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
    
    eval_indices = np.genfromtxt( eval_indices_as_csv, delimiter = ';', dtype = int )
    
    print( 'Subset to evaluate:', subset_to_evaluate )
    print( 'Path:', eval_indices_as_csv )
    print( 'Indices:', eval_indices )
    print( '\n' )
        
    # Model architecture definition.
    
    image_a = keras.Input( shape = ( SIAMESE_MODEL_IMAGE_HEIGHT, SIAMESE_MODEL_IMAGE_WIDTH, 1 ) )
    image_b = keras.Input( shape = ( SIAMESE_MODEL_IMAGE_HEIGHT, SIAMESE_MODEL_IMAGE_WIDTH, 1 ) )
    
    feature_extractor = model_creator.siameseSketchANet( 
      SIAMESE_MODEL_IMAGE_HEIGHT, 
      SIAMESE_MODEL_IMAGE_WIDTH, 
      custom_layer_6_kernel_size = SAN_L6_KERNEL_SIZE,
      custom_layer_7_filters = SAN_L7_FILTERS )
    
    feature_vector_a = feature_extractor( image_a )
    feature_vector_b = feature_extractor( image_b )
    
    distance = model_creator.PairCosineDistanceLayer( )( feature_vector_a, feature_vector_b )
    
    model = keras.Model( inputs = [ image_a, image_b ], outputs = distance )
    
    # Loading model weights.
    
    current_stopping_epoch_as_string = '{:04d}'.format( input_stopping_epoch_list[ fold_index ] )
    pretrained_model_path = experiment_trained_model_dir + 'fold' + current_fold_as_string +  '/' + current_stopping_epoch_as_string + '.hdf5'
    
    model.load_weights( pretrained_model_path )
    
    print( 'Loaded model:', pretrained_model_path )
    print( '\n' )
        
    # Option A. Distances between images.
    
    anchor, eval_x = ml_utils.processAnchorAndDataset( rocf_reference_path, X, eval_indices, RESCALE_FACTOR )
    
    anchor_embeddings = feature_extractor.predict(
      x = anchor,
      verbose = 1,
    )
    
    eval_x_embeddings = feature_extractor.predict(
      x = eval_x,
      verbose = 1,
    )
    
    distance_array = []
    
    anchor_single_embedding = anchor_embeddings[ 0 ]
    
    for i in range( eval_x_embeddings.shape[ 0 ] ):
      
      image_embedding = eval_x_embeddings[ i ]
      
      distance_between_anchor_and_image = ml_utils.cosine_distance_cpu( anchor_single_embedding, image_embedding )
      
      distance_array.append( distance_between_anchor_and_image )
     
    print( 'Distances (Option A).' )
    print( distance_array )
    print( '\n' )
    
    # Option B. Distances of pairs.
      
    eval_x = X[ eval_indices ]
    eval_y = y[ eval_indices ]
    ( tuple_eval_x, tuple_eval_y ) = ml_utils.convertDatasetIntoTupleDataset( eval_x, eval_y, rocf_reference_path, RESCALE_FACTOR )
        
    tuple_distances = model.predict(
      x = [ tuple_eval_x[ :, 0 ], tuple_eval_x[ :, 1 ] ],
      verbose = 1,
    )
    
    distance_array_alt = np.squeeze( tuple_distances )
    
    print( 'Distances (Option B).' )
    print( distance_array_alt )
    print( '\n' )
    
    # Converting distances into labels.
        
    if FOLD_CUSTOMIZED_THRESHOLD:
      predictions = ml_utils.convertDistancesIntoClasses( distance_array, FOLD_THRESHOLD_LIST[fold_index] )
    else:
      predictions = ml_utils.convertDistancesIntoClasses( distance_array, custom_distance_threshold )
    
    print( 'Predictions:', predictions )
    print( '\n' )
        
    # Saving the predictions to a CSV file.
    
    if SAVE_PREDICTIONS:
      
      predictions_path = experiment_output_predictions_dir + 'fold' + current_fold_as_string + '.csv'
  
      predictions_dataframe = pd.DataFrame( { "Filename": paths[ eval_indices ], "Predictions": predictions } )
      predictions_dataframe.to_csv( predictions_path, index = False, header = True )
  
      print( 'Saving predictions to', predictions_path )
      print( '\n' )
      
    # Creating a histogram with distances to anchor.
    
    if SAVE_DISTANCES_TO_ANCHOR:
      
      eval_indices_labels = y[ eval_indices ]
      
      distances_to_anchor_as_csv_path = experiment_output_distances_to_anchor_dir + 'fold' + current_fold_as_string + '.csv'
  
      distances_dataframe = pd.DataFrame( { "Filename": paths[ eval_indices ], "Distances": distance_array, "Label": eval_indices_labels } )
      distances_dataframe.to_csv( distances_to_anchor_as_csv_path, index = False, header = True )
  
      print( 'Saving distances to anchor as a CSV file to:', distances_to_anchor_as_csv_path )
      print( '\n' )