'''
Author: Juan Guerrero Martin.
Creation date: 27 december 2022.
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

# In order to use sibling modules.
sys.path.append( ".." )
import utils.machine_learning_utils as ml_utils

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'extract_metrics' )

parser.add_argument( "--dataset_information_path", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_info.csv", help = "CSV with ROCFD528 dataset information." )

parser.add_argument( "--input_predictions_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/predictions/", help = "Directory with the predictions." )

parser.add_argument( "--problem_type", type = str, default = "mci_det", help = "Choosing the problem type." )
parser.add_argument( "--model_type", type = str, default = "SaN", help = "Choosing the model to be used for training." )
parser.add_argument( "--dataset_type", type = str, default = "ROCFD528_augmented_v01", help = "Choosing the dataset to be used for training." )
parser.add_argument( "--execution_iteration", type = int, default = 0, help = "Execution iteration." )

parser.add_argument( "--subset_to_evaluate", type = str, default = "val", help = "Choosing the dataset to be used for predicting." )

parser.add_argument( "--threshold", type = float, default = 0.5, help = "Threshold." )

parser.add_argument( "--output_metrics_confusion_matrices_dir", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/metrics_confusion_matrices/", help = "Directory where to save the confusion matrices." )

args = parser.parse_args( )

'''
Part 3. Global constants and variables.
'''

# Variables.

dataset_information_path = args.dataset_information_path 

input_predictions_dir = args.input_predictions_dir

problem_type = args.problem_type
model_type = args.model_type
dataset_type = args.dataset_type
execution_iteration = args.execution_iteration

subset_to_evaluate = args.subset_to_evaluate

threshold = args.threshold

output_metrics_confusion_matrices_dir = args.output_metrics_confusion_matrices_dir

# Constants.
  
NUMBER_OF_FOLDS = 14
NUMBER_OF_CLASSES = 2

INFORMATION_CSV_DELIMITER = ';'
PREDICTIONS_CSV_DELIMITER = ','
CONFUSION_MATRIX_CSV_DELIMITER = ','

METRIC_DECIMALS = 4
METRIC_AS_PERCENTAGE_DECIMALS = 2

DO_SAVE_CONFUSION_MATRICES = True

'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  model_alias = model_type + '_with_' + dataset_type

  output_model_base_name = None
  if model_type == 'SaN' or model_type == 'YCNN' or model_type == 'PCNN' or model_type == 'MLP':
    class_probability_threshold_as_string = "{:.2f}".format( threshold )
    class_probability_threshold_as_string = class_probability_threshold_as_string.replace( '.', '_' )
    output_model_base_name = problem_type + '_' + model_alias + '_i' + str( execution_iteration ) + '_' + subset_to_evaluate + '_pt' + class_probability_threshold_as_string
  elif model_type == 'CLSSaN':
    distance_threshold_as_string = str( threshold )
    distance_threshold_as_string = distance_threshold_as_string.replace( '.', '_' )
    output_model_base_name = problem_type + '_' + model_alias + '_i' + str( execution_iteration ) + '_' + subset_to_evaluate + '_dt' + distance_threshold_as_string 
  else:
    print( 'Model type unknown.' )
    sys.exit( -1 )
  
  experiment_input_predictions_dir = input_predictions_dir + output_model_base_name + '/'
  
  print( 'Loading predictions from: ', experiment_input_predictions_dir )
  print( '\n' )
    
  # Creating experiment folders.
    
  experiment_output_metrics_confusion_matrices_dir = output_metrics_confusion_matrices_dir + output_model_base_name + '/'
  if not( os.path.isdir( experiment_output_metrics_confusion_matrices_dir ) ):
    os.mkdir( experiment_output_metrics_confusion_matrices_dir )
    
  # Creating information dataframe.
  
  print( 'Loading dataset information from: ', dataset_information_path )
  print( '\n' )
  
  dataset_information_dataframe = pd.read_csv( dataset_information_path, sep = INFORMATION_CSV_DELIMITER )
  
  # Feedback.
  '''
  print( dataset_information_dataframe )
  print( '\n' )
  '''

  # List for the metric values in each fold.
    
  accuracy_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )
  f1_list = np.zeros( shape = ( NUMBER_OF_FOLDS ), dtype = float )
  
  cumulative_confusion_matrix = np.zeros( shape = ( NUMBER_OF_CLASSES, NUMBER_OF_CLASSES ), dtype = int )
  
  for fold_index in range( 0, NUMBER_OF_FOLDS ):
    
    fold_counter = fold_index + 1
    
    current_fold_as_string = '{:02d}'.format( fold_counter )
        
    predictions_path = experiment_input_predictions_dir + 'fold' + current_fold_as_string + '.csv'
    
    predictions_dataframe = pd.read_csv( predictions_path, sep = PREDICTIONS_CSV_DELIMITER )
    
    fold_confusion_matrix = np.zeros( shape = ( NUMBER_OF_CLASSES, NUMBER_OF_CLASSES ), dtype = int )
    
    for index, row in predictions_dataframe.iterrows( ):
      
      # Predicted label.
      
      predicted_health_profile_label = int( row['Predictions'] )
      
      # Actual label.
      
      current_figure_id_and_name = row['Filename']
      # current_figure_id_and_name is 528_figure_name. The first 4 characters are "120_".
      current_figure_name = current_figure_id_and_name[4:]
      dataset_information_row = dataset_information_dataframe.loc[ dataset_information_dataframe['figure_name'] == current_figure_name ]
      
      actual_health_profile_label = dataset_information_row['health_profile_label'].values[0]
        
      # Generating the confusion matrices.
      
      fold_confusion_matrix[ actual_health_profile_label, predicted_health_profile_label ] += 1
      cumulative_confusion_matrix[ actual_health_profile_label, predicted_health_profile_label ] += 1
      
      # for row
      
    tn = fold_confusion_matrix[ 0 ][ 0 ]
    fp = fold_confusion_matrix[ 0 ][ 1 ]
    fn = fold_confusion_matrix[ 1 ][ 0 ]
    tp = fold_confusion_matrix[ 1 ][ 1 ]
    
    acc = np.round( ml_utils.calculate_accuracy( tp, fp, fn, tn ), decimals = 4 )
    f1 = np.round( ml_utils.calculate_f1_score( tp, fp, fn, tn ), decimals = 4 )
      
    accuracy_list[ fold_index ] = acc
    f1_list[ fold_index ] = f1
    
    if DO_SAVE_CONFUSION_MATRICES:
      
      # Saving fold confusion matrix.
      fold_confusion_matrix_path = experiment_output_metrics_confusion_matrices_dir + 'fold' + current_fold_as_string + '.csv'
      if not( os.path.isfile( fold_confusion_matrix_path ) ):
        ml_utils.saveConfusionMatrixWithHeader( fold_confusion_matrix_path, fold_confusion_matrix, NUMBER_OF_CLASSES, CONFUSION_MATRIX_CSV_DELIMITER )
        print( 'Fold confusion matrix saved to:', fold_confusion_matrix_path )
        print( '\n' )
        
    # for fold

  # Accuracy.
  
  print( 'Accuracies:', accuracy_list )
  
  mean_accuracy_as_proportion = np.round( np.mean( accuracy_list ), decimals = METRIC_DECIMALS )
  mean_accuracy_as_percentage = np.round( mean_accuracy_as_proportion * 100.0, decimals = METRIC_AS_PERCENTAGE_DECIMALS )
  
  print( 'mean:', mean_accuracy_as_proportion, '(', mean_accuracy_as_percentage, '% )'  )
  
  std_accuracy_as_proportion = np.round( np.std( accuracy_list ), decimals = METRIC_DECIMALS )
  std_accuracy_as_percentage = np.round( std_accuracy_as_proportion * 100.0, decimals = METRIC_AS_PERCENTAGE_DECIMALS )
  
  print( 'std:', std_accuracy_as_proportion, '(', std_accuracy_as_percentage, '% )'  )    

  # F1 score.
        
  print( 'F1 scores:', f1_list )
  
  mean_f1_as_proportion = np.round( np.mean( f1_list ), decimals = METRIC_DECIMALS )
  mean_f1_as_percentage = np.round( mean_f1_as_proportion * 100.0, decimals = METRIC_AS_PERCENTAGE_DECIMALS )
  
  print( 'mean:', mean_f1_as_proportion, '(', mean_f1_as_percentage, '% )'  )
  
  std_f1_as_proportion = np.round( np.std( f1_list ), decimals = METRIC_DECIMALS )
  std_f1_as_percentage = np.round( std_f1_as_proportion * 100.0, decimals = METRIC_AS_PERCENTAGE_DECIMALS )
  
  print( 'std:', std_f1_as_proportion, '(', std_f1_as_percentage, '% )'  )
  
  if DO_SAVE_CONFUSION_MATRICES:

    # Saving cumulative confusion matrix.
    cumulative_confusion_matrix_path = experiment_output_metrics_confusion_matrices_dir + 'cumulative.csv'
    if not( os.path.isfile( cumulative_confusion_matrix_path ) ):
      ml_utils.saveConfusionMatrixWithHeader( cumulative_confusion_matrix_path, cumulative_confusion_matrix, NUMBER_OF_CLASSES, CONFUSION_MATRIX_CSV_DELIMITER )
      print( 'Cumulative confusion matrix saved to:', cumulative_confusion_matrix_path )
      print( '\n' )