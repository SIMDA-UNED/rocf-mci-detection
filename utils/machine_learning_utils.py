'''
Author: Juan Guerrero Martí­n.
Creation date: 19 december 2022.
'''

'''
Script structure:
Part 1. Imports.
Part 2. Classes.
Part 3. Functions.
'''

'''
Part 1. Imports.
'''

import cv2
import numpy as np
import pickle
import tensorflow as tf

from tensorflow import keras

'''
Part 2. Functions.
'''

def unpickleDataset( pickle_file ):
  
  with open( pickle_file, 'rb' ) as pickle_file_descriptor:
  	pickle_file_elements = pickle.load( pickle_file_descriptor )
  	X = pickle_file_elements['X']
  	y = pickle_file_elements['y']
  	paths = pickle_file_elements['paths']
  
  # Converting arrays to tensor format (using numpy library).    
  
  X = np.array( X, dtype = np.float32 )
  # X.shape[1] and X.shape[2] may need to be interchanged.
  X = np.reshape( X, ( X.shape[0], X.shape[1], X.shape[2], 1 ) )
        
  y = np.array( y, dtype = np.uint8 )
  
  paths = np.array( paths )
  
  # Feedback.
  #'''
  print( 'Dataset information.' )
  print( '\n' )
  print( 'X shape:', X.shape )
  print( 'y shape:', y.shape )
  print( 'paths shape:', paths.shape )
  print( '\n' )
  #np.set_printoptions( threshold = sys.maxsize )
  CUSTOM_INDEX = 0
  print( 'First label (y):', y[CUSTOM_INDEX] )
  print( 'Label type. Regression: float. Classification: int.' )
  print( 'First label (y) type:', type( y[CUSTOM_INDEX] ) )
  print( 'First path:', paths[CUSTOM_INDEX] )
  print( '\n' )
  #print( y )
  #'''
  
  return X, y, paths

def convertToDDHHMMSSFormatAndPrint( seconds ):

  seconds_in_day = 86400
  seconds_in_hour = 3600
  seconds_in_minute = 60
  
  seconds = int( seconds )
  
  days = seconds // seconds_in_day
  seconds = seconds - (days * seconds_in_day)
  
  hours = seconds // seconds_in_hour
  seconds = seconds - (hours * seconds_in_hour)
  
  minutes = seconds // seconds_in_minute
  seconds = seconds - (minutes * seconds_in_minute)
  
  print("{0:.0f} days, {1:.0f} hours, {2:.0f} minutes, {3:.0f} seconds.".format(
      days, hours, minutes, seconds))
  print( '\n' )

  time_as_string = "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format( int( days ), int( hours ), int( minutes ), int( seconds ) )

  return time_as_string

def convertDatasetIntoTupleDataset( x, y, rocf_reference_path, rescale_factor ):
  
  # Loading the ROCF reference.
  
  rocf_reference = cv2.imread( rocf_reference_path, cv2.IMREAD_GRAYSCALE )
  rocf_reference = np.array( rocf_reference, dtype = np.float32 )
  
  # Rescaling all image pixel values.
  
  x_dim_0_size = x[ 0 ].shape[ 0 ]
  x_dim_1_size = x[ 0 ].shape[ 1 ]
  x_dim_2_size = x[ 0 ].shape[ 2 ]
  
  rocf_reference = rescale_factor * rocf_reference
  rocf_reference = rocf_reference.reshape( x_dim_0_size, x_dim_1_size, x_dim_2_size )
  
  for i in range( len( x ) ):
    x[ i ] = rescale_factor * x[ i ]
  
  # Generating the tuple datasets.
  
  tuple_x = [ ]
  tuple_y = [ ]
  
  for i in range( len( x ) ):
    
    tuple_x.append( [ rocf_reference, x[ i ] ] )
    
    # ROCF reference = 0. ROCF copy = 0. y = 1 (the image labels are equal = Healthy).
    # ROCF reference = 0. ROCF copy = 1. y = 0 (the image labels are different).
    current_tuple_y = None
    if( y[ i ] == 0 ):
      current_tuple_y = 1
    else:
      current_tuple_y = 0
      
    tuple_y.append( [ current_tuple_y ] )
      
  return ( np.array( tuple_x ), np.array( tuple_y ) )

# Formula.
# L = y * d + ( 1 - y ) * max( m - d, 0 )
# d = any distance function.  
def generic_contrastive_loss( y, preds, margin = 5e-5 ):
  
  y = tf.cast( y, preds.dtype )
      
  margin_minus_preds = keras.backend.maximum( margin - preds, 0 )
  
  loss = y * preds + (1 - y) * margin_minus_preds

  return loss

def convertProbabilitiesIntoClasses( probability_array, probability_threshold ):
  
  classes = [ ]  
  
  for probability in probability_array:  
    
    if probability < probability_threshold:
      classes.append( 0 )
    else:
      classes.append( 1 )   
  
  return classes

def processAnchorAndDataset( rocf_reference_path, input_x, indices, rescale_factor ):
  
  x = input_x[ indices ]
  
  # Loading the ROCF reference.
  
  rocf_reference = cv2.imread( rocf_reference_path, cv2.IMREAD_GRAYSCALE )
  rocf_reference = np.array( rocf_reference, dtype = np.float32 )
      
  # Rescaling all image pixel values.
      
  rocf_reference = rescale_factor * rocf_reference
  
  for i in range( len( x ) ):
    x[ i ] = rescale_factor * x[ i ]
    
  # Adding third dimension to ROCF reference.
    
  x_dim_0_size = x[ 0 ].shape[ 0 ]
  x_dim_1_size = x[ 0 ].shape[ 1 ]
  x_dim_2_size = x[ 0 ].shape[ 2 ]
  
  rocf_reference = rocf_reference.reshape( x_dim_0_size, x_dim_1_size, x_dim_2_size )
  
  rocf_reference = np.expand_dims( rocf_reference, axis = 0 )
    
  return rocf_reference, x

def cosine_distance_cpu( vector_a, vector_b ):
  
  # cosine_similarity range = [-1,1]
  dot_product = np.dot( vector_a, vector_b )
  norm_a = np.linalg.norm( vector_a )
  norm_b = np.linalg.norm( vector_b )
  cosine_similarity = dot_product / ( norm_a * norm_b )
  
  # cosine_distance range = [0,2] 
  cosine_distance = 1.0 - cosine_similarity
  
  # We convert cosine_distance range to [0,1]
  cosine_distance *= 0.5
  
  return cosine_distance

def convertDistancesIntoClasses( distance_array, distance_threshold ):
  
  classes = [ ]
  
  for distance in distance_array:
    
    # If a given ROCF is sufficiently similar to the reference ROCF, then its class is Healthy.
    if distance < distance_threshold:
      classes.append( 0 )
    # Otherwise, its class is MCI.
    else:
      classes.append( 1 )
      
  return classes

# Generated by Gemini 2.0 Flash (2025-05-15).
def calculate_accuracy(tp, fp, fn, tn):
  """
  Calculates the accuracy from the values of a 2x2 confusion matrix.

  Args:
    tp: True Positives
    fp: False Positives
    fn: False Negatives
    tn: True Negatives

  Returns:
    The accuracy (float).
  """
  total_predictions = tp + fp + fn + tn
  if total_predictions == 0:
    return 0
  else:
    accuracy = (tp + tn) / total_predictions
    return accuracy

# Generated by Gemini 2.0 Flash (2025-05-15).
def calculate_f1_score(tp, fp, fn, tn):
  """
  Calculates the F1 score from the values of a 2x2 confusion matrix.

  Args:
    tp: True Positives
    fp: False Positives
    fn: False Negatives
    tn: True Negatives (This value is used for completeness but not directly in F1)

  Returns:
    The F1 score (float) or 0 if precision or recall is 0.
  """
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0

  if precision == 0 or recall == 0:
    return 0
  else:
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def saveConfusionMatrixWithHeader( output_path, output_matrix, num_of_classes, matrix_value_separator ):
  
  confusion_matrix_header = ''
  for i in range( num_of_classes ):
    confusion_matrix_header += matrix_value_separator + str( i )
  confusion_matrix_header += '\n'
    
  with open( output_path, 'w' ) as output_file:

    output_file.write( confusion_matrix_header )

    for i in range( num_of_classes ):
  
      cm_row = output_matrix[i]
      cm_row_as_list = cm_row.tolist()
      cm_row_as_string = matrix_value_separator.join( [str( element ) for element in cm_row_as_list] )
      output_line = str( i ) + matrix_value_separator + cm_row_as_string + '\n'
      output_file.write( output_line )  