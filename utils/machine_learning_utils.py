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

# Formula.
# L = y * d + ( 1 - y ) * max( m - d, 0 )
# d = any distance function.  
def generic_contrastive_loss( y, preds, margin = 5e-5 ):
  
  y = tf.cast( y, preds.dtype )
      
  margin_minus_preds = keras.backend.maximum( margin - preds, 0 )
  
  loss = y * preds + (1 - y) * margin_minus_preds

  return loss