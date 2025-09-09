'''
Author: Juan Guerrero Martin.
Creation date: 19 december 2022.
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
import cv2
import natsort
import numpy as np
import os
import pandas as pd
import pickle

'''
Part 2. Argument parser.
'''

parser = argparse.ArgumentParser( description = 'dataset_to_pickle' )

parser.add_argument( "--dataset_dir", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented/", help = "Directory with images." )
parser.add_argument( "--dataset_info", type = str, default = "/home/jguerrero/Desarrollo/rocf-mci-detection/data/dataset_information/rocfd528_augmented_info.csv", help = "CSV with image information." )
parser.add_argument( "--dataset_pickle", type = str, default = "/home/jguerrero/Desarrollo/DATA/proyecto_REY/datasets/rocfd528_augmented.pickle", help = "Pickle with augmented ROCFD528 images and labels." )
                    
args = parser.parse_args( )
                    
'''
Part 3. Global constants and variables.
'''

dataset_dir = args.dataset_dir
dataset_info = args.dataset_info
dataset_pickle = args.dataset_pickle
     
CSV_DELIMITER = ';'
               
'''
Part 4. Main body.
'''

if __name__ == '__main__':
  
  dataset_info_dataframe = pd.read_csv( dataset_info, sep = CSV_DELIMITER )
  
  # Feedback.
  '''
  print( dataset_info )
  print( dataset_info_dataframe )
  print( '\n' )
  '''
  
  X = []
  y = []
  paths = []
  
  image_list = os.listdir( dataset_dir )

  ordered_image_list = natsort.natsorted( image_list, reverse = False )
  
  for file in ordered_image_list:

    if file.endswith( ".png" ):
      
      file_name = os.path.splitext( file )[0]
      
      ## Processing paths.

      paths.append( file_name )
      
      ## Processing X.
      
      image_path = os.path.join( dataset_dir, file )
      image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE )
      
      # Feedback.
      '''
      print( image.shape )
      print( np.amin( image ) )
      print( np.amax( image ) )
      print( '\n' )
      '''
      
      # Normalizing. Pixels are kept in the format (0, 255).
      image = np.array( image, dtype = np.float32 )
      
      X.append( image )
      
      ## Processing y.
      
      # For ROCFAD.
      row_of_interest = dataset_info_dataframe.loc[ dataset_info_dataframe['figure_path'] == file ]
      
      health_profile_label = int( row_of_interest.iloc[0]['health_profile_label'] )
            
      y.append( health_profile_label )

  # Creating the pickle.
  rocf_dataset = {}
  rocf_dataset['X'] = X
  rocf_dataset['y'] = y
  rocf_dataset['paths'] = paths
  
  # Feedback.
  '''
  print( y )
  print( paths )
  '''
  
  # Saving the pickle.
  #'''
  print( 'Generating pickle...' )
  with open( dataset_pickle, 'wb' ) as file:
    pickle.dump( rocf_dataset, file )
  print( 'Done.' )
  
  print( "Output path:", dataset_pickle )
  #'''