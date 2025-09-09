'''
Author: Juan Guerrero Martin.
Creation date: 19 december 2022.
'''

'''
Script structure:
Part 1. Imports.
Part 2. Global constants and variables.
Part 3. Classes.
Part 4. Functions.
'''

'''
Part 1. Imports.
'''

import tensorflow as tf

from tensorflow import keras

'''
Part 2. Global constants and variables.
'''

# Initializers (Keras default).
custom_kernel_initializer = keras.initializers.glorot_uniform()
custom_bias_initializer = keras.initializers.Zeros()

# Activations.
custom_activation_function = 'relu'

'''
Part 3. Classes.
'''

class PairCosineDistanceLayer( keras.layers.Layer ):
    
  def __init__( self, **kwargs ):
    super( ).__init__( **kwargs )
  
  def call( self, vector_a, vector_b ):
        
    # 1) tf.keras.layers.dot = pseudo tf.keras.losses.CosineSimilarity( )
    # 1 if two vectors have angle 0 (min angular distance).
    # 0 if two vectors have angle 90 (half angular distance).
    # -1 if two vectors have angle 180 (max angular distance).
    
    # 2) 1 - cosine_similarity = cosine distance.
    # 0 if two vectors have angle 0 (min angular distance).
    # 1 if two vectors have angle 90 (half angular distance).
    # 2 if two vectors have angle 180 (max angular distance).
    
    # 3) ( 1 - tf.keras.layers.dot ) / 2. Converting to ( 0, 1 ) range.
    # 0 if two vectors have angle 0 (min angular distance).
    # 0.5 if two vectors have angle 90 (half angular distance).
    # 1 if two vectors have angle 180 (max angular distance).
    
    distance = tf.keras.layers.dot( [ vector_a, vector_b ], axes = 1, normalize = True )
    distance = 1.0 - distance
    distance = 0.5 * distance
        
    return distance

'''
Part 4. Functions.
'''

def siameseSketchANet( 
  image_height, 
  image_width, 
  custom_layer_6_kernel_size = 13,
  custom_layer_7_filters = 512 ):
  
  inputs = keras.Input( shape = ( image_height, image_width, 1 ) )
  
  # Convolution and pooling layers.

  valid_padding = 'valid'
  same_padding = 'same'

  custom_pool_size = ( 3, 3 )
  # Original: ( 3, 3 )
  custom_pool_strides = ( 2, 2 )
  # Original: ( 2, 2 )
  
  # Layer 1.

  layer1_filters = 64
  # Original: 64
  layer1_kernel_size = ( 15, 15 )
  # Original: ( 15, 15 )
  layer1_strides = ( 3, 3 )
  # Original: ( 3, 3 )
  
  x = keras.layers.Conv2D( 
    layer1_filters, 
    kernel_size = layer1_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer1_strides,
    padding = valid_padding,
    name = 'conv2d' )( inputs )
  
  x = keras.layers.MaxPooling2D(
    pool_size = custom_pool_size,
    strides = custom_pool_strides,
    name = 'max_pooling2d' )( x )
  
  # Layer 2.

  layer2_filters = 128
  # Original: 128
  layer2_kernel_size = ( 5, 5 )
  # Original: ( 5, 5 )
  layer2_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  
  x = keras.layers.Conv2D( 
    layer2_filters, 
    kernel_size = layer2_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer2_strides,
    padding = valid_padding,
    name = 'conv2d_1' )( x )

  x = keras.layers.MaxPooling2D( 
    pool_size = custom_pool_size,
    strides = custom_pool_strides,
    name = 'max_pooling2d_1' )( x )
  
  # Layers 3, 4 and 5.

  layer345_filters = 256
  # Original: 256
  layer345_kernel_size = ( 3, 3 )
  # Original: ( 3, 3 )
  layer345_strides = ( 1, 1 )
  # Original: ( 1, 1 )

  x = keras.layers.Conv2D( 
    layer345_filters, 
    kernel_size = layer345_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer345_strides,
    padding = same_padding,
    name = 'conv2d_2' )( x )

  x = keras.layers.Conv2D( 
    layer345_filters, 
    kernel_size = layer345_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer345_strides,
    padding = same_padding,
    name = 'conv2d_3' )( x )

  x = keras.layers.Conv2D( 
    layer345_filters, 
    kernel_size = layer345_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer345_strides,
    padding = same_padding,
    name = 'conv2d_4' )( x )

  x = keras.layers.MaxPooling2D(
    pool_size = custom_pool_size,
    strides = custom_pool_strides,
    name = 'max_pooling2d_2' )( x )

  # Layer 6.

  layer6_filters = 512
  # Original: 512
  # Input size: 384x384. ROCF RD and SD images. layer_6_kernel_size = 13.
  # Input size: 256x256. Quick, Draw! images. layer_6_kernel_size = 8.
  layer6_kernel_size = ( custom_layer_6_kernel_size, custom_layer_6_kernel_size )
  # Original: ( N, N ) => Depends on input image dimensions.
  layer6_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  #layer6_dropout = 0.5
  # Original: 0.5

  x = keras.layers.Conv2D( 
    layer6_filters, 
    kernel_size = layer6_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer6_strides,
    padding = valid_padding,
    name = 'conv2d_5' )( x )

  # Ablation study.
  '''
  x = keras.layers.Dropout( 
    layer6_dropout,
    name = 'dropout' )( x )
  '''

  # Layer 7.
  
  layer7_filters = custom_layer_7_filters
  # Original: 512
  layer7_kernel_size = ( 1, 1 )
  # Original: ( 1, 1 )
  layer7_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  #layer7_dropout = 0.5
  # Original: 0.5

  x = keras.layers.Conv2D( 
    layer7_filters, 
    kernel_size = layer7_kernel_size,
    activation = custom_activation_function,
    kernel_initializer = custom_kernel_initializer,
    bias_initializer = custom_bias_initializer,
    strides = layer7_strides,
    padding = valid_padding,
    name = 'conv2d_6' )( x )

  # Ablation study.
  '''
  x = keras.layers.Dropout( 
    layer7_dropout,
    name = 'dropout_1' )( x )
  '''
  
  # Output layer.
  
  outputs = keras.layers.Flatten( 
    name = 'flatten' )( x )
  
  model = keras.Model( inputs, outputs )
  
  return model

# Name: Sketch-a-Net
# Authors: Qian Yu et al.
# Year: 2017.
# DOI: https://doi.org/10.1007/s11263-016-0932-3
def sketchANet( 
  image_height, 
  image_width, 
  num_of_classes, 
  activation_function, 
  custom_layer_6_kernel_size = 13,
  custom_layer_7_filters = 512 ):
    
  model = keras.Sequential()
  
  # Convolution and pooling layers.

  valid_padding = 'valid'
  same_padding = 'same'

  custom_pool_size = ( 3, 3 )
  # Original: ( 3, 3 )
  custom_pool_strides = ( 2, 2 )
  # Original: ( 2, 2 )

  # Layer 1.

  layer1_filters = 64
  # Original: 64
  layer1_kernel_size = ( 15, 15 )
  # Original: ( 15, 15 )
  layer1_strides = ( 3, 3 )
  # Original: ( 3, 3 )

  model.add( keras.layers.Conv2D( layer1_filters, 
                                  kernel_size = layer1_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer1_strides,
                                  padding = valid_padding,
                                  input_shape = ( image_height, image_width, 1 ),
                                  name = 'conv2d' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d' ) )

  # Layer 2.

  layer2_filters = 128
  # Original: 128
  layer2_kernel_size = ( 5, 5 )
  # Original: ( 5, 5 )
  layer2_strides = ( 1, 1 )
  # Original: ( 1, 1 )

  model.add( keras.layers.Conv2D( layer2_filters, 
                                  kernel_size = layer2_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer2_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_1' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d_1' ) )

  # Layers 3, 4 and 5.

  layer345_filters = 256
  # Original: 256
  layer345_kernel_size = ( 3, 3 )
  # Original: ( 3, 3 )
  layer345_strides = ( 1, 1 )
  # Original: ( 1, 1 )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_2' ) )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_3' ) )

  model.add( keras.layers.Conv2D( layer345_filters, 
                                  kernel_size = layer345_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer345_strides,
                                  padding = same_padding,
                                  name = 'conv2d_4' ) )

  model.add( keras.layers.MaxPooling2D( pool_size = custom_pool_size,
                                        strides = custom_pool_strides,
                                        name = 'max_pooling2d_2' ) )
  
  # Layer 6.

  layer6_filters = 512
  # Original: 512
  
  # Input size: 384x384. ROCF RD and SD images. layer_6_kernel_size = 13.
  # Input size: 256x256. Quick, Draw! images. layer_6_kernel_size = 8.
  layer6_kernel_size = ( custom_layer_6_kernel_size, custom_layer_6_kernel_size )
  # Original: ( N, N ) => Depends on input image dimensions.
  
  layer6_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  layer6_dropout = 0.5
  # Original: 0.5

  model.add( keras.layers.Conv2D( layer6_filters, 
                                  kernel_size = layer6_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer6_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_5' ) )

  model.add( keras.layers.Dropout( layer6_dropout,
                                   name = 'dropout' ) ) 

  # Layer 7.
  
  layer7_filters = custom_layer_7_filters
  # Original: 512
  layer7_kernel_size = ( 1, 1 )
  # Original: ( 1, 1 )
  layer7_strides = ( 1, 1 )
  # Original: ( 1, 1 )
  layer7_dropout = 0.5
  # Original: 0.5

  model.add( keras.layers.Conv2D( layer7_filters, 
                                  kernel_size = layer7_kernel_size,
                                  activation = custom_activation_function,
                                  kernel_initializer = custom_kernel_initializer,
                                  bias_initializer = custom_bias_initializer,
                                  strides = layer7_strides,
                                  padding = valid_padding,
                                  name = 'conv2d_6' ) )

  model.add( keras.layers.Dropout( layer7_dropout,
                                   name = 'dropout_1' ) )
  
  # Flatten.
  
  model.add( keras.layers.Flatten( name = 'flatten' ) )
  
  # Output layer.
  
  model.add( keras.layers.Dense( num_of_classes, 
                                 activation = activation_function,
                                 name = 'dense' ) )
  
  return model

# Name: Youn's CNN.
# Authors: Young Chul Youn et al.
# Year: 2021.
# DOI: https://doi.org/10.1186/s13195-021-00821-8
def younsCNN(
  image_height, 
  image_width, 
  num_of_classes, 
  activation_function ):
  
  COMMON_KERNEL_SIZE = ( 3, 3 )
  COMMON_ACTIVATION = 'relu'
  COMMON_PADDING = 'same'
  
  CUSTOM_DROPOUT = 0.4
  
  input = keras.Input( shape = ( image_height, image_width, 1 ) )
  
  x = keras.layers.Conv2D( 64, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( input )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 64, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 64, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 64, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 128, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Dropout( CUSTOM_DROPOUT )( x )
    
  x = keras.layers.Flatten( )( x )
  
  x = keras.layers.Dense( 128, activation = COMMON_ACTIVATION )( x )
    
  output = keras.layers.Dense( num_of_classes, activation = activation_function )( x )
    
  model = keras.Model( input, output )
  
  return model

# Name: Park's CNN.
# Authors: Jin Hyuck Park et al.
# Year: 2024.
# DOI: https://doi.org/10.1186/s12888-024-05622-5
def parksCNN(
  image_height, 
  image_width, 
  num_of_classes, 
  activation_function ):
  
  COMMON_KERNEL_SIZE = ( 3, 3 )
  COMMON_ACTIVATION = 'relu'
  COMMON_PADDING = 'same'
  
  CUSTOM_DROPOUT = 0.4
  
  input = keras.Input( shape = ( image_height, image_width, 1 ) )
  
  x = keras.layers.Conv2D( 32, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( input )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 32, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 32, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
  
  x = keras.layers.Conv2D( 32, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( )( x )
    
  x = keras.layers.Flatten( )( x )

  x = keras.layers.Dense( 256, activation = COMMON_ACTIVATION )( x )
  x = keras.layers.Dropout( CUSTOM_DROPOUT )( x )
    
  x = keras.layers.Dense( 128, activation = COMMON_ACTIVATION )( x )
  x = keras.layers.Dropout( CUSTOM_DROPOUT )( x )
    
  output = keras.layers.Dense( num_of_classes, activation = activation_function )( x )
    
  model = keras.Model( input, output )
  
  return model

# Name: Convolutional Auto Encoder + Multilayer Perceptron.
# Authors: Wen-Ting Cheah et al.
# Year: 2019.
# DOI: https://doi.org/10.1109/SMC.2019.8913880

# CAE part.
def convolutionalAutoEncoder( 
  image_height, 
  image_width ):
  
  COMMON_KERNEL_SIZE = ( 3, 3 )
  COMMON_POOL_SIZE = ( 2, 2 )
  COMMON_ACTIVATION = 'relu'
  COMMON_PADDING = 'same'
  
  COMMON_DECONV_STRIDE = 2
  
  input = keras.Input( shape = ( image_height, image_width, 1 ) )
  
  # Encoder.
  
  x = keras.layers.Conv2D( 128, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( input )
  x = keras.layers.MaxPooling2D( COMMON_POOL_SIZE, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2D( 64, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( COMMON_POOL_SIZE, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2D( 32, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( COMMON_POOL_SIZE, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2D( 16, COMMON_KERNEL_SIZE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  x = keras.layers.MaxPooling2D( COMMON_POOL_SIZE, padding = COMMON_PADDING )( x )
  
  # Decoder.
  
  # Conv2DTranspose performs both deconvolution and upsampling (or unpooling).
  
  x = keras.layers.Conv2DTranspose( 16, COMMON_KERNEL_SIZE, strides = COMMON_DECONV_STRIDE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2DTranspose( 32, COMMON_KERNEL_SIZE, strides = COMMON_DECONV_STRIDE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2DTranspose( 64, COMMON_KERNEL_SIZE, strides = COMMON_DECONV_STRIDE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  
  x = keras.layers.Conv2DTranspose( 128, COMMON_KERNEL_SIZE, strides = COMMON_DECONV_STRIDE, activation = COMMON_ACTIVATION, padding = COMMON_PADDING )( x )
  
  output = keras.layers.Conv2D( 1, COMMON_KERNEL_SIZE, activation = 'sigmoid', padding = 'same' )( x )       
  
  model = keras.Model( input, output )
  
  return model

# MLP part.
def simpleNeuralNetwork( embedding_length, num_of_classes, activation_function ):
  
  model = keras.Sequential( )
  model.add( keras.Input( shape = ( embedding_length, ) ) )
  model.add( keras.layers.Dense( 2048, activation = 'relu' ) )
  model.add( keras.layers.Dense( 512, activation = 'relu' ) )
  model.add( keras.layers.Dense( num_of_classes, activation = activation_function ) )
  return model
