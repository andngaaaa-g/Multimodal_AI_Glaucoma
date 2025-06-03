#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:29:33 2024

@author: Andy Ng
"""



#import the necessary packages
#Code file is glauc_mult_models.py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization

def create_mlp(dim, regress=False):
    # Define our Multi-Level Processor
    model = Sequential()
    model.add(Input(shape=(dim,)))  # Use Input layer to specify input shape
    model.add(Dense(8, activation="relu"))  # No need for input_dim here
    #model.add(Dropout(0.3)) #added on 10/23 after second run. this led to very bad results, accuracy was at a standstill of 70 percent
    model.add(Dense(4, activation="relu"))
    
    # Check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
        
    # Return our model
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # Initialize the input shape and channel dimension
    inputshape = (height, width, depth)  # Assuming TensorFlow/channels-last ordering
    
    # Define the model input 
    inputs = Input(shape=inputshape)
    
    # Loop over the number of filters
    x = inputs  # Start with the input layer
    for (i, f) in enumerate(filters):
        # Conv => ReLU => BN => Pool
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # Flatten the volume, then FC => ReLU => Dropout
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)
        
    # Apply another FC layer, this one to match the number of nodes coming out of the MLP
    x = Dense(4, activation="relu")(x)
        
    # Check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
        
    # Construct the CNN
    model = Model(inputs, x)
        
    # Return the model
    return model

    
    
        
        
        
        
    

