
#Credits @article{ahmed2016house, title={House price estimation from visual and textual features}, author={Ahmed, Eman and Moustafa, Mohamed}, journal={arXiv preprint arXiv:1609.08399}, year={2016} }
#Main Code that this was adapted from: https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/





#Import necessary Packages
#code file is called glauc_mult_dataset.py
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
#from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import os
#All files such as the csv file and the images should all be in the same file
#Doing so will enable the program to find all of the necessary data

#reads in CSV data from the image path that was taken
#from the pathway specified
#will look for the specific file name
#read in the csv data
def load_eye_attribute(inputpath):
    #initialize the list of column names in the  CSV file  
    #load it using Pandas
    cols = ["Age", "BMO", "NFL", "Lamina Depth", "MWR", "Glaucoma Diagnosis"]    
    #cols = ["Age", "BMO", "NFL", "Lamina Depth", "MWR", "NFLT", "NFLIN", "Glaucoma Diagnosis"]                         
    df = pd.read_csv(inputpath, sep = ",", header = None, names = cols)
    
    #initialize the OneHotEncoder
    #encoder = OneHotEncoder(sparse = False)
    
    #Fit and transform the data
    #encoded_labels = encoder.fit_transform(df[["Glaucoma Diagnosis"]])
    
    
    
    #return the Data Frame
    return df

#processes the eye attribute/data in the csv 
#initializing what is the continous data
#initializes catagorical data by setting it as binary
#combines the catagorical data and continous data
#returns the all of the data as X variables for test and train
# =============================================================================
# def process_eye_attributes(df, train, test):
#     #initializes the column names of the continuous data
#     continuous = ["Age", "BMO", "NFL", "Lamina Depth", "MWR"]                     #normally, diagnoses column needs to be manually dropped                   
#                                                                                #but this module automatically drops it this way
#     trainContinuous = train[continuous]                                  
#     testContinuous = test[continuous]
#     
#     funcBinarizer = LabelBinarizer().fit(df["EyeFuncLoss"])                                      #Change this section to be glaucoma as the binarize takes in numerical values from the csv volumes and turn them into ints
#     
#     #trainCategorical = funcBinarizer.transform(train["EyeFuncLoss"])                             #Change this section to be glaucoma as the binarize takes in numerical values from the csv volumes and turn them into int
#     #testCategorical = funcBinarizer.transform(test["EyeFuncLoss"])   
#     trainCategorical = funcBinarizer.transform(train["EyeFuncLoss"]).astype(float)     #changes the categorical data into floats to be read in, will need to remove this when not feeding in functloss
#     testCategorical = funcBinarizer.transform(test["EyeFuncLoss"]).astype(float)       #changes the categorical data into floats to be read in, will need to remove this when not feeding in functloss
#                          
#     #construct our training and testing data poitns by concatenating
#     #the categorical features with the continuous features
#     trainX = np.hstack([trainCategorical, trainContinuous])
#     testX = np.hstack([testCategorical, testContinuous])
#     
#     #retunrs the concatenated training and testing data
#     return (trainX, testX)
# =============================================================================

def process_eye_attributes(df, train, test):
    #initializes the column names of the continuous data
    continuous = ["Age", "BMO", "NFL", "Lamina Depth", "MWR"]
    #continuous = ["Age", "BMO", "NFL", "Lamina Depth", "MWR", "NFLT", "NFLIN"]
    trainContinuous = train[continuous]                                  
    testContinuous = test[continuous]
    
    # Binarize the encoded EyeFuncLoss
    #funcBinarizer = LabelBinarizer()
    #trainCategorical = funcBinarizer.fit_transform(train['EyeFuncLoss'])
    #testCategorical = funcBinarizer.transform(test['EyeFuncLoss'])
    #trainCategorical = train["EyeFuncLoss"]
    #testCategorical = test["EyeFuncLoss"]
    
    #trainCategorical = trainCategorical.values.reshape(-1, 1) #reshapes the dimensions so that it can work through np.hstack
    #testCategorical = testCategorical.values.reshape(-1, 1) #reshapes the dimensions so that it can work through np.hstack
    
    
    


    # Construct the training and testing datasets
    #trainX = np.hstack([trainCategorical, trainContinuous])
    #testX = np.hstack([testCategorical, testContinuous])
    trainX = trainContinuous
    testX = testContinuous
    

    return (trainX, testX)


#Loads in the images of the eye within the data set
#sets each images of the eye to be 128,128
#stitches four images into one big images of 256, 256
#Top right images SLO images (image of the whole Eye)
#Top left image is the 45 degree segmented image of BMO & ILM
#Bottom Right image is radial scan of 135 degrees
#Bottom Left is radial scan of 45 degrees
#appends the stitched image into an array and outputs it
def load_eye_images(df, inputpath):                                                       #I noticed that the original model uses "inputPath" with a capitalized P, whereas I didn't, This may pose a problem. We will see
    #initialize our images array (i.e., the eye images themselves)                        #I did this for all paths
    images = []
    
    #loop over the indexes of the eyes                                                    #We want to organize the image names based by numbers of 1-26 at the end or beginning of the name
    for i in df.index.values:
        #find the four images for the eye and for the file paths,
        #ensuring the four are always in the same order                                     #We may use more than four images so stitching and adjusting is very important 
        basepath = os.path.sep.join([inputpath, "{}_*".format(i + 1)])                      #will need more verifiction on how {}_* would totally work, could manipulate this to be better 
        eyepaths = sorted(list(glob.glob(basepath)))                                      #will need more verificaiton on this
        
        #initialize our list of imput images along with the outout image
        #after combining the four input images                                              #the following code will have to be changed depending on the number of images we want to stitch together. maybe like 6 or 16?
        inputimages = []                                                                    #the original model capitalizes "I" in "inputImages", I did not
        
        #outputimage = np.zeros((128, 128, 12), dtype = "uint8")
        outputimage = np.zeros((128, 128, 9), dtype = "uint8")                 #for all greyscale images
        
        
        #loop over hte input hosue paths
        for eyepath in eyepaths:                                                        
            #load the input iamge, resize it to be 32, 32, and then 
            #update the list of input images
            image = cv2.imread(eyepath)
            image = cv2.imread(eyepath, cv2.IMREAD_GRAYSCALE)                  #Should ensure that all images are now greyscaled (didn't have the opportunity to check every image personally)
            image = cv2.resize(image, (128, 128))                               #think abot how to increase speed of training , maybe I should resize only one dimension. the height is shorter, and the wideth has less imfo
            #image = cv2.resize(image, (64, 64))
            image = np.expand_dims(image, axis=-1)                              # Convert shape from (128, 128) to (128, 128, 1)
            inputimages.append(image)
            
       
        #concatenates the images on a channel dimension
        #outputimage = tf.concat(inputimages, axis = 0)
        outputimage = np.concatenate(inputimages, axis=2) # Concatenate along the third dimension to get 128*128*12
        
       
        #add the tiled image to our set of images the network will be 
        #trained on
        images.append(outputimage)
        
    #returns our set of images
    return np.array(images)




        
        
        
    
            
        
        
    
