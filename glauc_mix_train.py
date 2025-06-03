#!/usr/bin/env python3
#Credits @article{ahmed2016house, title={House price estimation from visual and textual features}, author={Ahmed, Eman and Moustafa, Mohamed}, journal={arXiv preprint arXiv:1609.08399}, year={2016} }
#Main Code that this was adapted from: https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

"""
Created on Mon Jun  3 12:48:54 2024

@author:0
this now does stratified random sampling
"""

# Importing the necessary packages
import glauc_mult_datasets
import glauc_mult_models                                            
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np
import argparse
import os

rand_num = 35
folder_path = f'/Users/hyang/Desktop/F1/Stratified and additional factors/RS {rand_num}'
# Construct the argument parser and parse the arguments                         
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path/name of input dataset of eye images")
args = vars(ap.parse_args())
print(f'This is Random State {rand_num}')
print("[INFO] loading eye attributes...\n\n\n")
inputPath = os.path.sep.join([args["dataset"], "FV9.5.csv"])

# Check if the file exists
if not os.path.isfile(inputPath):
    print(f"[ERROR] The file {inputPath} does not exist.")
    exit(1)
    
df = glauc_mult_datasets.load_eye_attribute(inputPath)

# Load the images and scale pixel intensities to the range [0, 1]
print("[INFO] loading eye images...")
images = glauc_mult_datasets.load_eye_images(df, args["dataset"])
images = images / 255.0

# Partition the data into training, validation, and testing splits
print("[INFO] processing data...")

# Extract labels for stratification
labels = df["Glaucoma Diagnosis"]

# First, split off 10% for the testing set (stratified)
trainValAttrX, testAttrX, trainValImagesX, testImagesX, trainValLabels, testLabels = train_test_split(
                                                                                                    df, images, labels, 
                                                                                                    test_size=0.10, stratify=labels, 
                                                                                                    random_state=rand_num)

# Then, split the remaining 90% into 80% training and 10% validation (stratified)
trainAttrX, valAttrX, trainImagesX, valImagesX, trainLabels, valLabels = train_test_split(
                                                                                        trainValAttrX, trainValImagesX, trainValLabels, 
                                                                                        test_size=0.11, 
                                                                                        stratify=trainValLabels, 
                                                                                        random_state=rand_num)

# Create the labels for training, validation, and testing
label_encoder = LabelEncoder()
trainY = label_encoder.fit_transform(trainLabels)
valY = label_encoder.transform(valLabels)
testY = label_encoder.transform(testLabels)

print("[INFO] Dataset split:")
print(f"Training set: {len(trainAttrX)} samples")
print(f"Validation set: {len(valAttrX)} samples")
print(f"Testing set: {len(testAttrX)} samples")

# Process the eye attributes data for training, validation, and testing
trainAttrX, valAttrX = glauc_mult_datasets.process_eye_attributes(df, trainAttrX, valAttrX)
trainAttrX, testAttrX = glauc_mult_datasets.process_eye_attributes(df, trainAttrX, testAttrX)

# Create the MLP and CNN models
mlp = glauc_mult_models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = glauc_mult_models.create_cnn(128, 128, 9, regress=False)  # Concatenated for greyscaled

# Create combined input
combinedInput = concatenate([mlp.output, cnn.output])

# Define the final layers
x = Dense(3, activation="softmax")(combinedInput)  # For 3 classes: glaucoma, suspected glaucoma, healthy

# Create the final model
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# Compile the model
opt = Adam(learning_rate=1e-3)
model.compile(optimizer=opt, 
              loss="sparse_categorical_crossentropy",                          
              metrics=["sparse_categorical_accuracy"])

# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint('Checkpoint_RV1.keras', monitor="val_sparse_categorical_accuracy", save_best_only=True, mode='max')

# Train the model
print("[INFO] training model...")
history = model.fit(
    x=[trainAttrX, trainImagesX], y=trainY,
    validation_data=([valAttrX, valImagesX], valY),
    epochs=250, batch_size=8, callbacks=[checkpoint])

# Save the model weights of the last epoch
model.save('run_RV1.keras')
print("model has been saved!")
print("The training has been completed.")

# Make predictions on the testing data
print("[INFO] making predictions...")
model = load_model("Checkpoint_RV1.keras")


val_loss, val_accuracy = model.evaluate([valAttrX, valImagesX], valY)
print(f"[INFO] Checkpoint Validation Accuracy: {val_accuracy:.4f}")
print(f"[INFO] Checkpoint Validation Loss: {val_loss:.4f}")

predictions = model.predict([testAttrX, testImagesX])

# Create confusion matrix
cm = confusion_matrix(testY, np.argmax(predictions, axis=1))
predicted_classes = np.argmax(predictions, axis=1)

# Calculate F1 score
f1 = f1_score(testY, predicted_classes, average='weighted')
print(f"[INFO] F1 Score: {f1:.4f}")


# Compute ROC and AUC
testY_bin = label_binarize(testY, classes=[0, 1, 2])  # Binarize the true labels for multi-class ROC
pred_prob = model.predict([testAttrX, testImagesX])  # Get predicted probabilities

# Initialize the figure for the subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Training and Validation Accuracy
axs[0, 0].plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
axs[0, 0].plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
axs[0, 0].set_title('Model Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend(loc='lower right')

# Plot Training and Validation Loss
axs[0, 1].plot(history.history['loss'], label='Training Loss')
axs[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axs[0, 1].set_title('Model Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend(loc='upper right')

# Plot Confusion MatrixRS 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, ax=axs[1, 0])
axs[1, 0].set_title("Confusion Matrix")
axs[1, 0].set_xlabel('Predicted Labels')
axs[1, 0].set_ylabel('True Labels')

# Adjust x and y ticks to reflect the classes
axs[1, 0].set_xticks(np.arange(len(label_encoder.classes_)))
axs[1, 0].set_xticklabels(label_encoder.classes_)
axs[1, 0].set_yticks(np.arange(len(label_encoder.classes_)))
axs[1, 0].set_yticklabels(label_encoder.classes_)

## Plot ROC Curve
colors = ['blue', 'red', 'green']
class_names = label_encoder.classes_  # Class names from the label encoder
for i in range(3):  # For each class (Glaucoma, Healthy, Suspected)
    fpr, tpr, _ = roc_curve(testY_bin[:, i], pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axs[1, 1].plot(fpr, tpr, color=colors[i], label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

# Plot ROC curve formatting
axs[1, 1].plot([0, 1], [0, 1], color='black', linestyle='--')  # Random classifier line
axs[1, 1].set_title('Receiver Operating Characteristic (ROC)')
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].legend(loc='lower right')



# Adjust layout
plt.tight_layout()
file_path = os.path.join(folder_path, f"RS{rand_num}plot.png")
plt.savefig(file_path)
#plt.show(block = False)
#plt.pause(2)
#plt.close()


print(f"This is Random State {rand_num}")
