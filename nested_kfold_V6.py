

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested K-Fold Cross Validation for Glaucoma Diagnosis Model (10 outer folds, 3 inner folds)
8 folds for training
1 folds for validaiton
1 fold for testing
"""


random_number = 33
folder_path = '/Users/hyang/Desktop/F1/K-fold Images/RS 33'
colors = ['blue', 'red', 'green']

import glauc_mult_datasets
import glauc_mult_models
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path/name of input dataset of eye images")
args = vars(ap.parse_args())

# Load data
inputPath = os.path.sep.join([args["dataset"], "FV8.csv"])
df = glauc_mult_datasets.load_eye_attribute(inputPath)
images = glauc_mult_datasets.load_eye_images(df, args["dataset"])
images = images / 255.0

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["Glaucoma Diagnosis"])
print("\n \n")
print("Classes:", label_encoder.classes_)
print("\n \n")

# Outer 10-Fold CV
outer_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = random_number)
outer_fold = 1
outer_results = []




print("[INFO] Starting Nested 10-Fold Cross-Validation...")
for outer_train_idx, outer_test_idx in outer_kf.split(df, labels):
    print(f"\n[INFO] Outer Fold {outer_fold}")

    train_df, test_df = df.iloc[outer_train_idx], df.iloc[outer_test_idx]
    train_images, test_images = images[outer_train_idx], images[outer_test_idx]
    train_labels, test_labels = labels[outer_train_idx], labels[outer_test_idx]

    # Inner Fold CV for tuning
    inner_kf = StratifiedKFold(n_splits=9, shuffle=True, random_state = random_number) #use 9 instead of 3 for more precise results, but will make it take way longer
    best_acc = 0
    best_model = None
    best_history = None

    for i, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(train_df, train_labels), 1):
        inner_train_df, inner_val_df = train_df.iloc[inner_train_idx], train_df.iloc[inner_val_idx]
        inner_train_images, inner_val_images = train_images[inner_train_idx], train_images[inner_val_idx]
        inner_train_labels, inner_val_labels = train_labels[inner_train_idx], train_labels[inner_val_idx]
        inner_fold = 1

        inner_train_attr, inner_val_attr = glauc_mult_datasets.process_eye_attributes(df, inner_train_df, inner_val_df)

        mlp = glauc_mult_models.create_mlp(inner_train_attr.shape[1], regress=False)
        cnn = glauc_mult_models.create_cnn(128, 128, 9, regress=False)
        combined = concatenate([mlp.output, cnn.output])
        output = Dense(3, activation="softmax")(combined)
        model = Model(inputs=[mlp.input, cnn.input], outputs=output)

        model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"])

        history = model.fit(x=[inner_train_attr, inner_train_images], y=inner_train_labels,
                            validation_data=([inner_val_attr, inner_val_images], inner_val_labels),
                            epochs=250, batch_size=8, verbose=0)

        train_acc = history.history['sparse_categorical_accuracy'][-1]
        val_acc = history.history['val_sparse_categorical_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        print(f"[INFO] Inner Fold {i}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print("\n")

        preds = model.predict([inner_val_attr, inner_val_images])
        pred_classes = np.argmax(preds, axis=1)
        

        # Plot accuracy and loss curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'Loss - Inner Fold {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['sparse_categorical_accuracy'], label='Train Acc')
        plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Acc')
        plt.title(f'Accuracy - Inner Fold {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        file_path = os.path.join(folder_path, f"Innerfold accuracy and loss curve{inner_fold}plot.png")
        plt.savefig(file_path)
        #plt.show(block = False)
        #plt.pause(2)
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(inner_val_labels, pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= label_encoder.classes_)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - Inner Fold {i}')
        file_path = os.path.join(folder_path, f"InnerFold CM{inner_fold}plot.png")
        plt.savefig(file_path)
        #plt.show(block = False)
        #plt.pause(2)
        plt.close()

        # Define class names (make sure they are in the same order as the classes in your label encoder)
        #class_names = ['Glauc', 'Healthy', 'Sus']
        class_names = label_encoder.classes_

        # AUC-ROC
        val_labels_bin = label_binarize(inner_val_labels, classes=[0, 1, 2])  #extranuous, but stop messing wit
        plt.figure()

        # Loop through each class
        for j in range(3):
            fpr, tpr, _ = roc_curve(val_labels_bin[:, j], preds[:, j])
            auc_score = roc_auc_score(val_labels_bin[:, j], preds[:, j])  # Compute AUC for each class
            plt.plot(fpr, tpr, label=f'{class_names[j]} (AUC = {auc_score:.2f})',  color=colors[j])  # Use class names in the label

        # Add a diagonal line representing random classifier performance
        plt.plot([0, 1], [0, 1], 'k--')

        # Add title and labels
        plt.title(f'ROC Curve - Inner Fold {i}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # Add legend
        plt.legend(loc='lower right')

        # Enable grid
        plt.grid(True)

        # Save the plot to a file
        file_path = os.path.join(folder_path, f"Innerfold AUC_ROC{inner_fold}plot.png")
        plt.savefig(file_path)

        # Show the plot
        #plt.show(block=False)
        #plt.pause(2)
        plt.close()

        # Increment the fold counter
        inner_fold += 1

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_history = history 
        print(f"[INFO] Highest Validation Accuracy in Inner Fold {inner_fold}: {best_acc:.4f}")
        print("\n")

    # Final Evaluation on Outer Test Fold
    test_attr = glauc_mult_datasets.process_eye_attributes(df, train_df, test_df)[1]
    preds = best_model.predict([test_attr, test_images])
    pred_classes = np.argmax(preds, axis=1)
    class_names = label_encoder.classes_

    acc = np.mean(pred_classes == test_labels)
    f1 = f1_score(test_labels, pred_classes, average='weighted')
    auc = roc_auc_score(label_binarize(test_labels, classes=[0, 1, 2]), preds, average='macro', multi_class='ovr')
    

    print(f"[RESULT] Outer Fold {outer_fold}: Test Accuracy = {acc:.4f}, F1 = {f1:.4f}, AUC = {auc:.4f}")
    print(f"[RESULT] Outer Fold {outer_fold}: Best Validation Accuracy = {best_acc:.4f}, Final Training Accuracy = {best_history.history['sparse_categorical_accuracy'][-1]:.4f}, Final Training Loss = {best_history.history['loss'][-1]:.4f}, Final Validation Loss = {best_history.history['val_loss'][-1]:.4f}")
    print("[RESULT] Classification Report - Outer Fold {}:\n".format(outer_fold))
    print(classification_report(test_labels, pred_classes, target_names=['Glaucoma', 'Sus', 'Healthy']))
    outer_results.append((acc, f1, auc))

    # Plot final accuracy and loss curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(best_history.history['loss'], label='Train Loss')
    plt.plot(best_history.history['val_loss'], label='Val Loss')
    plt.title(f'Final Loss - Outer Fold {outer_fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(best_history.history['sparse_categorical_accuracy'], label='Train Acc')
    plt.plot(best_history.history['val_sparse_categorical_accuracy'], label='Val Acc')
    plt.title(f'Final Accuracy - Outer Fold {outer_fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    file_path = os.path.join(folder_path, f"{outer_fold} accuracy and lossplot Outerfold.png")
    plt.savefig(file_path)
    #plt.show(block = False)
    #plt.pause(2)
    plt.close()

    # Confusion Matrix for outer fold
    cm = confusion_matrix(test_labels, pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - Outer Fold {outer_fold}')
    file_path = os.path.join(folder_path, f"{outer_fold}_CM_plot Outerfold.png")
    plt.savefig(file_path)
    #plt.show(block = False)
    #plt.pause(2)
    plt.close()
    

    # AUC-ROC for outer fold
    test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2])
    plt.figure()
    for j in range(3):
        fpr, tpr, _ = roc_curve(test_labels_bin[:, j], preds[:, j])
        plt.plot(fpr, tpr, label=f'{class_names[j]} (AUC = {roc_auc_score(test_labels_bin[:, j], preds[:, j]):.2f})',  color=colors[j])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - Outer Fold {outer_fold}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    file_path = os.path.join(folder_path, f"{outer_fold}AUC_ROC plot Outerfold.png")
    plt.savefig(file_path)
    #plt.show(block = False)
    #plt.pause(2)
    plt.close()
    
    model.save(f"Outerfold_{outer_fold}_RandState_{random_number}.keras")

    outer_fold += 1

# Summary
accs, f1s, aucs = zip(*outer_results)
print("\n[SUMMARY] Nested 10-Fold CV Results:")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
