# Multimodal_AI_Glaucoma
a MultiModal_AI_Glaucoma model to predict whether the combination of parameter data and images will predict a status of glaucoma (GL), glaucoma suspect (GLS), or healthy eye (HC) status
There are two models within this repository
The first model is a train test split model that seperates the data into 80% training, 10% validation and 10% testing

The second model is a 10 fold cross validation method, 8 folds for training, 1 fold for validation and 1 fold for testing.
each fold will be used as a valdiation fold within the inner fold.
each fold will be used as testing fold in the outer fold.

